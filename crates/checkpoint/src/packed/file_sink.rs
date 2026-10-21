// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded export of trained low-precision models for its clients. If your
// team needs expertise in writing checkpoints in the format your serving stack loads, without ever holding a
// tensor in full precision, then you can procure our services by sending an email to info@swedishembedded.com.

//! [`SafetensorsSink`]: a [`TensorSink`] that writes a safetensors file tile by tile in the encoding each
//! descriptor declares, so a trained model leaves in bf16 / FP8 / f32 with the host holding one tile.
//!
//! The file layout is exactly what [`super::PackedReader`] infers an encoding from: the payload under the
//! tensor's name with the dtype of its element format, and each scale level as a sidecar tensor named
//! `<name><suffix>` (`weight_scale_inv` for tiled FP8) with the dtype of the scale format and the shape of its
//! grid. The header is written up front from the descriptors, tiles are placed at their byte offsets, and the
//! file appears under its final name only on [`SafetensorsSink::commit`], so a failed export leaves nothing
//! half written.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use super::codec::Tile;
use super::encoding::{Codebook, Element, FloatFormat, ScaleGranularity, TensorDescriptor};
use super::reader::PACKED_METADATA_KEY;
use super::sink::TensorSink;
use crate::st::CONFIG_KEY;

fn dtype_tag(f: FloatFormat) -> Result<&'static str, String> {
    Ok(match f {
        FloatFormat::F32 => "F32",
        FloatFormat::F16 => "F16",
        FloatFormat::Bf16 => "BF16",
        FloatFormat::E4M3Fn => "F8_E4M3",
        FloatFormat::E5M2 => "F8_E5M2",
        FloatFormat::E2M1 | FloatFormat::E8M0 => return Err(format!("{f:?} has no safetensors dtype in this writer")),
    })
}

/// The safetensors dtype and shape a descriptor's payload is stored under. Float elements keep their own dtype and
/// the logical shape. brain's packed integer layouts (int8 four to a word, NF4 eight to a word, little endian)
/// are `U32` words along the innermost axis, the convention of its int8-native checkpoints; the file's `brain.packed`
/// metadata declares the descriptor, so `PackedReader` needs no `with_descriptor` for it.
fn payload_layout(d: &TensorDescriptor) -> Result<(&'static str, Vec<u64>), String> {
    let packed = |per_word: u64| -> Result<(&'static str, Vec<u64>), String> {
        let mut shape = d.shape.clone();
        let inner = shape.last_mut().ok_or_else(|| format!("{}: a scalar cannot be word-packed", d.name))?;
        if !(*inner).is_multiple_of(per_word) {
            return Err(format!("{}: innermost axis {inner} is not a multiple of {per_word} for word packing", d.name));
        }
        *inner /= per_word;
        Ok(("U32", shape))
    };
    match d.encoding.element {
        Element::Float(f) => Ok((dtype_tag(f)?, d.shape.clone())),
        Element::Int { bits: 8, signed: true } => packed(4),
        Element::Codebook(Codebook::Nf4) => packed(8),
        _ => Err(format!("{}: {} cannot be written as safetensors here", d.name, d.encoding)),
    }
}

/// Where one tensor's pieces live in the file, relative to the start of the data blob.
struct Placed {
    payload: u64,
    scales: Vec<u64>,
    written: u64,
}

pub struct SafetensorsSink {
    file: File,
    tmp: PathBuf,
    path: PathBuf,
    blob_start: u64,
    placed: HashMap<String, Placed>,
    order: Vec<String>,
    finished: usize,
}

impl SafetensorsSink {
    /// Plan the file for `descs` (every tensor of the export, in the order it will arrive) and write its header.
    /// `config` is stored under the brain config key so the file opens as a brain checkpoint.
    pub fn create(path: &Path, descs: &[TensorDescriptor], config: &Value) -> io::Result<SafetensorsSink> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
        let mut header = Map::new();
        let mut off = 0u64;
        let mut placed = HashMap::new();
        let mut order = Vec::new();
        for d in descs {
            d.validate().map_err(bad)?;
            if placed.contains_key(&d.name) {
                return Err(bad(format!("{}: listed twice", d.name)));
            }
            let nbytes = d.payload_bytes();
            let (tag, shape) = payload_layout(d).map_err(bad)?;
            header.insert(d.name.clone(), json!({"dtype": tag, "shape": shape, "data_offsets": [off, off + nbytes]}));
            let payload = off;
            off += nbytes;
            let mut scales = Vec::new();
            for (l, level) in d.encoding.scales.iter().enumerate() {
                let (shape, n): (Vec<u64>, u64) = match level.granularity {
                    ScaleGranularity::Tile2d { rows, cols } => {
                        let grid = vec![d.shape[0].div_ceil(rows as u64), d.shape[1].div_ceil(cols as u64)];
                        let n = grid.iter().product();
                        (grid, n)
                    }
                    _ => (vec![d.scale_numel(l)], d.scale_numel(l)),
                };
                let bytes = n * level.format.bits() as u64 / 8;
                header.insert(format!("{}{}", d.name, level.suffix), json!({"dtype": dtype_tag(level.format).map_err(bad)?, "shape": shape, "data_offsets": [off, off + bytes]}));
                scales.push(off);
                off += bytes;
            }
            placed.insert(d.name.clone(), Placed { payload, scales, written: 0 });
            order.push(d.name.clone());
        }
        // What the dtype alone cannot say (NF4 codes in words, FP8 scale grids) rides in the file itself, so the packed
        // reader needs no declaration from the caller (`PACKED_METADATA_KEY`).
        let mut packed = Map::new();
        for d in descs {
            if let Some(spec) = d.encoding.spec() {
                packed.insert(d.name.clone(), json!({"shape": d.shape, "encoding": spec}));
            }
        }
        let mut meta = Map::new();
        meta.insert(CONFIG_KEY.to_string(), Value::String(serde_json::to_string(config)?));
        meta.insert(PACKED_METADATA_KEY.to_string(), Value::String(Value::Object(packed).to_string()));
        header.insert("__metadata__".to_string(), Value::Object(meta));
        let mut hbytes = serde_json::to_vec(&Value::Object(header))?;
        // Pad to a multiple of 8 so every 4-byte-wide tensor lands word aligned (see StWriter).
        while hbytes.len() % 8 != 0 {
            hbytes.push(b' ');
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("safetensors.tmp");
        let file = File::create(&tmp)?;
        file.write_all_at(&(hbytes.len() as u64).to_le_bytes(), 0)?;
        file.write_all_at(&hbytes, 8)?;
        Ok(SafetensorsSink { file, tmp, path: path.to_path_buf(), blob_start: 8 + hbytes.len() as u64, placed, order, finished: 0 })
    }

    /// Publish the file under its final name. Errors when a planned tensor never arrived.
    pub fn commit(self) -> io::Result<()> {
        if self.finished != self.order.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{} of {} planned tensors were written; the file is not committed", self.finished, self.order.len())));
        }
        self.file.sync_all()?;
        std::fs::rename(&self.tmp, &self.path)
    }

    fn place(&mut self, name: &str) -> Result<&mut Placed, String> {
        self.placed.get_mut(name).ok_or_else(|| format!("{name}: not in the export plan"))
    }
}

impl Drop for SafetensorsSink {
    fn drop(&mut self) {
        // After a successful commit the temporary name no longer exists; this only cleans up an abandoned export.
        let _ = std::fs::remove_file(&self.tmp);
    }
}

impl TensorSink for SafetensorsSink {
    fn begin(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        self.place(&dst.name)?.written = 0;
        Ok(())
    }

    fn write_tile(&mut self, dst: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
        let blob = self.blob_start;
        let p = self.place(&dst.name)?;
        let (payload_at, scale_at) = (p.payload, p.scales.clone());
        p.written += tile.elems.end - tile.elems.start;
        let pr = dst.payload_range(&tile.elems);
        let err = |e: io::Error| format!("{}: {e}", dst.name);
        self.file.write_all_at(tile.payload, blob + payload_at + pr.start).map_err(err)?;
        for (l, bytes) in tile.scales.iter().enumerate() {
            let w = dst.encoding.scales[l].format.bits() as u64 / 8;
            let r = dst.scale_range(l, &tile.elems);
            self.file.write_all_at(bytes, blob + scale_at[l] + r.start * w).map_err(err)?;
        }
        Ok(())
    }

    fn finish(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        let written = self.place(&dst.name)?.written;
        if written != dst.numel() {
            return Err(format!("{}: {written} of {} elements written", dst.name, dst.numel()));
        }
        self.finished += 1;
        Ok(())
    }

    fn abort(&mut self, dst: &TensorDescriptor) {
        if let Some(p) = self.placed.get_mut(&dst.name) {
            p.written = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::codec::{CodecRegistry, EncodeCtx, TileBuf};
    use crate::packed::{PackedReader, PackedTensorReader, QuantEncoding};

    /// Spec: what the sink writes, the packed reader reads back as the same descriptors and the same bytes
    /// (payload and scale sidecars) the encoder produced, tile by tile, for plain and 2-D-tile-scaled tensors.
    #[test]
    fn a_written_file_reads_back_as_the_encoded_bytes() {
        let dir = std::env::temp_dir().join(format!("brain-st-sink-{}", std::process::id()));
        let path = dir.join("m.safetensors");
        let descs = vec![
            TensorDescriptor::new("a.weight", vec![8, 16], QuantEncoding::bf16()),
            TensorDescriptor::new("q.weight", vec![4, 64], QuantEncoding::int8_blocked(32)),
            TensorDescriptor::new("n4.weight", vec![4, 64], QuantEncoding::nf4_blocked(32)),
            TensorDescriptor::new("b.weight", vec![16, 32], QuantEncoding::fp8_e4m3_tiled(16)),
            TensorDescriptor::new("n.weight", vec![16], QuantEncoding::f32()),
        ];
        let reg = CodecRegistry::reference();
        let mut sink = SafetensorsSink::create(&path, &descs, &json!({"d": 1})).unwrap();
        let mut want: Vec<(Vec<u8>, Vec<Vec<u8>>)> = Vec::new();
        for d in &descs {
            let vals: Vec<f32> = (0..d.numel()).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
            sink.begin(d).unwrap();
            let q = d.encode_quantum().max(32);
            let mut full = TileBuf::default();
            let mut payload = vec![0u8; d.payload_bytes() as usize];
            let mut scales: Vec<Vec<u8>> = (0..d.encoding.scales.len()).map(|l| vec![0; d.scale_bytes(l) as usize]).collect();
            let mut start = 0u64;
            while start < d.numel() {
                let end = (start + q).min(d.numel());
                let e = start..end;
                reg.encoder(&d.encoding).unwrap().encode_tile(d, &e, &vals[start as usize..end as usize], &EncodeCtx::default(), &mut full).unwrap();
                let t = Tile { elems: e.clone(), payload: &full.payload, scales: full.scales.iter().map(|s| s.as_slice()).collect() };
                sink.write_tile(d, &t).unwrap();
                let pr = d.payload_range(&e);
                payload[pr.start as usize..pr.end as usize].copy_from_slice(&full.payload);
                for (l, s) in full.scales.iter().enumerate() {
                    let w = d.encoding.scales[l].format.bits() as usize / 8;
                    let r = d.scale_range(l, &e);
                    scales[l][r.start as usize * w..r.end as usize * w].copy_from_slice(s);
                }
                start = end;
            }
            sink.finish(d).unwrap();
            want.push((payload, scales));
        }
        sink.commit().unwrap();
        let r = PackedReader::open(&path).unwrap();
        let r = descs.iter().filter(|d| !matches!(d.encoding.element, Element::Float(_))).fold(r, |r, d| r.with_descriptor(d.clone()));
        assert_eq!(r.names().len(), 5, "sidecars are not listed as tensors");
        for (d, (payload, scales)) in descs.iter().zip(&want) {
            assert_eq!(&r.descriptor(&d.name).unwrap(), d);
            let mut buf = TileBuf::default();
            r.read_blocks(d, &(0..d.numel()), &mut buf).unwrap();
            assert_eq!(&buf.payload, payload, "{} payload", d.name);
            assert_eq!(&buf.scales, scales, "{} scales", d.name);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Spec: a file with a missing tensor is not committed, and an abandoned export leaves no temporary file.
    #[test]
    fn an_incomplete_export_is_not_published() {
        let dir = std::env::temp_dir().join(format!("brain-st-sink-inc-{}", std::process::id()));
        let path = dir.join("m.safetensors");
        let descs = vec![TensorDescriptor::new("a", vec![4], QuantEncoding::f32())];
        let sink = SafetensorsSink::create(&path, &descs, &json!({})).unwrap();
        assert!(sink.commit().is_err());
        assert!(!path.exists());
        assert!(std::fs::read_dir(&dir).unwrap().next().is_none(), "the temporary file is removed");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
