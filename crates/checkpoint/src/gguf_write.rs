// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A from-scratch GGUF v3 writer -- the mirror of [`crate::gguf`]'s reader.
//! Produces files [`crate::gguf::MmapGguf`]/[`crate::gguf::read`] read
//! back byte-identically (same magic, KV encoding, tensor-info layout, and
//! alignment convention), so anything this crate quantizes with
//! [`crate::quant`] round-trips through this crate's own reader with no
//! external tool involved anywhere in the loop.

use std::io::{self, Write};

use crate::gguf::GgufValue;

/// One tensor to write: `name`, its shape in **torch order** (reversed to
/// GGUF's fastest-varying-first `ne` on write, mirroring the reader's
/// `ne.reverse()` on read), its ggml type id, and its already-encoded bytes
/// (raw F32 LE for an unquantized tensor, or the output of
/// [`crate::quant::quantize`] for a quantized one).
pub struct TensorOut {
    pub name: String,
    pub shape: Vec<usize>,
    pub ty: u32,
    pub data: Vec<u8>,
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// The GGUF metadata value-type tag for one [`GgufValue`] (mirrors
/// `Cursor::value`'s read-side `match` in `gguf.rs`, in reverse).
fn value_type_tag(v: &GgufValue) -> u32 {
    match v {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

/// The element-type tag written once for a whole `Array` (every element of
/// a GGUF array shares one type); `0` (U8) for an empty array, since the
/// reader never inspects an empty array's declared element type.
fn array_element_type_tag(items: &[GgufValue]) -> u32 {
    items.first().map(value_type_tag).unwrap_or(0)
}

fn write_value_payload(out: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::U8(x) => out.push(*x),
        GgufValue::I8(x) => out.push(*x as u8),
        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Bool(x) => out.push(*x as u8),
        GgufValue::String(s) => write_string(out, s),
        GgufValue::Array(items) => {
            out.extend_from_slice(&array_element_type_tag(items).to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_value_payload(out, item);
            }
        }
        GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
    }
}

/// One tensor's identity and encoded byte length, declared BEFORE its bytes
/// exist. A GGUF header carries every tensor's offset, so the whole plan has
/// to be known up front -- but only the *sizes*, never the data. That is what
/// lets [`Writer`] stream a checkpoint far larger than host RAM.
pub struct TensorPlan {
    pub name: String,
    pub shape: Vec<usize>,
    pub ty: u32,
    /// Encoded length in bytes: `numel / block_elems * block_bytes` for the
    /// declared `ty`. [`Writer::write_tensor`] rejects a body that disagrees.
    pub nbytes: usize,
}

/// An incremental GGUF v3 writer: the header (KV + every tensor's info and
/// offset) is emitted from the plan on [`create`](Writer::create), then each
/// tensor's already-encoded bytes are appended in plan order and dropped by
/// the caller. Peak host allocation is therefore ONE tensor, not the whole
/// output -- the same discipline `weightio::StWriter` applies to safetensors,
/// and the reason a multi-gigabyte quantization does not need a
/// multi-gigabyte `Vec` per tensor kept alive until the end.
///
/// [`write`] is this type used eagerly, so there is exactly one
/// implementation of the container format.
pub struct Writer {
    file: io::BufWriter<std::fs::File>,
    tmp: String,
    path: String,
    plan: Vec<TensorPlan>,
    next: usize,
    alignment: usize,
}

impl Writer {
    /// Plan and open `path`: writes `kv` in the given order, then a tensor
    /// info block for every entry of `plan` in the given order, then pads to
    /// the data start. `alignment` (a power of two, `>= 1`) is the padding
    /// between the header and the tensor data and between consecutive
    /// tensors -- pass `32` to match the reader's default when the file
    /// declares none.
    pub fn create(path: &str, kv: &[(String, GgufValue)], plan: Vec<TensorPlan>, alignment: usize) -> io::Result<Writer> {
        let alignment = alignment.max(1);
        let mut header: Vec<u8> = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3u32.to_le_bytes()); // version
        header.extend_from_slice(&(plan.len() as u64).to_le_bytes());
        header.extend_from_slice(&(kv.len() as u64).to_le_bytes());

        for (key, val) in kv {
            write_string(&mut header, key);
            header.extend_from_slice(&value_type_tag(val).to_le_bytes());
            write_value_payload(&mut header, val);
        }

        // Tensor infos: name, ndim, ne[ndim] (torch shape reversed to GGUF's
        // fastest-varying-first), type, offset (relative to the aligned data
        // start, itself aligned so each tensor's data begins on an
        // `alignment` boundary -- mirrors every real GGUF writer's
        // convention, though the reader here only requires the header's
        // `data_start` rounding).
        let mut offset: u64 = 0;
        for t in &plan {
            write_string(&mut header, &t.name);
            header.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for &dim in t.shape.iter().rev() {
                header.extend_from_slice(&(dim as u64).to_le_bytes());
            }
            header.extend_from_slice(&t.ty.to_le_bytes());
            header.extend_from_slice(&offset.to_le_bytes());
            offset += (t.nbytes as u64).div_ceil(alignment as u64) * alignment as u64;
        }

        let data_start = header.len().div_ceil(alignment) * alignment;
        header.resize(data_start, 0);

        let tmp = format!("{path}.tmp");
        let mut file = io::BufWriter::new(std::fs::File::create(&tmp)?);
        file.write_all(&header)?;
        Ok(Writer { file, tmp, path: path.to_string(), plan, next: 0, alignment })
    }

    /// Append the next planned tensor's encoded bytes. `name` and
    /// `data.len()` must match the plan entry at this position exactly -- a
    /// misordered or mis-sized body would silently shift every subsequent
    /// tensor off the offsets already committed to the header, which the
    /// reader cannot detect.
    pub fn write_tensor(&mut self, name: &str, data: &[u8]) -> io::Result<()> {
        let want = self.plan.get(self.next).ok_or_else(|| {
            io::Error::other(format!("gguf_write: tensor '{name}' is past the end of a {}-tensor plan", self.plan.len()))
        })?;
        if want.name != name {
            return Err(io::Error::other(format!("gguf_write: expected tensor '{}' at index {}, got '{name}'", want.name, self.next)));
        }
        if want.nbytes != data.len() {
            return Err(io::Error::other(format!("gguf_write: tensor '{name}' planned {} bytes, got {}", want.nbytes, data.len())));
        }
        self.file.write_all(data)?;
        let pad = (data.len() as u64).div_ceil(self.alignment as u64) * self.alignment as u64 - data.len() as u64;
        if pad > 0 {
            self.file.write_all(&vec![0u8; pad as usize])?;
        }
        self.next += 1;
        Ok(())
    }

    /// Flush and atomically move the temporary into place. Fails if any
    /// planned tensor was never written -- the header already promises those
    /// bytes exist, so a short file is a corrupt file, not a partial one.
    pub fn finish(mut self) -> io::Result<()> {
        if self.next != self.plan.len() {
            return Err(io::Error::other(format!("gguf_write: {} of {} planned tensors written", self.next, self.plan.len())));
        }
        self.file.flush()?;
        drop(self.file);
        std::fs::rename(&self.tmp, &self.path)
    }
}

/// Write a GGUF v3 file: `kv` in the given order, then `tensors` in the
/// given order. `alignment` (must be a power of two, `>= 1`) sets
/// `general.alignment`-equivalent padding between the header and the tensor
/// data, and between consecutive tensors -- pass `32` to match the reader's
/// default when the file declares none. Every tensor's `data` length must
/// already match its declared shape's element count under `ty`'s on-disk
/// encoding (this function does not validate that -- [`crate::quant`] and
/// the raw-F32 path both produce exactly the right length by construction).
///
/// This is the eager form of [`Writer`], for a model small enough to hold
/// whole. [`crate::quantize`] uses the streaming form instead.
pub fn write(path: &str, kv: &[(String, GgufValue)], tensors: &[TensorOut], alignment: usize) -> io::Result<()> {
    let plan = tensors
        .iter()
        .map(|t| TensorPlan { name: t.name.clone(), shape: t.shape.clone(), ty: t.ty, nbytes: t.data.len() })
        .collect();
    let mut w = Writer::create(path, kv, plan, alignment)?;
    for t in tensors {
        w.write_tensor(&t.name, &t.data)?;
    }
    w.finish()
}

/// Write a split GGUF: one file per entry of `tensors`, at
/// `<dir>/<base>-NNNNN-of-MMMMM.gguf` (part 1 first) - [`crate::gguf::
/// MmapGguf::open`]'s split path reads this same convention back. `kv` is
/// written to every part unchanged; each part additionally gets the three
/// keys that path validates: `split.no` (0-based), `split.count`, and
/// `split.tensors.count` (the sum of every part's tensor count, identical
/// on every part - real split writers duplicate it rather than reserving it
/// for part 1 alone, so this does too). Returns part 1's path, the one
/// `MmapGguf::open` should be given - it locates every sibling from it.
pub fn write_split(dir: &str, base: &str, kv: &[(String, GgufValue)], tensors: &[Vec<TensorOut>], alignment: usize) -> io::Result<String> {
    let count = tensors.len() as u32;
    let width = count.to_string().len().max(5);
    let total_tensors: u64 = tensors.iter().map(|t| t.len() as u64).sum();
    let mut first_path = None;
    for (i, part_tensors) in tensors.iter().enumerate() {
        let part = i as u32 + 1;
        let fname = crate::split::split_sibling(base, part, count, width, "gguf");
        let path = format!("{dir}/{fname}");
        let mut part_kv = kv.to_vec();
        part_kv.push(("split.no".to_string(), GgufValue::U32(i as u32)));
        part_kv.push(("split.count".to_string(), GgufValue::U32(count)));
        part_kv.push(("split.tensors.count".to_string(), GgufValue::U64(total_tensors)));
        write(&path, &part_kv, part_tensors, alignment)?;
        first_path.get_or_insert(path);
    }
    Ok(first_path.expect("write_split: at least one part"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{self, T_F32, T_Q8_0};

    fn scratch_path(name: &str) -> String {
        std::env::temp_dir().join(name).to_string_lossy().into_owned()
    }

    #[test]
    fn writes_a_file_the_readers_own_parser_reads_back() {
        let path = scratch_path("gguf-write-test-basic.gguf");
        let kv = vec![
            ("general.architecture".to_string(), GgufValue::String("qwen".to_string())),
            ("general.name".to_string(), GgufValue::String("test-model".to_string())),
            ("qwen.context_length".to_string(), GgufValue::U32(2048)),
            ("general.file_type".to_string(), GgufValue::U32(0)),
        ];
        let data: Vec<u8> = (0..16.0f32 as i32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let tensors = vec![TensorOut { name: "w".to_string(), shape: vec![4, 4], ty: T_F32, data }];
        write(&path, &kv, &tensors, 32).unwrap();

        let model = gguf::load_gguf(&path).unwrap();
        assert_eq!(model.shapes["w"], vec![4, 4]);
        assert_eq!(model.tensors["w"], (0..16).map(|i| i as f32).collect::<Vec<f32>>());
        assert_eq!(model.kv["general.architecture"].as_str(), Some("qwen"));
        assert_eq!(model.kv["general.name"].as_str(), Some("test-model"));
        let card = model.model_card();
        assert_eq!(card.family, "qwen");
        assert_eq!(card.display_name.as_deref(), Some("test-model"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn writes_a_quantized_tensor_that_the_reader_dequantizes_correctly() {
        let path = scratch_path("gguf-write-test-quant.gguf");
        let raw: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let bytes = crate::quant::quantize(T_Q8_0, &raw).unwrap();
        let tensors = vec![TensorOut { name: "w".to_string(), shape: vec![2, 32], ty: T_Q8_0, data: bytes }];
        write(&path, &[], &tensors, 32).unwrap();

        let model = gguf::load_gguf(&path).unwrap();
        assert_eq!(model.shapes["w"], vec![2, 32]);
        let decoded = &model.tensors["w"];
        assert_eq!(decoded.len(), 64);
        let (rmse, cosine) = crate::quant::round_trip_stats(T_Q8_0, &raw).unwrap();
        assert!(cosine > 0.999, "cosine {cosine}");
        assert!(rmse < 0.01, "rmse {rmse}");
    }

    #[test]
    fn multiple_tensors_and_an_array_kv_round_trip() {
        let path = scratch_path("gguf-write-test-multi.gguf");
        let kv = vec![(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(vec![GgufValue::String("a".to_string()), GgufValue::String("bb".to_string()), GgufValue::String("ccc".to_string())]),
        )];
        let tensors = vec![
            TensorOut { name: "a".to_string(), shape: vec![2], ty: T_F32, data: vec![0u8; 8] },
            TensorOut { name: "b".to_string(), shape: vec![3, 5], ty: T_F32, data: vec![0u8; 60] },
        ];
        write(&path, &kv, &tensors, 32).unwrap();

        let model = gguf::load_gguf(&path).unwrap();
        assert_eq!(model.shapes["a"], vec![2]);
        assert_eq!(model.shapes["b"], vec![3, 5]);
        match &model.kv["tokenizer.ggml.tokens"] {
            GgufValue::Array(items) => {
                let strs: Vec<&str> = items.iter().filter_map(|v| v.as_str()).collect();
                assert_eq!(strs, vec!["a", "bb", "ccc"]);
            }
            other => panic!("expected an array, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn non_default_alignment_still_round_trips() {
        let path = scratch_path("gguf-write-test-align.gguf");
        let tensors = vec![TensorOut { name: "w".to_string(), shape: vec![3], ty: T_F32, data: vec![0u8; 12] }];
        write(&path, &[], &tensors, 64).unwrap();
        let model = gguf::load_gguf(&path).unwrap();
        assert_eq!(model.shapes["w"], vec![3]);
        std::fs::remove_file(&path).ok();
    }
}
