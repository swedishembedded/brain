// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading of quantized HuggingFace checkpoints for its clients. If
// your team needs expertise in serving mixture-of-experts releases as published then you can procure our services by
// sending an email to info@swedishembedded.com.

//! [`SlicedReader`]: named sub-tensors of stored tensors, read in place.
//!
//! HuggingFace mixture-of-experts releases store every expert of a layer as one fused 3-D tensor
//! (`mlp.experts.gate_up_proj` `[experts, 2 * ff, d]`), where a model wants one 2-D matrix per expert. A view names a
//! contiguous element range of a stored tensor as a tensor of its own, so the load reads the file's bytes where they
//! are: no copy of the fused tensor is made and nothing is rewritten on disk.
//!
//! A view of a scaled or sub-byte tensor (NVFP4, MXFP8, block-scaled int) carries its scales with it: it must cover
//! whole rows of the source with the source's row length, so every block and row scale it needs is its own, and a
//! per-slab level (one scale per expert of a stacked tensor) becomes the view's per-tensor level when the view lies
//! inside one slab. [`SlicedReader::with_slab_views`] cuts a stacked `[experts, rows, cols]` tensor into one 2-D view
//! per expert this way, each a matrix in the encoding of a single stored expert (NVFP4 with its own f32 global).
//! Rows keep their stored order: a gate/up projection stored row-interleaved stays interleaved, and the model declares
//! that layout to the kernels that consume it. 2-D tile scales are refused (a row range does not own its tiles).

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use super::codec::TileBuf;
use super::encoding::{ScaleGranularity, TensorDescriptor};
use super::reader::PackedTensorReader;

struct View {
    source: TensorDescriptor,
    elems: Range<u64>,
    shape: Vec<u64>,
}

/// A [`PackedTensorReader`] that serves `inner`'s tensors and, besides them, named views of element ranges.
pub struct SlicedReader {
    inner: Box<dyn PackedTensorReader>,
    views: HashMap<String, View>,
    consumed: HashSet<String>,
}

impl SlicedReader {
    pub fn new(inner: Box<dyn PackedTensorReader>) -> SlicedReader {
        SlicedReader { inner, views: HashMap::new(), consumed: HashSet::new() }
    }

    /// Add the tensor `name` of `shape`: elements `elems` of stored tensor `source`. The source stops being listed
    /// by [`PackedTensorReader::names`] (its views stand for it).
    pub fn with_view(mut self, name: &str, source: &str, elems: Range<u64>, shape: Vec<u64>) -> Result<SlicedReader, String> {
        let source = self.inner.descriptor(source)?;
        if elems.end > source.numel() || elems.start >= elems.end || shape.iter().product::<u64>() != elems.end - elems.start {
            return Err(format!("view {name}: elements {elems:?} of {} ({} elements) do not form shape {shape:?}", source.name, source.numel()));
        }
        let encoding = view_encoding(name, &source, &elems, &shape)?;
        let view = View { source, elems, shape };
        TensorDescriptor::new(name, view.shape.clone(), encoding).validate()?;
        self.consumed.insert(view.source.name.clone());
        self.views.insert(name.to_string(), view);
        Ok(self)
    }

    /// One 2-D view per index of `source`'s leading axis (`[slabs, rows, cols]` -> `slabs` views of `[rows, cols]`),
    /// named `name(slab)`: the experts of a stacked mixture-of-experts tensor as matrices of their own, scales
    /// included.
    pub fn with_slab_views(mut self, source: &str, name: impl Fn(u64) -> String) -> Result<SlicedReader, String> {
        let desc = self.inner.descriptor(source)?;
        if desc.shape.len() < 3 {
            return Err(format!("{source}: slab views need a stacked tensor of rank 3 or more, got shape {:?}", desc.shape));
        }
        let slab = desc.slab_numel();
        for e in 0..desc.shape[0] {
            self = self.with_view(&name(e), source, e * slab..(e + 1) * slab, desc.shape[1..].to_vec())?;
        }
        Ok(self)
    }
}

/// The encoding of a view of `source`'s elements `elems`: the source's own, with a per-slab level narrowed to the one
/// slab the view lies in. Refused when a scale or a packed byte would be shared with elements outside the view.
fn view_encoding(name: &str, source: &TensorDescriptor, elems: &Range<u64>, shape: &[u64]) -> Result<super::encoding::QuantEncoding, String> {
    let mut enc = source.encoding.clone();
    let geometry = enc.block_geometry().elems;
    if !elems.start.is_multiple_of(geometry) || !(elems.end - elems.start).is_multiple_of(geometry) {
        return Err(format!("view {name}: elements {elems:?} of {} split a {geometry}-element payload block of {}", source.name, enc));
    }
    if enc.scales.is_empty() {
        return Ok(enc);
    }
    let inner = *source.shape.last().unwrap_or(&1);
    if shape.last() != Some(&inner) || !elems.start.is_multiple_of(inner) || !(elems.end - elems.start).is_multiple_of(inner) {
        return Err(format!("view {name}: a view of scaled {} ({}) must cover whole rows of {inner} elements, got {elems:?} as {shape:?}", source.name, enc));
    }
    for level in &mut enc.scales {
        match level.granularity {
            ScaleGranularity::Tensor | ScaleGranularity::Block(_) | ScaleGranularity::Row => {}
            ScaleGranularity::Slab => {
                let slab = source.slab_numel();
                if elems.start / slab != (elems.end - 1) / slab {
                    return Err(format!("view {name}: elements {elems:?} of {} cross a slab and its per-slab scale", source.name));
                }
                level.granularity = ScaleGranularity::Tensor;
            }
            ScaleGranularity::Tile2d { .. } => return Err(format!("view {name}: {} has 2-D tile scales, which a row range does not own", source.name)),
        }
    }
    Ok(enc)
}

impl PackedTensorReader for SlicedReader {
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.inner.names().into_iter().filter(|n| !self.consumed.contains(n)).collect();
        names.extend(self.views.keys().cloned());
        names
    }

    fn descriptor(&self, name: &str) -> Result<TensorDescriptor, String> {
        match self.views.get(name) {
            Some(v) => Ok(TensorDescriptor::new(name, v.shape.clone(), view_encoding(name, &v.source, &v.elems, &v.shape)?)),
            None => self.inner.descriptor(name),
        }
    }

    fn read_blocks(&self, desc: &TensorDescriptor, elems: &Range<u64>, dst: &mut TileBuf) -> Result<(), String> {
        match self.views.get(&desc.name) {
            Some(v) => self.inner.read_blocks(&v.source, &(v.elems.start + elems.start..v.elems.start + elems.end), dst),
            None => self.inner.read_blocks(desc, elems, dst),
        }
    }

    fn release(&self, desc: &TensorDescriptor, elems: &Range<u64>) {
        match self.views.get(&desc.name) {
            Some(v) => self.inner.release(&v.source, &(v.elems.start + elems.start..v.elems.start + elems.end)),
            None => self.inner.release(desc, elems),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::testfix::{bf16_bytes, bytes, write_safetensors, RawTensor, TmpDir};
    use crate::packed::{PackedReader, QuantEncoding};

    #[test]
    fn views_of_a_fused_tensor_read_their_own_bytes_in_place() {
        let dir = TmpDir::new("slice");
        let fused: Vec<f32> = (0..3 * 4 * 2).map(|i| i as f32).collect(); // [3 experts, 4 rows, 2 cols]
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "fused", dtype: "BF16", shape: vec![3, 4, 2], bytes: bf16_bytes(&fused) },
                RawTensor { name: "q", dtype: "U8", shape: vec![4], bytes: bytes(1, 4) },
            ],
        );
        let r = SlicedReader::new(Box::new(PackedReader::open(&dir.join("m.safetensors")).unwrap()))
            .with_view("e1.gate", "fused", 8..16, vec![4, 2])
            .unwrap();
        let mut names = r.names();
        names.sort();
        assert_eq!(names, ["e1.gate", "q"], "the fused source is replaced by its views");
        let d = r.descriptor("e1.gate").unwrap();
        assert_eq!((d.shape.clone(), d.encoding.clone()), (vec![4, 2], QuantEncoding::bf16()));
        let mut buf = TileBuf::default();
        r.read_blocks(&d, &(2..6), &mut buf).unwrap();
        assert_eq!(buf.payload, bf16_bytes(&fused[10..14]), "tile 2..6 of the view is elements 10..14 of the fused tensor");
        assert!(SlicedReader::new(Box::new(PackedReader::open(&dir.join("m.safetensors")).unwrap())).with_view("bad", "fused", 20..30, vec![10]).is_err());
    }

    /// Spec: per-expert views of a stacked NVFP4 tensor (per-expert global) and of a stacked MXFP8 tensor are 2-D
    /// matrices in the single-expert encoding, and decode to exactly the stacked tensor's slab; a view that would share
    /// a scale with elements outside it is refused.
    #[test]
    fn per_expert_views_of_stacked_scaled_tensors_carry_their_own_scales() {
        use crate::packed::codec::{CodecRegistry, Tile};
        use crate::packed::testfix::{e4m3_bytes, f32_bytes};
        let dir = TmpDir::new("slabviews");
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "e", dtype: "U8", shape: vec![3, 4, 16], bytes: bytes(1, 192) },
                RawTensor { name: "e_scale", dtype: "F8_E4M3", shape: vec![3, 4, 2], bytes: e4m3_bytes(2, 24) },
                RawTensor { name: "e_scale_2", dtype: "F32", shape: vec![3], bytes: f32_bytes(&[0.5, 0.25, 3.0]) },
            ],
        );
        let reg = CodecRegistry::reference();
        let decode = |r: &dyn PackedTensorReader, d: &TensorDescriptor, elems: Range<u64>| {
            let mut buf = TileBuf::default();
            r.read_blocks(d, &elems, &mut buf).unwrap();
            let tile = Tile { elems, payload: &buf.payload, scales: buf.scales.iter().map(Vec::as_slice).collect() };
            let mut out = Vec::new();
            reg.decoder(&d.encoding).unwrap().decode_tile(d, &tile, &mut out).unwrap();
            out
        };
        let stacked = PackedReader::open(&dir.join("m.safetensors")).unwrap();
        let sd = stacked.descriptor("e").unwrap();
        assert_eq!(sd.encoding, QuantEncoding::nvfp4_per_slab());
        let r = SlicedReader::new(Box::new(PackedReader::open(&dir.join("m.safetensors")).unwrap())).with_slab_views("e", |x| format!("e.{x}")).unwrap();
        let mut names = r.names();
        names.sort();
        assert_eq!(names, ["e.0", "e.1", "e.2"]);
        for x in 0..3u64 {
            let d = r.descriptor(&format!("e.{x}")).unwrap();
            assert_eq!((d.shape.clone(), d.encoding.clone()), (vec![4, 32], QuantEncoding::nvfp4()));
            assert_eq!(decode(&r, &d, 0..128), decode(&stacked, &sd, x * 128..(x + 1) * 128), "expert {x}");
        }
        let half_row = SlicedReader::new(Box::new(PackedReader::open(&dir.join("m.safetensors")).unwrap())).with_view("bad", "e", 0..16, vec![16]);
        assert!(half_row.err().unwrap().contains("whole rows"));
        let crossing = SlicedReader::new(Box::new(PackedReader::open(&dir.join("m.safetensors")).unwrap())).with_view("bad", "e", 96..160, vec![2, 32]);
        assert!(crossing.err().unwrap().contains("cross a slab"));
    }
}
