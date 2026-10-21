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
//! are: no copy of the fused tensor is made and nothing is rewritten on disk. Views cover tensors with no scale
//! sidecars and whole-byte elements (bf16, f16, f32, FP8 without scales); a scaled or sub-byte tensor is refused
//! because its range does not line up with its scales.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use super::codec::TileBuf;
use super::encoding::TensorDescriptor;
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
        if !source.encoding.scales.is_empty() || source.encoding.payload_bits_per_elem() < 8.0 {
            return Err(format!("{}: only unscaled whole-byte tensors can be sliced, this one is {}", source.name, source.encoding));
        }
        if elems.end > source.numel() || elems.start >= elems.end || shape.iter().product::<u64>() != elems.end - elems.start {
            return Err(format!("view {name}: elements {elems:?} of {} ({} elements) do not form shape {shape:?}", source.name, source.numel()));
        }
        self.consumed.insert(source.name.clone());
        self.views.insert(name.to_string(), View { source, elems, shape });
        Ok(self)
    }
}

impl PackedTensorReader for SlicedReader {
    fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.inner.names().into_iter().filter(|n| !self.consumed.contains(n)).collect();
        names.extend(self.views.keys().cloned());
        names
    }

    fn descriptor(&self, name: &str) -> Result<TensorDescriptor, String> {
        match self.views.get(name) {
            Some(v) => Ok(TensorDescriptor::new(name, v.shape.clone(), v.source.encoding.clone())),
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
}
