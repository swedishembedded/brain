// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A row-wise concatenation of several tensors as one virtual tensor, so a fused
//! projection (`wq|wk|wv`, `gate|up`) loads packed through the same tile
//! executor as any other tensor: the fused payload is the parts' payloads end to
//! end, and so are their scale tensors.
//!
//! Only parts that can be joined without re-quantizing are accepted: identical
//! encodings, rank 2 with the same inner extent, per-block/row/tile scales
//! (a per-tensor scale differs between parts), and, for 2-D tile scales, parts
//! whose row counts are whole tiles. Anything else is an error at construction;
//! the caller then falls back to the f32 interface.

use std::ops::Range;

use super::codec::TileBuf;
use super::encoding::{ScaleGranularity, TensorDescriptor};
use super::reader::PackedTensorReader;

pub(super) struct ConcatView<'a> {
    reader: &'a dyn PackedTensorReader,
    parts: Vec<TensorDescriptor>,
    /// First flat element of each part within the fused tensor.
    offsets: Vec<u64>,
    fused: TensorDescriptor,
}

impl<'a> ConcatView<'a> {
    pub(super) fn new(reader: &'a dyn PackedTensorReader, parts: Vec<TensorDescriptor>) -> Result<ConcatView<'a>, String> {
        let first = parts.first().ok_or("fused tensor with no parts")?;
        let names = parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("+");
        let k = *first.shape.last().unwrap_or(&0);
        if first.shape.len() != 2 {
            return Err(format!("{names}: only rank-2 tensors can be fused, {} has shape {:?}", first.name, first.shape));
        }
        for s in &first.encoding.scales {
            if s.granularity == ScaleGranularity::Tensor {
                return Err(format!("{names}: a per-tensor scale differs between parts and cannot be fused without re-quantizing"));
            }
        }
        let (mut rows, mut offsets) = (0u64, Vec::new());
        for p in &parts {
            if p.encoding != first.encoding || p.shape.len() != 2 || p.shape[1] != k {
                return Err(format!("{names}: parts differ in encoding or inner extent ({} is {} {:?})", p.name, p.encoding, p.shape));
            }
            if let Some(ScaleGranularity::Tile2d { rows: tile, .. }) = first.encoding.scales.first().map(|s| s.granularity) {
                if !p.shape[0].is_multiple_of(tile as u64) {
                    return Err(format!("{names}: {} has {} rows, not a whole number of {tile}-row scale tiles", p.name, p.shape[0]));
                }
            }
            offsets.push(rows * k);
            rows += p.shape[0];
        }
        let fused = TensorDescriptor::new(format!("fused({names})"), vec![rows, k], first.encoding.clone());
        fused.validate()?;
        Ok(ConcatView { reader, parts, offsets, fused })
    }

    pub(super) fn descriptor(&self) -> &TensorDescriptor {
        &self.fused
    }

    fn pieces(&self, elems: &Range<u64>) -> impl Iterator<Item = (usize, Range<u64>)> + '_ {
        let elems = elems.clone();
        self.parts.iter().enumerate().filter_map(move |(i, p)| {
            let (lo, hi) = (self.offsets[i], self.offsets[i] + p.numel());
            let (s, e) = (elems.start.max(lo), elems.end.min(hi));
            (s < e).then(|| (i, s - lo..e - lo))
        })
    }
}

impl PackedTensorReader for ConcatView<'_> {
    fn names(&self) -> Vec<String> {
        vec![self.fused.name.clone()]
    }

    fn descriptor(&self, name: &str) -> Result<TensorDescriptor, String> {
        if name == self.fused.name { Ok(self.fused.clone()) } else { Err(format!("{name}: not the fused tensor {}", self.fused.name)) }
    }

    fn read_blocks(&self, desc: &TensorDescriptor, elems: &Range<u64>, dst: &mut TileBuf) -> Result<(), String> {
        dst.payload.clear();
        dst.scales.resize_with(desc.encoding.scales.len(), Vec::new);
        dst.scales.iter_mut().for_each(Vec::clear);
        let mut part_buf = TileBuf::default();
        for (i, local) in self.pieces(elems) {
            self.reader.read_blocks(&self.parts[i], &local, &mut part_buf)?;
            dst.payload.extend_from_slice(&part_buf.payload);
            for (l, s) in part_buf.scales.iter().enumerate() {
                dst.scales[l].extend_from_slice(s);
            }
        }
        Ok(())
    }

    fn release(&self, _desc: &TensorDescriptor, elems: &Range<u64>) {
        for (i, local) in self.pieces(elems) {
            self.reader.release(&self.parts[i], &local);
        }
    }
}
