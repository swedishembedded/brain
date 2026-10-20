// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! [`PackedSource`]: the compatibility bridge from the tile reader to
//! [`TensorSource`], the by-name f32 interface existing model builders pull
//! from.
//!
//! A model supplies only a declarative name map ([`Binding`]: canonical name to
//! stored name, plus an optional additive constant for checkpoints that store
//! a fold of the value). Everything else, discovering scale tensors, decoding
//! FP8/NF4/GGUF/bf16 bytes, bounding scratch, releasing file pages, is the
//! reader and the codec registry. Chunks are decoded tile by tile, so host
//! scratch is `O(max_elems)`, never `O(tensor)`; only [`TensorSource::with_tensor`]
//! (whose contract is a whole-tensor slice) assembles a whole tensor, in tiles.

use std::collections::HashMap;
use std::ops::Range;

use super::codec::{CodecRegistry, Tile, TileBuf};
use super::encoding::TensorDescriptor;
use super::reader::PackedTensorReader;
use crate::TensorSource;

/// Tile length used to assemble a whole tensor for `with_tensor`.
const WHOLE_TENSOR_TILE_ELEMS: u64 = 1 << 22;

/// Where one canonical tensor comes from.
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// The tensor's name in the checkpoint.
    pub source: String,
    /// Added to every decoded value (a stored `w` that is applied as `1 + w`
    /// is declared with `add: 1.0`). `0.0` is the identity.
    pub add: f32,
}

impl Binding {
    pub fn new(source: impl Into<String>) -> Binding {
        Binding { source: source.into(), add: 0.0 }
    }
}

struct Bound {
    desc: TensorDescriptor,
    add: f32,
}

/// A [`TensorSource`] over a [`PackedTensorReader`].
pub struct PackedSource {
    reader: Box<dyn PackedTensorReader>,
    codecs: CodecRegistry,
    tensors: HashMap<String, Bound>,
}

impl PackedSource {
    /// Serve `bindings` from `reader`. Every binding is resolved and every
    /// encoding checked decodable now, from headers alone, so a mismatch is one
    /// named error at open rather than a failure deep into a load.
    pub fn new(reader: Box<dyn PackedTensorReader>, codecs: CodecRegistry, bindings: impl IntoIterator<Item = (String, Binding)>) -> Result<PackedSource, String> {
        let mut tensors = HashMap::new();
        for (canonical, b) in bindings {
            let desc = reader.descriptor(&b.source)?;
            codecs.decoder(&desc.encoding).map_err(|e| format!("{}: {e}", b.source))?;
            tensors.insert(canonical, Bound { desc, add: b.add });
        }
        Ok(PackedSource { reader, codecs, tensors })
    }

    /// Every tensor under its own stored name.
    pub fn identity(reader: Box<dyn PackedTensorReader>, codecs: CodecRegistry) -> Result<PackedSource, String> {
        let bindings: Vec<_> = reader.names().into_iter().map(|n| (n.clone(), Binding::new(n))).collect();
        PackedSource::new(reader, codecs, bindings)
    }

    /// Canonical names served, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name).map(|b| &b.desc)
    }

    /// Decode tile `elems` of `b` into `out` (replacing its contents), with
    /// `read` as the staging buffer.
    fn decode_range(&self, b: &Bound, elems: &Range<u64>, read: &mut TileBuf, out: &mut Vec<f32>) -> Result<(), String> {
        let desc = &b.desc;
        self.reader.read_blocks(desc, elems, read)?;
        let t = Tile { elems: elems.clone(), payload: &read.payload, scales: read.scales.iter().map(Vec::as_slice).collect() };
        out.clear();
        out.reserve_exact((elems.end - elems.start) as usize);
        self.codecs.decoder(&desc.encoding)?.decode_tile(desc, &t, out)?;
        if b.add != 0.0 {
            out.iter_mut().for_each(|v| *v += b.add);
        }
        self.reader.release(desc, elems);
        Ok(())
    }

    /// Tile length for chunks of at most `max_elems`: rounded down to the
    /// tensor's tile quantum but never below one quantum; `0` means one tile.
    fn tile_len(desc: &TensorDescriptor, max_elems: u64) -> u64 {
        let quantum = desc.tile_quantum();
        if max_elems == 0 { desc.numel() } else { (max_elems / quantum).max(1) * quantum }
    }

    /// Decode `b` in tiles of at most `max_elems`, calling `f` with each tile's
    /// element offset and values, in order.
    fn for_each_tile(&self, b: &Bound, max_elems: u64, f: &mut dyn FnMut(u64, &[f32])) -> Result<(), String> {
        let (numel, tile) = (b.desc.numel(), Self::tile_len(&b.desc, max_elems));
        let (mut read, mut out) = (TileBuf::default(), Vec::new());
        let mut start = 0;
        while start < numel {
            let elems = start..(start + tile).min(numel);
            self.decode_range(b, &elems, &mut read, &mut out)?;
            f(start, &out);
            start = elems.end;
        }
        Ok(())
    }

    /// The whole tensor in one vector, decoded tile by tile straight into it
    /// (no second tensor-sized scratch).
    fn decode_whole(&self, b: &Bound) -> Result<Vec<f32>, String> {
        let (numel, tile) = (b.desc.numel(), Self::tile_len(&b.desc, WHOLE_TENSOR_TILE_ELEMS));
        let mut read = TileBuf::default();
        if tile >= numel {
            let mut out = Vec::new();
            self.decode_range(b, &(0..numel), &mut read, &mut out)?;
            return Ok(out);
        }
        let (mut whole, mut part) = (Vec::with_capacity(numel as usize), Vec::new());
        let mut start = 0;
        while start < numel {
            let elems = start..(start + tile).min(numel);
            self.decode_range(b, &elems, &mut read, &mut part)?;
            whole.extend_from_slice(&part);
            start = elems.end;
        }
        Ok(whole)
    }
}

impl TensorSource for PackedSource {
    /// Panics on a read error after open validated the headers (a truncated or
    /// concurrently modified file): silently reporting "not found" would
    /// surface as a missing-tensor naming bug far from the cause.
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        let Some(b) = self.tensors.get(name) else { return false };
        f(&self.decode_whole(b).unwrap_or_else(|e| panic!("PackedSource: {name}: {e}")));
        true
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        let Some(b) = self.tensors.get(name) else { return false };
        self.for_each_tile(b, max_elems as u64, f).unwrap_or_else(|e| panic!("PackedSource: {name}: {e}"));
        true
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.tensors.get(name).map(|b| b.desc.numel() as usize)
    }

    fn advise_drop(&self, name: &str) {
        if let Some(b) = self.tensors.get(name) {
            self.reader.release(&b.desc, &(0..b.desc.numel()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed::reader::PackedReader;
    use crate::packed::testfix::{bf16_bytes, e4m3_bytes, write_safetensors, RawTensor, TmpDir};
    use crate::safetensors::e4m3fn_to_f32;

    #[test]
    fn fp8_and_plain_tensors_decode_through_a_name_map_with_bounded_chunks() {
        let dir = TmpDir::new("src");
        let (rows, cols) = (32usize, 48usize);
        let w = e4m3_bytes(21, rows * cols);
        let grid = [0.5f32, 1.0, 2.0, 4.0, 0.25, 8.0];
        let norm = [0.5f32, -0.25, 1.5];
        write_safetensors(
            &dir.join("m.safetensors"),
            &[
                RawTensor { name: "a.weight", dtype: "F8_E4M3", shape: vec![rows as u64, cols as u64], bytes: w.clone() },
                RawTensor { name: "a.weight_scale_inv", dtype: "BF16", shape: vec![2, 3], bytes: bf16_bytes(&grid) },
                RawTensor { name: "n.weight", dtype: "BF16", shape: vec![3], bytes: bf16_bytes(&norm) },
            ],
        );
        let reader = PackedReader::open(&dir.join("m.safetensors")).unwrap().with_fp8_block(16);
        let bindings = [("blocks.0.a".to_string(), Binding::new("a.weight")), ("blocks.0.ln".to_string(), Binding { source: "n.weight".into(), add: 1.0 })];
        let src = PackedSource::new(Box::new(reader), CodecRegistry::reference(), bindings).unwrap();
        assert_eq!(src.numel("blocks.0.a"), Some(rows * cols));

        let want: Vec<f32> = (0..rows * cols).map(|i| e4m3fn_to_f32(w[i]) * half::bf16::from_f32(grid[(i / cols) / 16 * 3 + (i % cols) / 16]).to_f32()).collect();
        let (mut got, mut largest) = (Vec::new(), 0usize);
        assert!(src.with_tensor_chunks("blocks.0.a", 2 * cols, &mut |off, c| {
            assert_eq!(off as usize, got.len());
            largest = largest.max(c.len());
            got.extend_from_slice(c);
        }));
        assert_eq!(got, want);
        assert!(largest <= 2 * cols, "chunk of {largest} elements exceeds the requested 2 rows");
        let mut whole = Vec::new();
        assert!(src.with_tensor("blocks.0.a", &mut |d| whole = d.to_vec()));
        assert_eq!(whole, want);
        assert!(src.with_tensor("blocks.0.ln", &mut |d| assert_eq!(d, [1.5, 0.75, 2.5])));
        assert!(!src.with_tensor("missing", &mut |_| {}));
    }
}
