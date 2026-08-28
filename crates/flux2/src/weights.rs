// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where `Flux2Model::new_batched` gets its weights from, so that a Q8_0 GGUF
//! does not have to become 36 GB of host fp32 first.
//!
//! Swedish Embedded AB implements quantized checkpoint import and low-memory
//! model loading for its clients. If your team needs expertise in on-device
//! model loading then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this exists
//!
//! The DiT's int8 tier wants, per linear, `(packed u32, per-32-group scale)`.
//! A Q8_0 checkpoint already stores int8 with a per-32-block fp16 scale - the
//! SAME group size, which is why `model::int8::GROUP` is 32. Going
//! Q8_0 -> fp32 -> group-wise int8 therefore materializes an entire fp32 model
//! purely as an intermediate: on klein-9b that is 36.3 GB written, read back
//! twice by the quantizer, and freed again, none of which the result depends
//! on.
//!
//! [`DitWeights::Gguf`] goes straight from the block bytes to the packed
//! words, one weight matrix at a time.
//!
//! # Bit-identity
//!
//! The direct path is **bit-identical** to the fp32 round trip, not an
//! approximation of it, and the argument is short enough to state:
//!
//! - `deq_q8_0` yields exactly `(q as i8 as f32) * d`. `q` needs 7 bits and
//!   the fp16 scale `d` needs 11, so the product needs at most 18 - inside
//!   fp32's 24-bit significand. It is exact, not rounded.
//! - Every f32 that the round trip would have fed to the quantizer is
//!   therefore reproduced exactly by decoding the same block.
//! - The scales and the packing then run through `model::int8::group_scales`
//!   and `pack_row`, which are the very functions `quantize_weight` calls. Not
//!   a reimplementation of them - the same code.
//!
//! Now that `model::int8::GROUP` is 32, one of brain's scale groups IS one
//! Q8_0 block, and the requantization is not merely bit-identical to the fp32
//! round trip - it is the IDENTITY. A Q8_0 block stores `d = max|x|/127` and
//! `q = round(x/d)`, so `max|q| = 127`; the group's own absmax is therefore
//! `127*d`, brain's scale comes out as exactly `d`, and every `q` re-quantizes
//! to itself. Requantizing a Q8_0 checkpoint into this layout loses nothing at
//! all, which was not true when the scale spanned a whole row.
//!
//! So the packed `u32` words and the `f32` scales agree bit for bit, which is
//! why `gguf_int8_is_bit_identical_to_the_fp32_round_trip` asserts with
//! `assert_eq!` rather than a cosine/rel_l2 pair. If that gate ever goes red,
//! the premise above is wrong and the right response is to find out why, not
//! to relax the assertion.
//!
//! # Block alignment
//!
//! Every sub-matrix the model builds is a rectangle of a stored tensor: row
//! slices of a fused `qkv`/`linear1`, and the two column blocks of a
//! `linear2`. A Q8_0 block spans 32 consecutive elements of the row-major
//! flattening, so a rectangle is only directly decodable when its row stride
//! and both column bounds are multiples of 32. For klein-4b and klein-9b they
//! all are (hidden 3072/4096, mlp 9216/12288). This is CHECKED per rectangle,
//! never assumed: [`DitWeights::try_i8_rect`] returns `None` when it does not
//! hold, and the caller falls back to the fp32 route, which is always correct.

use std::cell::RefCell;

use checkpoint::gguf::MmapGguf;

use crate::import::Tensors;

/// A third-party LoRA that has been read and validated but NOT yet folded,
/// so that folding can happen one weight matrix at a time inside the model
/// build instead of over a resident fp32 map.
///
/// Pre-folding is what the fp32 path does, and it is why the adapter case
/// used to cost the same 36 GB peak as the unadapted one: the fold needs a
/// float domain, so every tensor the adapter touches had to be fp32 at the
/// same moment. A full-coverage klein-9b adapter touches 112 of 201 tensors,
/// and those 112 are the big fused matrices - very nearly the whole
/// parameter count - so "only the ones it touches" is no saving at all
/// unless they are also handled one at a time.
pub struct PendingLora {
    pairs: Vec<model::lora::ExternalPair>,
    scale: f32,
}

impl PendingLora {
    /// Read an adapter and validate every pair against `shapes` BEFORE
    /// anything is folded, so a rejected adapter leaves the build untouched
    /// rather than half applied - the same contract
    /// `lora::fold_external_adapter` gives, checked against the checkpoint's
    /// declared shapes instead of against a materialized map.
    pub fn open(path: &str, scale: f32, shapes: &dyn Fn(&str) -> Option<Vec<usize>>) -> Result<PendingLora, String> {
        let pairs = model::lora::read_external_adapter(path)?;
        for p in &pairs {
            match shapes(&p.base_key) {
                None => {
                    return Err(format!(
                        "lora {path}: adapter targets '{}' (from '{}'), which this FLUX.2 variant \
                         does not have - wrong base model for this adapter?",
                        p.base_key, p.stem
                    ))
                }
                Some(s) => {
                    if s.as_slice() != [p.out, p.inn] {
                        return Err(format!(
                            "lora {path}: '{}' is {s:?}, but the adapter for it is [{}, {}]",
                            p.base_key, p.out, p.inn
                        ));
                    }
                }
            }
        }
        Ok(PendingLora { pairs, scale })
    }

    /// Adapted linears, and the file's largest rank - what the caller logs.
    pub fn summary(&self) -> (usize, usize, f32) {
        (self.pairs.len(), self.pairs.iter().map(|p| p.r).max().unwrap_or(0), self.scale)
    }

    fn touches(&self, name: &str) -> bool {
        self.pairs.iter().any(|p| p.base_key == name)
    }

    /// Apply every pair targeting `name` to a full fp32 copy of that tensor.
    /// Same operation, same order and same scaling as the batch fold.
    fn apply(&self, name: &str, w: &mut [f32]) {
        for p in self.pairs.iter().filter(|p| p.base_key == name) {
            p.as_pair().delta(self.scale * p.alpha_mult, w);
        }
    }
}

/// The weight source a model build pulls from.
pub enum DitWeights<'a> {
    /// Everything already materialized as fp32 - what a safetensors import,
    /// a test fixture or a training path hands over.
    Map(&'a Tensors),
    /// A mapped Q8_0 GGUF, decoded per tensor on demand, with an optional
    /// LoRA folded in as each tensor is produced.
    Gguf {
        /// The mapped checkpoint; only its header has been read.
        gguf: &'a MmapGguf,
        /// Folded lazily, per tensor, never over a resident map.
        lora: Option<&'a PendingLora>,
        /// One-entry fp32 cache. The build asks for three row slices of the
        /// same `qkv` back to back, so without this the tensor would be
        /// decoded (and LoRA-folded) three times. One entry is enough
        /// precisely because the access pattern is grouped by tensor.
        cache: RefCell<Option<(String, Vec<f32>)>>,
    },
}

impl<'a> DitWeights<'a> {
    /// A GGUF-backed source with no adapter.
    pub fn gguf(gguf: &'a MmapGguf) -> DitWeights<'a> {
        DitWeights::Gguf { gguf, lora: None, cache: RefCell::new(None) }
    }

    /// A GGUF-backed source that folds `lora` into each tensor as it is read.
    pub fn gguf_adapted(gguf: &'a MmapGguf, lora: Option<&'a PendingLora>) -> DitWeights<'a> {
        DitWeights::Gguf { gguf, lora, cache: RefCell::new(None) }
    }

    /// A tensor's declared shape, if the source has it.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        match self {
            DitWeights::Map(ts) => ts.get(name).map(|(s, _)| s.clone()),
            DitWeights::Gguf { gguf, .. } => gguf.shape(name).map(|s| s.to_vec()),
        }
    }

    /// Lend `name`'s full fp32 contents to `f`, decoding and LoRA-folding on
    /// demand for a GGUF source. Panics naming the tensor if it is absent -
    /// the manifest was validated before the build started, so a miss here is
    /// a bug, not user error.
    pub fn with_f32<R>(&self, name: &str, f: impl FnOnce(&[f32]) -> R) -> R {
        match self {
            DitWeights::Map(ts) => {
                let (_, d) = ts.get(name).unwrap_or_else(|| panic!("flux2: missing tensor {name}"));
                f(d)
            }
            DitWeights::Gguf { gguf, lora, cache } => {
                {
                    let c = cache.borrow();
                    if let Some((n, d)) = c.as_ref() {
                        if n == name {
                            return f(d);
                        }
                    }
                }
                let mut d = match gguf.tensor(name) {
                    Some(Ok(v)) => v,
                    Some(Err(e)) => panic!("flux2: {name}: {e}"),
                    None => panic!("flux2: missing tensor {name}"),
                };
                if let Some(l) = lora {
                    l.apply(name, &mut d);
                }
                let r = f(&d);
                *cache.borrow_mut() = Some((name.to_string(), d));
                r
            }
        }
    }

    /// Requantize the rectangle `rows [r0, r0+n_out) x cols [c0, c0+k)` of a
    /// tensor stored as `[_, stride]` straight from Q8_0 blocks to
    /// `(packed, scales)` - `gguf::try_i8_rect`, the shared implementation
    /// (this crate's own module was where the byte-repack fact was first
    /// proven; it now lives in `crates/gguf` so a second GGUF-sourced model
    /// does not re-derive it).
    ///
    /// `None` - meaning "use the fp32 route" - whenever the direct path does
    /// not apply: an fp32 map source, a non-Q8_0 tensor, a tensor the LoRA
    /// touches (the fold needs a float domain, checked here since it is
    /// this crate's own concern), or a rectangle whose bounds are not
    /// Q8_0-block-aligned. Returning `None` is always safe; the caller's
    /// fallback produces the same bytes by the longer route.
    pub fn try_i8_rect(&self, name: &str, stride: usize, r0: usize, n_out: usize, c0: usize, k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
        let DitWeights::Gguf { gguf, lora, .. } = self else { return None };
        if lora.is_some_and(|l| l.touches(name)) {
            return None;
        }
        gguf::try_i8_rect(gguf, name, stride, r0, n_out, c0, k)
    }
}
