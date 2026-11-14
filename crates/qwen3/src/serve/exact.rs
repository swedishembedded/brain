// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible LLM serving for evaluation and
// on-policy training for its clients, where a request must decode the same
// tokens and log-probabilities whatever else shares its batch. If your team
// needs expertise in batch-invariant inference then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The engine's exact-decode regime (`--exact-decode`): a decode row computes what it computes alone, whatever else
//! shares its step.
//!
//! Every choice a decode step makes from its row count is pinned to one answer for every row count:
//!
//! - each linear and the head run the variant `select::row_invariant_candidates` names (the decode GEMV), chosen once
//!   per weight shape through the autotuner, never the per-bucket choice or a library GEMM;
//! - a step carries at most [`ExactDecode::group_rows`] rows (`select::exact_decode_group_rows`, the rows over which
//!   that variant keeps its arithmetic); a larger batch is decoded as consecutive groups, each a whole step;
//! - the int8 attention splits every row's keys as a single sequence's are split, and the f32-KV attention runs the
//!   one form that serves every batch size;
//! - a prompt reuses only cached blocks of whole prefill chunks at the cold prompt's chunk boundaries, so a cached
//!   prefix holds the arithmetic a cold prefill of it computes, and the rest of the prompt is the cold prompt's
//!   remaining chunks;
//! - speculation is off: a verify pass forwards a sequence's consecutive tokens as one chunk.
//!
//! A configuration with a row-dependent part this regime cannot pin (a per-row adapter bank, latent sparse
//! attention, a 4-bit tier served through the ops facade, an int8 tier on a device whose only int8 candidate is not
//! declared invariant) is refused by name.

use std::collections::HashMap;

use gpu_core::select::{AutoTuner, Dtype, KernelVariant, Op, OpShape, DECODE_REGIME_MAX_ROWS};
use model::paged::BlockTable;

use super::{i8_shapes, int8_decode_part_words, int8_decode_splits, Engine, INT8_DECODE_PART_STRIDE};

/// What the exact-decode regime pins, chosen once when it is turned on.
pub(super) struct ExactDecode {
    /// The most rows one decode step carries: the narrowest group of the engine's weight tiers.
    pub(super) group_rows: u32,
    /// Each int8 linear shape's one variant, by `(n, k)`.
    i8: HashMap<(u32, u32), KernelVariant>,
}

impl ExactDecode {
    /// The int8 variant pinned for an `[n, k]` weight.
    pub(super) fn i8_variant(&self, n: u32, k: u32) -> KernelVariant {
        *self.i8.get(&(n, k)).unwrap_or_else(|| panic!("qwen3 exact decode: no pinned int8 variant for a [{n}, {k}] weight"))
    }

    /// The longest prefix of `matched` cached blocks (of `block_size` tokens) that ends on a boundary of the cold
    /// prompt's prefill chunks (`chunk` rows each, from position 0): only there does the rest of the prompt run as the
    /// cold prompt's remaining chunks.
    pub(super) fn reusable_blocks(matched: usize, block_size: u32, chunk: u32) -> usize {
        let unit = lcm(block_size, chunk) as usize;
        (matched * block_size as usize) / unit * unit / block_size as usize
    }

    /// Tokens of a prompt's leading run that lie in whole prefill chunks of `chunk` rows: the blocks a later prompt
    /// may reuse.
    pub(super) fn publishable_tokens(lead: usize, chunk: u32) -> usize {
        lead / chunk as usize * chunk as usize
    }
}

fn lcm(a: u32, b: u32) -> u32 {
    let gcd = |mut x: u32, mut y: u32| {
        while y != 0 {
            (x, y) = (y, x % y);
        }
        x
    };
    a / gcd(a, b) * b
}

impl Engine {
    /// Turn the exact-decode regime on or off (see the module doc). Off is the default. Refuses, naming the reason,
    /// a configuration with a row-dependent part the regime cannot pin.
    pub fn set_exact_decode(&mut self, on: bool) -> Result<(), String> {
        self.tape_cache.clear();
        if !on {
            self.exact = None;
            return Ok(());
        }
        if self.bank.is_some() {
            return Err("qwen3: --exact-decode is not supported with an adapter bank: a per-row adapter's correction is tiled by the batch's row count".to_string());
        }
        if self.lsa.is_some() {
            return Err("qwen3: --exact-decode is not supported with --sparse-attention: the sparse read's chunking follows the batch".to_string());
        }
        if self.packed_ops.is_some() {
            return Err("qwen3: --exact-decode is not supported at a 4-bit weight tier: its products are chosen per row count inside the ops facade".to_string());
        }
        let caps = self.caps.clone();
        let group_rows = self.lin_weights.values().map(|w| gpu_core::select::exact_decode_group_rows(&caps, w.dtype())).fold(self.max_batch.min(DECODE_REGIME_MAX_ROWS), u32::min).max(1);
        let i8_shapes = i8_shapes(&self.lin_weights);
        let mut i8 = HashMap::new();
        if !i8_shapes.is_empty() {
            let scratch = self.i8_scratch.as_ref().expect("qwen3 serve: I8 weight built without i8_scratch");
            let store = gpu_core::tune::FileTuneStore::for_device(&self.gpu, super::i8_tune_fingerprint()).map(|s| Box::new(s) as Box<dyn gpu_core::select::TuneStore>);
            let tuner = AutoTuner::new(store);
            for (n, k, wb, sw) in i8_shapes {
                // Timed at the group's rows: the most a step carries, where a slower choice costs most.
                let shape = OpShape { m: group_rows, n, k, dtype: Dtype::I8 };
                let mut measure = |v: KernelVariant| Self::measure_i8(&self.gpu, scratch, wb, sw, n, k, group_rows, v);
                let v = tuner.resolve_row_invariant(Op::MatMul, shape, &caps, &mut measure).ok_or_else(|| {
                    "qwen3: --exact-decode is not supported for int8 weights on this device: none of its int8 products keeps a row's arithmetic at every row count".to_string()
                })?;
                i8.insert((n, k), v);
            }
        }
        // Every row splits its keys as a single sequence's are split, so the partial buffer holds a whole group at
        // the single-sequence split count.
        let words = u64::from(group_rows) * u64::from(self.cfg.n_heads) * u64::from(int8_decode_splits(&self.cfg, 1, self.cap)) * INT8_DECODE_PART_STRIDE;
        if self.kv_int8 && words > int8_decode_part_words(&self.cfg, self.max_batch.max(self.max_prefill), self.cap) {
            self.sc.part_i8 = self.gpu.storage(words);
        }
        self.exact = Some(ExactDecode { group_rows, i8 });
        Ok(())
    }

    /// Whether the exact-decode regime is on.
    pub fn exact_decode(&self) -> bool {
        self.exact.is_some()
    }

    /// Run a decode step over `tables` (row `i` fed `inputs[i]`) as consecutive groups of at most the regime's
    /// [`ExactDecode::group_rows`], or as one step outside the regime; one answer per row, in row order.
    pub(super) fn grouped<T>(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], mut step: impl FnMut(&mut Engine, &mut [&mut BlockTable], &[u32]) -> Vec<T>) -> Vec<T> {
        let group = match &self.exact {
            Some(e) if tables.len() > e.group_rows as usize => e.group_rows as usize,
            _ => return step(self, tables, inputs),
        };
        let mut out = Vec::with_capacity(tables.len());
        for (rows, ins) in tables.chunks_mut(group).zip(inputs.chunks(group)) {
            out.extend(step(self, rows, ins));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cached prefix is reused only up to a boundary of both the cold prompt's chunks and the blocks.
    #[test]
    fn a_cached_prefix_is_reused_to_a_chunk_boundary_only() {
        assert_eq!(ExactDecode::reusable_blocks(40, 16, 256), 32, "640 cached tokens reuse the 512 of two whole chunks");
        assert_eq!(ExactDecode::reusable_blocks(15, 16, 256), 0, "less than one chunk is computed cold");
        assert_eq!(ExactDecode::reusable_blocks(10, 16, 40), 10, "160 tokens are four chunks of 40 and ten blocks");
        assert_eq!(ExactDecode::reusable_blocks(9, 16, 40), 5, "144 tokens end inside a chunk; 80 is the last common boundary");
        assert_eq!(ExactDecode::publishable_tokens(700, 256), 512);
        assert_eq!(ExactDecode::publishable_tokens(255, 256), 0);
    }
}
