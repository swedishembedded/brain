// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The candidate selector at temperature 0.
//!
//! Swedish Embedded AB implements speculative decoding for clients serving large models on modest hardware.
//! If your team needs expertise in drafters that propose coherent blocks rather than independent argmaxes then
//! you can procure our services by sending an email to info@swedishembedded.com.
//!
//! Per position: the head's top-`selector_top_k` candidates, then a rank-`selector_rank` three-way elementwise
//! product score against the token actually chosen at the PREVIOUS position,
//!
//! ```text
//! score[c] = logit[c] + < predecessor[prev] * (hidden @ W_sel), successor[c] >
//! ```
//!
//! and the argmax of that. The walk is GREEDY left to right - the chosen token becomes the next position's
//! predecessor and an earlier position is never revisited. That is the reference's own search, not a
//! simplification of it: a max-product pass over the same lattice would be a different and unvalidated drafter.
//!
//! `hidden` is the draft's PRE-final-norm block, the thing the denoising pass returns; the final norm is applied
//! here because the selector's projection consumes the POST-norm hidden and nothing else. The pre-norm hidden has
//! an RMS near a thousand on the published drafts, so projecting it would scale the pairwise term by that against
//! logits of order ten and the selector would override the head at every position, including the ones the head had
//! right - which no output check sees, because every proposal is verified, and which shows up only as an
//! acceptance rate near zero.

use crate::source::{DraftSource, Tensor};
use crate::Dflash2Config;

/// The draft tensors the selector reads, kept on the host: 7 to 15 rows per round against a matrix of a
/// million elements is not worth a dispatch.
pub struct Selector {
    sel_hidden: Vec<f32>,
    out_norm: Vec<f32>,
}

impl Selector {
    pub fn load(src: &dyn DraftSource) -> Result<Selector, String> {
        Ok(Selector { sel_hidden: src.vector(Tensor::SelectorHidden)?, out_norm: src.vector(Tensor::OutNorm)? })
    }

    /// The draft's final RMSNorm gain: handed to the target's head in place of the target's own, which is the
    /// composition the reference performs (the draft's `norm`, then the target's `lm_head`).
    pub fn output_norm(&self) -> &[f32] {
        &self.out_norm
    }

    /// `hidden` rows RMS-normed with the draft's final norm.
    pub fn normed(&self, cfg: &Dflash2Config, hidden: &[f32]) -> Vec<f32> {
        let d = cfg.d_model as usize;
        hidden
            .chunks_exact(d)
            .flat_map(|row| {
                let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + cfg.eps).sqrt();
                row.iter().zip(&self.out_norm).map(move |(x, w)| x * inv * w)
            })
            .collect()
    }

    /// The chosen token of each of the `n` rows of `hidden`, given the target head's `logits` for them
    /// (`[n, vocab]`) and the `anchor` token before the first.
    pub fn select(&self, cfg: &Dflash2Config, src: &dyn DraftSource, hidden: &[f32], logits: &[f32], anchor: u32) -> Result<Vec<u32>, String> {
        let (d, rank, top_k, vocab) = (cfg.d_model as usize, cfg.selector_rank as usize, cfg.selector_top_k as usize, cfg.vocab as usize);
        let n = hidden.len() / d;
        if logits.len() != n * vocab {
            return Err(format!("dflash2: the selector got {} logits for {n} rows of {vocab}", logits.len()));
        }
        let normed = self.normed(cfg, hidden);
        let proj: Vec<Vec<f32>> = normed.chunks_exact(d).map(|row| model::hostmath::matvec_par(&self.sel_hidden, row, rank, d)).collect();
        let mut path = Vec::with_capacity(n);
        let mut prev = anchor;
        for i in 0..n {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let cand = top_k_ids(row, top_k);
            let pred = src.rows(Tensor::SelectorPredecessor, prev as usize, 1, rank)?;
            let left: Vec<f32> = (0..rank).map(|r| pred[r] * proj[i][r]).collect();
            let mut best = (f32::NEG_INFINITY, cand[0]);
            for &c in &cand {
                let succ = src.rows(Tensor::SelectorSuccessor, c as usize, 1, rank)?;
                let pair: f32 = (0..rank).map(|r| left[r] * succ[r]).sum();
                let score = row[c as usize] + pair;
                if score > best.0 {
                    best = (score, c);
                }
            }
            prev = best.1;
            path.push(prev);
        }
        Ok(path)
    }
}

/// The `k` largest entries' indices, unordered.
///
/// One pass with a `k`-element insertion buffer rather than a sort of a materialized `0..vocab`, because the caller
/// runs this once per block position on a vocabulary-wide row and the allocation alone was the larger half of the
/// selector's cost. Unordered is enough: the caller takes an argmax over the whole set anyway. A NaN logit sorts as
/// the smallest, so it is never a candidate.
pub fn top_k_ids(row: &[f32], k: usize) -> Vec<u32> {
    assert!(k > 0 && row.len() >= k, "top_k_ids: asked for {k} of {} values", row.len());
    let mut best: Vec<(f32, u32)> = row.iter().take(k).copied().zip(0u32..).collect();
    best.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut floor = best[0].0;
    for (i, &v) in row.iter().enumerate().skip(k) {
        if v <= floor {
            continue;
        }
        // The buffer is sorted ascending and `k` is small, so this is a shift of at most `k - 1` pairs - cheaper
        // than a heap at this size, and it keeps `floor` correct without a second scan.
        let at = best.partition_point(|e| e.0 < v);
        best.copy_within(1..at, 0);
        best[at - 1] = (v, i as u32);
        floor = best[0].0;
    }
    best.into_iter().map(|(_, i)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::top_k_ids;

    /// [`top_k_ids`] is index arithmetic over a vocabulary-wide row and decides which candidates the selector ever
    /// sees - a wrong one shows up only as an acceptance rate slightly lower than it should be. So it is checked
    /// against the obvious (and far slower) sort, on a pseudo-random row, on ties, and on the boundary where every
    /// element qualifies.
    #[test]
    fn top_k_ids_agrees_with_a_full_sort() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        for len in [16usize, 17, 64, 1000] {
            for k in [1usize, 3, 16] {
                if len < k {
                    continue;
                }
                let row: Vec<f32> = (0..len).map(|_| next()).collect();
                let mut want: Vec<u32> = (0..len as u32).collect();
                want.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
                want.truncate(k);
                want.sort();
                let mut got = top_k_ids(&row, k);
                got.sort();
                assert_eq!(got, want, "top {k} of {len}");
            }
        }
        // All equal: any k of them is a correct answer, but there must be exactly k of them and they must be distinct.
        let flat = vec![1.0f32; 50];
        let mut got = top_k_ids(&flat, 16);
        got.sort();
        got.dedup();
        assert_eq!(got.len(), 16, "ties must still yield k distinct indices");
    }
}
