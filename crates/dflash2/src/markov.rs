// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The recurrent Markov head of a DSpark draft: what turns a denoised block into a coherent left-to-right path.
//!
//! Swedish Embedded AB implements speculative decoding - drafters, verifiers and the state rollback a looped or hybrid
//! stack needs - for clients serving large models on modest hardware. If your team needs a drafter whose proposals
//! hold together as a sequence then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The block's mask rows are denoised in parallel, so each row's own argmax ignores what the rows before it chose. The
//! head walks them in order with a gated recurrent state of rank `r`:
//!
//! ```text
//! f      = rmsnorm(sum_t softmax(fuse_weight[-1])_t * tap_t(last committed position)) * norm
//! state  = 0, prev = anchor
//! raw    = W [state; embed(prev); h_i; f] + b            (W is [3r, 2r + 2d])
//! gate   = sigmoid(raw[0:r]),  state = gate * state + (1 - gate) * tanh(raw[r:2r])
//! logits = head(h_i + W_out tanh(raw[2r:3r]) + b_out)    (the target's head and its soft cap)
//! prev   = argmax(logits)
//! ```
//!
//! The reference adds `head(h_i)` and `head(W_out latent + b_out)` as two products (it folds `head @ W_out` once); the
//! head is linear before its cap, so one product of the summed row is the same logits with one vocabulary-wide product
//! per step instead of two. The `[h_i; f]` columns of `W` are applied to every step at once; each step adds the
//! `[state; embed(prev)]` columns. All of it is host work: `3r x (2r + 2d)` per row is a few million products, and the
//! head's product is the target's own, on its device.

use model::hostmath::matvec_par;

use crate::selector::top_k_ids;

use crate::source::{DraftSource, Tensor};
use crate::Dflash2Config;

const MODEL: &str = "dflash2";

/// `sum_t alpha_t * taps[t]` of one row of `taps.len() / d` tapped hidden states, RMS-normed with `gain`.
pub fn mix_taps(taps: &[f32], alpha: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
    let d = gain.len();
    let mut mixed = vec![0.0f32; d];
    for (t, &a) in alpha.iter().enumerate() {
        for (m, &x) in mixed.iter_mut().zip(&taps[t * d..(t + 1) * d]) {
            *m += a * x;
        }
    }
    let inv = 1.0 / (mixed.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
    mixed.iter().zip(gain).map(|(x, w)| x * inv * w).collect()
}

/// A softmax of `logits`, in f64.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
    let e: Vec<f64> = logits.iter().map(|&v| (f64::from(v) - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().map(|v| (v / sum) as f32).collect()
}

/// The target's head as the walk calls it: rows `[m, d_model]` in, logits `[m, vocab]` out.
pub type Project<'a> = dyn FnMut(&[f32]) -> Result<Vec<f32>, String> + 'a;

/// One sequence's walk.
pub struct MarkovJob<'a> {
    /// The block's mask rows after the draft's final norm, `[block_size - 1, d_model]`.
    pub hidden: &'a [f32],
    /// The target's taps at the last committed position, `[taps * d_model]`.
    pub last_taps: &'a [f32],
    /// The anchor token: the block's first row, the walk's first `prev`.
    pub anchor: u32,
    /// Steps to walk (at most the mask rows).
    pub want: usize,
}

/// The head's tensors, on the host.
pub struct MarkovHead {
    rank: usize,
    d: usize,
    /// `[3r, 2r]`: the columns over `[state; embed(prev)]`.
    w_step: Vec<f32>,
    /// `[3r, 2d]`: the columns over `[hidden; context]`.
    w_ctx: Vec<f32>,
    bias: Vec<f32>,
    /// `[d, r]` and `[d]`.
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    norm: Vec<f32>,
    /// The head's softmax over the taps (the last row of `fuse_weight`).
    alpha: Vec<f32>,
    eps: f32,
}

impl MarkovHead {
    /// Read the head of `cfg` (a DSpark draft) from `src`; `fuse` is `fuse_weight`, `[layers + 1, taps]`.
    pub fn load(cfg: &Dflash2Config, src: &dyn DraftSource, fuse: &[f32]) -> Result<MarkovHead, String> {
        let rank = cfg.markov_rank().ok_or_else(|| format!("{MODEL}: a Markov head on a draft that has none"))? as usize;
        let (d, taps) = (cfg.d_model as usize, cfg.target_layers.len());
        let read = |t: Tensor, len: usize| -> Result<Vec<f32>, String> {
            let v = src.vector(t)?;
            if v.len() != len {
                return Err(format!("{MODEL}: {} holds {} values, the config implies {len}", t.name_hf(), v.len()));
            }
            Ok(v)
        };
        let width = 2 * rank + 2 * d;
        let w = read(Tensor::MarkovRnn, 3 * rank * width)?;
        let (mut w_step, mut w_ctx) = (Vec::with_capacity(3 * rank * 2 * rank), Vec::with_capacity(3 * rank * 2 * d));
        for row in w.chunks_exact(width) {
            w_step.extend_from_slice(&row[..2 * rank]);
            w_ctx.extend_from_slice(&row[2 * rank..]);
        }
        if fuse.len() != (cfg.n_layers + 1) * taps {
            return Err(format!("{MODEL}: fuse_weight holds {} values, the config implies {} x {taps}", fuse.len(), cfg.n_layers + 1));
        }
        if let Some(shape) = src.shape(Tensor::MarkovEmbed).filter(|s| *s != [cfg.vocab as usize, rank]) {
            return Err(format!("{MODEL}: {} is {shape:?}, the config implies [{}, {rank}]", Tensor::MarkovEmbed.name_hf(), cfg.vocab));
        }
        Ok(MarkovHead {
            rank,
            d,
            w_step,
            w_ctx,
            bias: read(Tensor::MarkovRnnBias, 3 * rank)?,
            w_out: read(Tensor::MarkovOut, d * rank)?,
            b_out: read(Tensor::MarkovOutBias, d)?,
            norm: read(Tensor::MarkovNorm, d)?,
            alpha: softmax(&fuse[cfg.n_layers * taps..]),
            eps: cfg.eps,
        })
    }

    /// The walks of `jobs`, `want` tokens each. `project` is the target's head: rows `[m, d]` in, logits `[m, vocab]`
    /// out (soft-capped as the target caps them, padding rows floored), one call per step for every job still walking.
    pub fn walk(&self, src: &dyn DraftSource, jobs: &[MarkovJob<'_>], project: &mut Project<'_>) -> Result<Vec<Vec<u32>>, String> {
        let (r, d) = (self.rank, self.d);
        // Per job and step, the bias plus the `[hidden; context]` columns.
        let mut fixed: Vec<Vec<Vec<f32>>> = Vec::with_capacity(jobs.len());
        for j in jobs {
            if !j.hidden.len().is_multiple_of(d) || j.want > j.hidden.len() / d || j.last_taps.len() != self.alpha.len() * d {
                return Err(format!("{MODEL}: a Markov walk of {} steps over {} hidden values and {} tap values", j.want, j.hidden.len(), j.last_taps.len()));
            }
            let context = mix_taps(j.last_taps, &self.alpha, &self.norm, self.eps);
            let steps = (0..j.want)
                .map(|i| {
                    let mut input = j.hidden[i * d..(i + 1) * d].to_vec();
                    input.extend_from_slice(&context);
                    matvec_par(&self.w_ctx, &input, 3 * r, 2 * d).iter().zip(&self.bias).map(|(a, b)| a + b).collect()
                })
                .collect();
            fixed.push(steps);
        }
        let mut state = vec![vec![0.0f32; r]; jobs.len()];
        let mut prev: Vec<u32> = jobs.iter().map(|j| j.anchor).collect();
        let mut paths: Vec<Vec<u32>> = jobs.iter().map(|j| Vec::with_capacity(j.want)).collect();
        let longest = jobs.iter().map(|j| j.want).max().unwrap_or(0);
        for i in 0..longest {
            let walking: Vec<usize> = (0..jobs.len()).filter(|&j| jobs[j].want > i).collect();
            let mut rows = Vec::with_capacity(walking.len() * d);
            for &j in &walking {
                let mut input = state[j].clone();
                input.extend(src.rows(Tensor::MarkovEmbed, prev[j] as usize, 1, r)?);
                let raw: Vec<f32> = matvec_par(&self.w_step, &input, 3 * r, 2 * r).iter().zip(&fixed[j][i]).map(|(a, b)| a + b).collect();
                let latent: Vec<f32> = raw[2 * r..].iter().map(|v| v.tanh()).collect();
                for (c, s) in state[j].iter_mut().enumerate() {
                    let gate = 1.0 / (1.0 + (-raw[c]).exp());
                    *s = gate * *s + (1.0 - gate) * raw[r + c].tanh();
                }
                let corr = matvec_par(&self.w_out, &latent, d, r);
                rows.extend(jobs[j].hidden[i * d..(i + 1) * d].iter().zip(corr.iter().zip(&self.b_out)).map(|(h, (c, b))| h + c + b));
            }
            let logits = project(&rows)?;
            if logits.is_empty() || logits.len() % walking.len() != 0 {
                return Err(format!("{MODEL}: the head answered {} logits for {} rows", logits.len(), walking.len()));
            }
            let vocab = logits.len() / walking.len();
            for (k, &j) in walking.iter().enumerate() {
                // The argmax, skipping NaN.
                let best = top_k_ids(&logits[k * vocab..(k + 1) * vocab], 1);
                prev[j] = *best.first().ok_or_else(|| format!("{MODEL}: the head's logits at block position {i} are all NaN"))?;
                paths[j].push(prev[j]);
            }
        }
        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixing_weights_the_taps_then_normalises() {
        // Two taps of width 2: 0.25 * [4, 0] + 0.75 * [0, 4] = [1, 3], rms sqrt(5).
        let got = mix_taps(&[4.0, 0.0, 0.0, 4.0], &[0.25, 0.75], &[1.0, 2.0], 0.0);
        let inv = 1.0 / 5.0f32.sqrt();
        assert!((got[0] - inv).abs() < 1e-6 && (got[1] - 6.0 * inv).abs() < 1e-6, "{got:?}");
        let p = softmax(&[0.0, f32::ln(3.0)]);
        assert!((p[0] - 0.25).abs() < 1e-6 && (p[1] - 0.75).abs() < 1e-6);
    }
}
