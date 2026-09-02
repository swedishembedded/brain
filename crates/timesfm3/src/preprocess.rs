// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side pre/postprocessing around [`crate::model::Timesfm3::core_forward`]:
//! patching, running RevIN, linear detrending, CPM iterative refinement and
//! forecast stitching - everything the reference does in plain tensor ops
//! outside its own transformer stack. Small arrays (context is at most a few
//! thousand floats even at the model's max), so this is ordinary `Vec<f32>`
//! host math, not device dispatch - the same split `fincast`/`chronos2` use.
//!
//! # Layout
//!
//! A "panel" here is `[b, v, t]` row-major: `b` batch, `v` variates in
//! **target, then past-only covariates, then past-and-future covariates**
//! order (the reference's own concatenation order - `patch_is_target` covers
//! the first `num_target + num_past_only` variates, never just the targets),
//! `t` raw time steps (not yet patched). Boolean masks use the reference's
//! own convention throughout this module: **`true` = masked/invalid**,
//! inverted from "attend".

use crate::config::Timesfm3Config;

/// One decode() call's shape: how many variates of each kind, and how long
/// the context/horizon are.
#[derive(Clone, Copy, Debug)]
pub struct DecodeShape {
    pub batch: usize,
    pub num_target: usize,
    pub num_past_only: usize,
    pub num_past_future: usize,
    pub context: usize,
    pub horizon: usize,
}

impl DecodeShape {
    pub fn num_variates(&self) -> usize {
        self.num_target + self.num_past_only + self.num_past_future
    }
}

/// `torch.finfo(f32).eps`-free "safe division" guard the reference uses for
/// RevIN specifically (`_TOLERANCE = 1e-6` in `util.py`, distinct from the
/// RMSNorm epsilon): a sigma below this is treated as exactly `1.0`, not
/// clamped up to it.
const REVIN_TOLERANCE: f32 = 1e-6;

#[inline]
fn safe_sigma(sigma: f32) -> f32 {
    if sigma < REVIN_TOLERANCE {
        1.0
    } else {
        sigma
    }
}

/// `(x - mu) / safe(sigma)` forward, `x * sigma + mu` reverse.
#[inline]
pub fn revin(x: f32, mu: f32, sigma: f32, reverse: bool) -> f32 {
    if reverse {
        x * sigma + mu
    } else {
        (x - mu) / safe_sigma(sigma)
    }
}

/// Masked mean/variance of one patch's raw values (`true` = masked/invalid,
/// excluded). Returns `(n_valid, mean, std)`; `std` uses the BIASED
/// (divide-by-n) estimator, matching the reference exactly - not Bessel's
/// correction.
fn masked_patch_stats(vals: &[f32], mask: &[bool]) -> (f32, f32, f32) {
    let mut n = 0f32;
    let mut sum = 0f32;
    for (&v, &m) in vals.iter().zip(mask) {
        if !m {
            n += 1.0;
            sum += v;
        }
    }
    let mu = if n > 0.0 { sum / n } else { 0.0 };
    let mut sq = 0f32;
    for (&v, &m) in vals.iter().zip(mask) {
        if !m {
            sq += (v - mu) * (v - mu);
        }
    }
    let sigma = if n > 0.0 { (sq / n).sqrt() } else { 0.0 };
    (n, mu, sigma)
}

/// Merge running `(n, mu, sigma)` with one new patch's `(inc_n, inc_mu,
/// inc_sigma)` - the reference's masked parallel-variance merge
/// (`util.update_running_stats`), biased (÷n) throughout.
fn update_running_stats(n: f32, mu: f32, sigma: f32, inc_n: f32, inc_mu: f32, inc_sigma: f32) -> (f32, f32, f32) {
    let new_n = n + inc_n;
    if new_n == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let new_mu = (n * mu + inc_mu * inc_n) / new_n;
    let var = (n * sigma * sigma + inc_n * inc_sigma * inc_sigma + n * (mu - new_mu) * (mu - new_mu) + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)) / new_n;
    (new_n, new_mu, var.sqrt())
}

/// Per-(batch, variate) sequential scan over patches producing cumulative
/// `(n, mu, sigma)` at every patch boundary - the running RevIN statistics
/// `_preprocess` computes before standardizing. `values`/`masks` are
/// `[b, v, n_patches*patch_len]` (already patch-boundary-aligned).
pub fn running_stats(values: &[f32], masks: &[bool], b: usize, v: usize, n_patches: usize, patch_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out_n = vec![0f32; b * v * n_patches];
    let mut out_mu = vec![0f32; b * v * n_patches];
    let mut out_sigma = vec![0f32; b * v * n_patches];
    for bi in 0..b {
        for vi in 0..v {
            let (mut n, mut mu, mut sigma) = (0f32, 0f32, 0f32);
            for p in 0..n_patches {
                let off = ((bi * v + vi) * n_patches + p) * patch_len;
                let (inc_n, inc_mu, inc_sigma) = masked_patch_stats(&values[off..off + patch_len], &masks[off..off + patch_len]);
                (n, mu, sigma) = update_running_stats(n, mu, sigma, inc_n, inc_mu, inc_sigma);
                let idx = (bi * v + vi) * n_patches + p;
                out_n[idx] = n;
                out_mu[idx] = mu;
                out_sigma[idx] = sigma;
            }
        }
    }
    (out_n, out_mu, out_sigma)
}

/// `util.get_output_patch_via_roll`: patch `p`'s "future covariate" slot gets
/// the next `rolls` patches' raw values concatenated (`[rolls*patch_len]`
/// wide), with a wrap-around mask for whatever runs past the end.
fn roll_future(values: &[f32], masks: &[bool], b: usize, v: usize, n_patches: usize, patch_len: usize, rolls: usize) -> (Vec<f32>, Vec<bool>) {
    let w = rolls * patch_len;
    let mut out_v = vec![0f32; b * v * n_patches * w];
    let mut out_m = vec![false; b * v * n_patches * w];
    for bi in 0..b {
        for vi in 0..v {
            for p in 0..n_patches {
                for k in 0..w {
                    let src_patch = p + 1 + k / patch_len;
                    let dst = ((bi * v + vi) * n_patches + p) * w + k;
                    if src_patch >= n_patches {
                        out_m[dst] = true; // wrap: no such patch, masked
                        continue;
                    }
                    let src = ((bi * v + vi) * n_patches + src_patch) * patch_len + (k % patch_len);
                    out_v[dst] = values[src];
                    out_m[dst] = masks[src];
                }
            }
        }
    }
    (out_v, out_m)
}

/// Per-(batch, variate) masked least-squares trend over normalized time
/// `t/context`, applied only where it shrinks the std below `threshold *
/// std_orig` (the reference's own gate against over-fitting a trend to a
/// mostly-flat or noisy series). Returns `(m_trend, c_trend, applied)` per
/// (batch, variate) - `y_detrended = y - (m*t_norm + c)`.
fn linear_detrend_fit(vals: &[f32], mask: &[bool], context: usize, threshold: f32) -> (f32, f32, bool) {
    let (mut n_v, mut sum_t, mut sum_t2, mut sum_y, mut sum_ty) = (0f32, 0f32, 0f32, 0f32, 0f32);
    for i in 0..context {
        if mask[i] {
            continue;
        }
        // t = -(context-1) ..= 0, normalized by /context - the reference's
        // own convention (the LAST context step is t=0).
        let t = (i as f32 - (context - 1) as f32) / context as f32;
        n_v += 1.0;
        sum_t += t;
        sum_t2 += t * t;
        sum_y += vals[i];
        sum_ty += t * vals[i];
    }
    let det = n_v * sum_t2 - sum_t * sum_t;
    let (m, c) = if det == 0.0 {
        (0.0, if n_v > 0.0 { sum_y / n_v.max(1.0) } else { 0.0 })
    } else {
        let m = (n_v * sum_ty - sum_t * sum_y) / det;
        (m, (sum_y - m * sum_t) / n_v.max(1.0))
    };

    let mean_y = sum_y / n_v.max(1.0);
    let mut sum_y2 = 0f32;
    let mut sum_yd = 0f32;
    let mut sum_yd2 = 0f32;
    for i in 0..context {
        if mask[i] {
            continue;
        }
        let t = (i as f32 - (context - 1) as f32) / context as f32;
        sum_y2 += vals[i] * vals[i];
        let yd = vals[i] - (m * t + c);
        sum_yd += yd;
        sum_yd2 += yd * yd;
    }
    let var_orig = (sum_y2 / n_v.max(1.0) - mean_y * mean_y).max(0.0);
    let std_orig = var_orig.sqrt();
    let mean_yd = sum_yd / n_v.max(1.0);
    let var_det = (sum_yd2 / n_v.max(1.0) - mean_yd * mean_yd).max(0.0);
    let std_det = var_det.sqrt();

    (m, c, std_det < threshold * std_orig)
}

/// One (batch, variate)'s full detrend: fit on the context, subtract from
/// both the context and (for a past-and-future covariate) the future part
/// using the SAME fitted line evaluated at `t = 1..=horizon` normalized by
/// context length (not `context+horizon`) - the reference's own convention.
pub struct Trend {
    pub m: f32,
    pub c: f32,
    pub applied: bool,
}

pub fn fit_and_apply_detrend(ctx: &mut [f32], ctx_mask: &[bool], future: Option<&mut [f32]>, future_mask: Option<&[bool]>, context: usize, threshold: f32) -> Trend {
    let (m, c, applied) = linear_detrend_fit(ctx, ctx_mask, context, threshold);
    if !applied {
        return Trend { m, c, applied };
    }
    for (i, x) in ctx.iter_mut().enumerate().take(context) {
        let t = (i as f32 - (context - 1) as f32) / context as f32;
        *x -= m * t + c;
    }
    if let (Some(future), Some(future_mask)) = (future, future_mask) {
        for (i, fv) in future.iter_mut().enumerate() {
            if future_mask[i] {
                continue;
            }
            let t = (i + 1) as f32 / context as f32;
            *fv -= m * t + c;
        }
    }
    Trend { m, c, applied }
}

/// `util.stitch_patches`: linear cross-fade over the `overlap =
/// stitch_extract_len - input_patch_len` window between consecutive forecast
/// patches. `patch_preds` is `[b, v, num_forecast_patches, extract_len,
/// num_quantiles]`; output is `[b, v, num_forecast_patches*patch_len +
/// overlap, num_quantiles]` (truncated to `horizon` by the caller).
///
/// Two easy-to-miss pieces the reference's own shape hides: at exactly ONE
/// forecast patch (`stitch_weights`/pairing never run at all) the WHOLE
/// `extract_len`-wide patch is returned verbatim, not just its first
/// `patch_len`; and even with several patches, the LAST one's own overlap
/// tail (`[patch_len..extract_len)`) is appended once at the very end, on top
/// of the per-pair stitching - both are real output positions a "first_chunk
/// + stitched middles" reading of the algorithm silently drops.
pub fn stitch_patches(patch_preds: &[f32], b: usize, v: usize, num_forecast_patches: usize, extract_len: usize, patch_len: usize, num_quantiles: usize) -> Vec<f32> {
    let overlap = extract_len - patch_len;
    let out_len = num_forecast_patches * patch_len + overlap;
    let mut out = vec![0f32; b * v * out_len * num_quantiles];
    let patch_stride = extract_len * num_quantiles;
    let bv_stride = num_forecast_patches * patch_stride;

    if num_forecast_patches == 1 {
        // out_len == extract_len here - the single patch, verbatim.
        out.copy_from_slice(&patch_preds[..b * v * extract_len * num_quantiles]);
        return out;
    }

    // torch.linspace(1.0, 0.0, overlap): both ends inclusive, so the step is
    // 1/(overlap-1), NOT 1/overlap.
    let lin_step = 1.0 / (overlap.max(2) - 1) as f32;

    for bi in 0..b {
        for vi in 0..v {
            let base = (bi * v + vi) * bv_stride;
            let obase = (bi * v + vi) * out_len * num_quantiles;
            // first_chunk: patch 0's first `patch_len` steps, verbatim.
            for i in 0..patch_len {
                for q in 0..num_quantiles {
                    out[obase + i * num_quantiles + q] = patch_preds[base + i * num_quantiles + q];
                }
            }
            // stitched overlaps + middles, one pass per consecutive pair.
            for p in 0..num_forecast_patches - 1 {
                let prev = base + p * patch_stride;
                let next = base + (p + 1) * patch_stride;
                for k in 0..overlap {
                    let w = 1.0 - k as f32 * lin_step;
                    let oi = obase + (p * patch_len + patch_len + k) * num_quantiles;
                    for q in 0..num_quantiles {
                        let a = patch_preds[prev + (patch_len + k) * num_quantiles + q]; // prev patch's overlap tail
                        let bq = patch_preds[next + k * num_quantiles + q]; // next patch's overlap head
                        out[oi + q] = w * a + (1.0 - w) * bq;
                    }
                }
                // middle: next patch's [overlap..patch_len) region, verbatim.
                for i in overlap..patch_len {
                    let oi = obase + (p * patch_len + patch_len + i) * num_quantiles;
                    for q in 0..num_quantiles {
                        out[oi + q] = patch_preds[next + i * num_quantiles + q];
                    }
                }
            }
            // tail: the LAST patch's own overlap region, verbatim - appended
            // once, past every stitched pair.
            let last = base + (num_forecast_patches - 1) * patch_stride;
            let tail_off = obase + ((num_forecast_patches - 1) * patch_len + patch_len) * num_quantiles;
            for k in 0..overlap {
                for q in 0..num_quantiles {
                    out[tail_off + k * num_quantiles + q] = patch_preds[last + (patch_len + k) * num_quantiles + q];
                }
            }
        }
    }
    out
}

/// CPM iterative RevIN refinement (`cpm_revin_refine.cpm_iterative_revin_refine`):
/// a second sequential scan over patches, run ONLY where `patch_cpm_mask` is
/// set (the horizon patches), advancing the running stats using the model's
/// OWN median-quantile prediction instead of unavailable ground truth.
///
/// `raw_logits` is `[b, v, n_patches, output_patch_len, num_quantiles]`;
/// `running_{n,mu,sigma}` are the PRE-refine running stats from
/// [`running_stats`]; `patch_cpm_mask` is `[n_patches]` (shared across
/// batch/variate - `false` on context patches, `true` on horizon patches).
/// Returns `(refined_mu, refined_sigma)`, `[b, v, n_patches]`, equal to the
/// input running stats wherever `patch_cpm_mask` is `false`.
#[allow(clippy::too_many_arguments)]
pub fn cpm_iterative_revin_refine(raw_logits: &[f32], running_n: &[f32], running_mu: &[f32], running_sigma: &[f32], patch_cpm_mask: &[bool], b: usize, v: usize, n_patches: usize, output_patch_len: usize, num_quantiles: usize, rolls: usize, value_clip: f32) -> (Vec<f32>, Vec<f32>) {
    let median_q = num_quantiles / 2;
    let mut out_mu = running_mu.to_vec();
    let mut out_sigma = running_sigma.to_vec();
    let patch_len = output_patch_len / rolls;
    for bi in 0..b {
        for vi in 0..v {
            let (mut carry_n, mut carry_mu, mut carry_sigma) = (0f32, 0f32, 0f32);
            // `anchor[r][p]`, `r in 0..rolls, p in 0..patch_len`: the most
            // recently COMPLETED cycle's own reconstructed prediction grid -
            // consumed one row per step (indexed by `block_offset`) and
            // replaced WHOLESALE (every row at once) whenever a full
            // `rolls`-step cycle finishes. Crucially (and easy to miss reading
            // the algorithm at a glance), the cycle-completion condition
            // (`new_block_offset == 0`) is true at EVERY non-CPM (context)
            // step too, not only at CPM ones - `block_offset` resets to 0
            // unconditionally off-CPM, which trivially satisfies "== 0". So
            // the anchor keeps getting refreshed from each CONTEXT patch's
            // own prediction all the way through the context region, and by
            // the time the first real CPM step runs, the anchor already holds
            // the last context patch's reconstruction - never all-zero in
            // practice. Skipping the anchor update off-CPM (as if only CPM
            // steps could touch it) reproduces every context-patch value
            // (which only reads the pre-refine running stats) while silently
            // starting CPM refinement from a stale all-zero anchor - the
            // exact failure a resblock_input-only parity gate cannot see.
            let mut anchor = vec![vec![0f32; patch_len]; rolls];
            let mut block_offset = 0usize;
            for (p, &is_cpm) in patch_cpm_mask.iter().enumerate().take(n_patches) {
                let idx = (bi * v + vi) * n_patches + p;

                let pred = &anchor[block_offset];
                let (inc_n, inc_mu, inc_sigma) = masked_patch_stats(pred, &vec![false; patch_len]);
                let (new_n, new_mu, new_sigma) = update_running_stats(carry_n, carry_mu, carry_sigma, inc_n, inc_mu, inc_sigma);

                let (this_n, this_mu, this_sigma) = if is_cpm { (new_n, new_mu, new_sigma) } else { (running_n[idx], running_mu[idx], running_sigma[idx]) };
                out_mu[idx] = this_mu;
                out_sigma[idx] = this_sigma;

                let new_block_offset = if is_cpm { (block_offset + 1) % rolls } else { 0 };
                if new_block_offset == 0 {
                    // Reverse-RevIN this patch's own FULL (rolls*patch_len)
                    // median-quantile logits with THIS step's stats -
                    // becomes the anchor's next full replacement grid.
                    let base = ((bi * v + vi) * n_patches + p) * output_patch_len * num_quantiles;
                    for (r, row) in anchor.iter_mut().enumerate().take(rolls) {
                        for (i, cell) in row.iter_mut().enumerate().take(patch_len) {
                            let o = r * patch_len + i;
                            let logit = raw_logits[base + o * num_quantiles + median_q];
                            *cell = revin(logit, this_mu, this_sigma, true).clamp(-value_clip, value_clip);
                        }
                    }
                }

                carry_n = this_n;
                carry_mu = this_mu;
                carry_sigma = this_sigma;
                block_offset = new_block_offset;
            }
        }
    }
    (out_mu, out_sigma)
}

/// Build `core_forward`'s input from raw (unpatched) target / past-only /
/// past-and-future covariate panels: linear detrend, patch, running RevIN,
/// future-covariate rolling, and the `[values | future | mask | future_mask]`
/// concatenation `pre_transformer_resblock` expects. No left-padding path yet
/// (`context` must already be a multiple of `input_patch_len` - the ledgered
/// gap; every context this crate is tested against today satisfies it).
///
/// `target`/`past_only`/`past_future` are `[b, num_*, context]` (`past_future`
/// is `[b, num_past_future, context+horizon]`); all fully observed (no NaNs) -
/// per-step missing-value masking is a further ledgered gap.
pub struct BuiltInput {
    pub resblock_input: Vec<f32>,
    pub patch_mask: Vec<bool>,
    pub running_n: Vec<f32>,
    pub running_mu: Vec<f32>,
    pub running_sigma: Vec<f32>,
    pub trends: Vec<Trend>, // one per variate, batch-major then variate (same order core_forward's rows use)
    pub num_context_patches: usize,
    pub num_horizon_patches: usize,
}

pub fn build_input(cfg: &Timesfm3Config, shape: DecodeShape, target: &[f32], past_only: &[f32], past_future: &[f32]) -> BuiltInput {
    let (b, context, horizon) = (shape.batch, shape.context, shape.horizon);
    assert_eq!(context % cfg.input_patch_len, 0, "left-padding to a patch boundary is not implemented yet");
    let v = shape.num_variates();
    let patch_len = cfg.input_patch_len;
    let rolls = cfg.rolls();

    // ---- assemble + detrend the raw context (and past-future's future) ----
    let mut ctx = vec![0f32; b * v * context];
    for bi in 0..b {
        for ti in 0..shape.num_target {
            let vi = ti;
            ctx[(bi * v + vi) * context..(bi * v + vi) * context + context].copy_from_slice(&target[(bi * shape.num_target + ti) * context..(bi * shape.num_target + ti) * context + context]);
        }
        for pi in 0..shape.num_past_only {
            let vi = shape.num_target + pi;
            ctx[(bi * v + vi) * context..(bi * v + vi) * context + context].copy_from_slice(&past_only[(bi * shape.num_past_only + pi) * context..(bi * shape.num_past_only + pi) * context + context]);
        }
        for fi in 0..shape.num_past_future {
            let vi = shape.num_target + shape.num_past_only + fi;
            let src = (bi * shape.num_past_future + fi) * (context + horizon);
            ctx[(bi * v + vi) * context..(bi * v + vi) * context + context].copy_from_slice(&past_future[src..src + context]);
        }
    }
    let ctx_mask = vec![false; b * v * context]; // no NaNs supported yet - see the struct doc

    let stitch_extract_len = cfg.stitch_extract_len();
    let overlap = stitch_extract_len - patch_len;
    let num_forecast_patches = (horizon.saturating_sub(overlap)).div_ceil(patch_len).max(1);
    let num_horizon_patches = num_forecast_patches + rolls - 1;
    let padded_horizon = num_horizon_patches * patch_len;
    let num_context_patches = context / patch_len;

    let mut trends = Vec::with_capacity(b * v);
    let mut future_ctx = vec![0f32; b * v * context]; // detrended copy for context
    let mut future_hor = vec![0f32; b * v * padded_horizon]; // future values for past-future covariates only
    let mut future_hor_mask = vec![true; b * v * padded_horizon]; // target/past-only stay fully masked over the horizon
    for bi in 0..b {
        for vi in 0..v {
            let off = (bi * v + vi) * context;
            let mut c = ctx[off..off + context].to_vec();
            let is_pf = vi >= shape.num_target + shape.num_past_only;
            let trend = if is_pf {
                let fi = vi - shape.num_target - shape.num_past_only;
                let src = (bi * shape.num_past_future + fi) * (context + horizon) + context;
                let mut fut: Vec<f32> = past_future[src..src + horizon].to_vec();
                let fut_mask = vec![false; horizon];
                let t = fit_and_apply_detrend(&mut c, &ctx_mask[off..off + context], Some(&mut fut), Some(&fut_mask), context, cfg.linear_detrending_threshold);
                if t.applied {
                    let hoff = (bi * v + vi) * padded_horizon;
                    future_hor[hoff..hoff + horizon].copy_from_slice(&fut);
                    for k in 0..horizon {
                        future_hor_mask[hoff + k] = false;
                    }
                } else {
                    let hoff = (bi * v + vi) * padded_horizon;
                    let fut_orig = &past_future[src..src + horizon];
                    future_hor[hoff..hoff + horizon].copy_from_slice(fut_orig);
                    for k in 0..horizon {
                        future_hor_mask[hoff + k] = false;
                    }
                }
                t
            } else {
                fit_and_apply_detrend(&mut c, &ctx_mask[off..off + context], None, None, context, cfg.linear_detrending_threshold)
            };
            future_ctx[off..off + context].copy_from_slice(&c);
            trends.push(trend);
        }
    }

    // ---- concat ctx+horizon, patch, running RevIN ----
    let total_t = context + padded_horizon;
    let n_patches = num_context_patches + num_horizon_patches;
    let mut all_vals = vec![0f32; b * v * total_t];
    let mut all_masks = vec![false; b * v * total_t];
    for bi in 0..b {
        for vi in 0..v {
            let ob = (bi * v + vi) * total_t;
            let cb = (bi * v + vi) * context;
            all_vals[ob..ob + context].copy_from_slice(&future_ctx[cb..cb + context]);
            all_masks[ob..ob + context].copy_from_slice(&ctx_mask[cb..cb + context]);
            let hb = (bi * v + vi) * padded_horizon;
            all_vals[ob + context..ob + total_t].copy_from_slice(&future_hor[hb..hb + padded_horizon]);
            all_masks[ob + context..ob + total_t].copy_from_slice(&future_hor_mask[hb..hb + padded_horizon]);
        }
    }
    // zero out masked positions, matching `ctx_vals = where(mask, 0, ctx_vals)`.
    for i in 0..all_vals.len() {
        if all_masks[i] {
            all_vals[i] = 0.0;
        }
    }

    let (running_n, running_mu, running_sigma) = running_stats(&all_vals, &all_masks, b, v, n_patches, patch_len);
    let mut values_bvnp = vec![0f32; b * v * n_patches * patch_len];
    for bi in 0..b {
        for vi in 0..v {
            for p in 0..n_patches {
                let idx = (bi * v + vi) * n_patches + p;
                for i in 0..patch_len {
                    let src = ((bi * v + vi) * n_patches + p) * patch_len + i;
                    values_bvnp[src] = if all_masks[src] { 0.0 } else { revin(all_vals[src], running_mu[idx], running_sigma[idx], false) };
                }
            }
        }
    }

    // patch_is_target: true for target AND past-only variates (both get
    // their future-covariate slot suppressed - the reference's own rule).
    let num_pt = shape.num_target + shape.num_past_only;

    let (fcov_raw, fcov_wrap) = roll_future(&all_vals, &all_masks, b, v, n_patches, patch_len, rolls);
    let mut values_fcov = vec![0f32; fcov_raw.len()];
    let fw = rolls * patch_len;
    for bi in 0..b {
        for vi in 0..v {
            let is_target_like = vi < num_pt;
            for p in 0..n_patches {
                let idx = (bi * v + vi) * n_patches + p;
                for k in 0..fw {
                    let i = ((bi * v + vi) * n_patches + p) * fw + k;
                    let masked = fcov_wrap[i] || is_target_like;
                    values_fcov[i] = if masked { 0.0 } else { revin(fcov_raw[i], running_mu[idx], running_sigma[idx], false) };
                }
            }
        }
    }

    // resblock_input = [values(patch_len) | fcov(fw) | mask(patch_len) | fcov_mask(fw)], per (b,v,p).
    let resblock_in_dim = cfg.resblock_in_dim();
    let mut resblock_input = vec![0f32; b * v * n_patches * resblock_in_dim];
    let mut patch_mask_raw = vec![false; b * v * n_patches]; // masks.all(dim=3): true iff EVERY position in [values|fcov|masks|fcov_masks] is masked
    for bi in 0..b {
        for vi in 0..v {
            let is_target_like = vi < num_pt;
            for p in 0..n_patches {
                let row = ((bi * v + vi) * n_patches + p) * resblock_in_dim;
                let src_v = ((bi * v + vi) * n_patches + p) * patch_len;
                let src_f = ((bi * v + vi) * n_patches + p) * fw;
                let mut all_masked = true;
                for i in 0..patch_len {
                    resblock_input[row + i] = values_bvnp[src_v + i];
                    if !all_masks[src_v + i] {
                        all_masked = false;
                    }
                }
                for k in 0..fw {
                    resblock_input[row + patch_len + k] = values_fcov[src_f + k];
                    let m = fcov_wrap[src_f + k] || is_target_like;
                    if !m {
                        all_masked = false;
                    }
                }
                for i in 0..patch_len {
                    resblock_input[row + patch_len + fw + i] = if all_masks[src_v + i] { 1.0 } else { 0.0 };
                }
                for k in 0..fw {
                    let m = fcov_wrap[src_f + k] || is_target_like;
                    resblock_input[row + patch_len + fw + patch_len + k] = if m { 1.0 } else { 0.0 };
                }
                patch_mask_raw[(bi * v + vi) * n_patches + p] = all_masked;
            }
        }
    }

    // Inference-only trick: only mask LEADING fully-masked patches
    // (`cumprod` along the patch axis) - horizon patches stay visible.
    let mut patch_mask = vec![false; b * v * n_patches];
    for bi in 0..b {
        for vi in 0..v {
            let mut still_leading = true;
            for p in 0..n_patches {
                let idx = (bi * v + vi) * n_patches + p;
                still_leading = still_leading && patch_mask_raw[idx];
                patch_mask[idx] = still_leading;
            }
        }
    }

    BuiltInput { resblock_input, patch_mask, running_n, running_mu, running_sigma, trends, num_context_patches, num_horizon_patches }
}

/// The tail of `decode()`: CPM refinement, RevIN-reverse, stitching and
/// trend re-addition, given `core_forward`'s raw logits and
/// [`build_input`]'s RevIN stats/trends. Returns `[b, v, horizon,
/// num_quantiles]`.
pub fn postprocess(cfg: &Timesfm3Config, shape: DecodeShape, built: &BuiltInput, raw_logits: &[f32]) -> Vec<f32> {
    let (b, horizon) = (shape.batch, shape.horizon);
    let v = shape.num_variates();
    let n_patches = built.num_context_patches + built.num_horizon_patches;
    let mut patch_cpm_mask = vec![false; n_patches];
    patch_cpm_mask[built.num_context_patches..n_patches].fill(true);

    let (mu, sigma) = if cfg.use_iterative_cpm_revin {
        cpm_iterative_revin_refine(raw_logits, &built.running_n, &built.running_mu, &built.running_sigma, &patch_cpm_mask, b, v, n_patches, cfg.output_patch_len, cfg.num_quantiles, cfg.rolls(), cfg.value_clip)
    } else {
        (built.running_mu.clone(), built.running_sigma.clone())
    };

    let mut revin_logits = vec![0f32; raw_logits.len()];
    for bi in 0..b {
        for vi in 0..v {
            for p in 0..n_patches {
                let idx = (bi * v + vi) * n_patches + p;
                let base = idx * cfg.output_patch_len * cfg.num_quantiles;
                for o in 0..cfg.output_patch_len {
                    for q in 0..cfg.num_quantiles {
                        let i = base + o * cfg.num_quantiles + q;
                        revin_logits[i] = revin(raw_logits[i], mu[idx], sigma[idx], true).clamp(-cfg.value_clip, cfg.value_clip);
                    }
                }
            }
        }
    }

    let extract_len = cfg.stitch_extract_len();
    let num_forecast_patches = built.num_horizon_patches - (cfg.rolls() - 1);
    let mut patch_preds = vec![0f32; b * v * num_forecast_patches * extract_len * cfg.num_quantiles];
    for bi in 0..b {
        for vi in 0..v {
            for fp in 0..num_forecast_patches {
                let src_patch = built.num_context_patches - 1 + fp;
                let src = ((bi * v + vi) * n_patches + src_patch) * cfg.output_patch_len * cfg.num_quantiles;
                let dst = ((bi * v + vi) * num_forecast_patches + fp) * extract_len * cfg.num_quantiles;
                patch_preds[dst..dst + extract_len * cfg.num_quantiles].copy_from_slice(&revin_logits[src..src + extract_len * cfg.num_quantiles]);
            }
        }
    }
    let stitched = stitch_patches(&patch_preds, b, v, num_forecast_patches, extract_len, cfg.input_patch_len, cfg.num_quantiles);
    let stitched_len = num_forecast_patches * cfg.input_patch_len + (extract_len - cfg.input_patch_len);

    let mut out = vec![0f32; b * v * horizon * cfg.num_quantiles];
    for bi in 0..b {
        for vi in 0..v {
            let trend = &built.trends[bi * v + vi];
            for h in 0..horizon {
                let src = ((bi * v + vi) * stitched_len + h) * cfg.num_quantiles;
                let dst = ((bi * v + vi) * horizon + h) * cfg.num_quantiles;
                let add = if trend.applied { trend.m * ((h + 1) as f32 / shape.context as f32) + trend.c } else { 0.0 };
                for q in 0..cfg.num_quantiles {
                    out[dst + q] = stitched[src + q] + add;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revin_round_trips_and_guards_a_near_zero_sigma() {
        assert!((revin(revin(3.0, 1.0, 2.0, false), 1.0, 2.0, true) - 3.0).abs() < 1e-6);
        // sigma below the tolerance is treated as exactly 1.0, not clamped up to it.
        assert_eq!(revin(5.0, 2.0, 0.0, false), 3.0);
    }

    #[test]
    fn masked_patch_stats_ignores_masked_entries() {
        let (n, mu, sigma) = masked_patch_stats(&[1.0, 100.0, 3.0], &[false, true, false]);
        assert_eq!(n, 2.0);
        assert!((mu - 2.0).abs() < 1e-6); // mean of {1,3}, 100 excluded
        assert!((sigma - 1.0).abs() < 1e-6); // std of {1,3} around mean 2
    }

    #[test]
    fn masked_patch_stats_of_an_entirely_masked_patch_is_zero() {
        let (n, mu, sigma) = masked_patch_stats(&[9.0, 9.0], &[true, true]);
        assert_eq!((n, mu, sigma), (0.0, 0.0, 0.0));
    }

    #[test]
    fn update_running_stats_freezes_on_a_zero_increment() {
        let (n, mu, sigma) = update_running_stats(10.0, 5.0, 2.0, 0.0, 0.0, 0.0);
        assert_eq!((n, mu, sigma), (10.0, 5.0, 2.0));
    }

    #[test]
    fn stitch_patches_at_one_forecast_patch_returns_it_verbatim() {
        // b=1,v=1,num_forecast_patches=1,extract_len=3,patch_len=2,q=1: the
        // whole extract_len-wide patch must survive, not just its first
        // patch_len entries - this is the exact bug this test guards.
        let preds = vec![1.0, 2.0, 3.0];
        let out = stitch_patches(&preds, 1, 1, 1, 3, 2, 1);
        assert_eq!(out, preds);
    }

    #[test]
    fn stitch_patches_cross_fades_the_overlap_and_appends_the_final_tail() {
        // b=1,v=1, 2 forecast patches, extract_len=4, patch_len=2 (overlap=2), q=1.
        // patch0 = [0,10,20,30], patch1 = [100,110,120,130].
        // out_len = 2*2+2 = 6.
        let preds = vec![0.0, 10.0, 20.0, 30.0, 100.0, 110.0, 120.0, 130.0];
        let out = stitch_patches(&preds, 1, 1, 2, 4, 2, 1);
        assert_eq!(out.len(), 6);
        // first_chunk: patch0[0..2] verbatim.
        assert_eq!(&out[0..2], &[0.0, 10.0]);
        // overlap (k=0,1): linspace(1,0,2) = [1,0] -> w=1 picks patch0's
        // overlap entirely at k=0, w=0 picks patch1's overlap entirely at k=1.
        assert_eq!(out[2], 20.0); // w=1: all patch0[2]
        assert_eq!(out[3], 110.0); // w=0: all patch1[1] (patch1's overlap head is index 0..overlap=[100,110])
        // tail: the LAST patch's own overlap region (patch1[2..4]), verbatim -
        // appended once, past every stitched pair.
        assert_eq!(&out[4..6], &[120.0, 130.0]);
    }
}
