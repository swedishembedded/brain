// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements numerically defensible validation of model forward passes for its clients. If your
// team needs expertise in deriving test tolerances from floating-point error bounds then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The tolerance for comparing two binary32 evaluations of this decoder's logits.

use std::collections::HashMap;

use gpt2::model::GptConfig;

/// Per-logit bound inside which two binary32 evaluations of the same logits must agree, derived from the round-off
/// bound `k u S / (1 - k u)` ([`brain_testutil::rounding`]) for this model's own shapes and weights, at context
/// length `t`. It serves every comparison of two evaluation orders of this decoder: two backends, or the cached
/// decode against the full recompute.
///
/// A logit is `sum_j W[v, j] h_j` over the final-norm output `h`, with `W` the tied embedding table. A normalised row
/// has unit mean square, so `|x_hat_j| <= sqrt(d)` and `|h_j| <= H = sqrt(d) max|g| + max|b|` for the norm's gain `g`
/// and shift `b`; the terms of logit `v` therefore sum in absolute value to `S_v <= H sum_j |W[v, j]|`. `k` is the
/// number of rounded operations on the longest chain from a token to a logit: every reduction of every stage in turn
/// (norm statistics, projection widths, attention over `t` keys, the 4d feed-forward contraction) plus the head's own
/// `d` adds, and a few for each elementwise step. The count takes each stage to pass relative error on with gain at
/// most one, which is a modelling assumption (the norms and the residual stream keep the stages near it), not a
/// theorem about the whole forward; the tests that use the bound are what check it against real evaluations. Two
/// evaluations each within the accumulation bound of the exact logit are within twice that of each other, and that is
/// what is returned.
///
/// A real miscompile (a barrier some threads never reach, a uniform member read at the wrong offset, a shift the PTX ISA
/// clamps) moves a logit by orders of magnitude more than this.
///
/// perf-number: a numerical-tolerance ratio, not a throughput claim.
pub(crate) fn logit_bounds(cfg: &GptConfig, init: &HashMap<String, Vec<f32>>, t: usize) -> Vec<f32> {
    let (d, ff) = (cfg.d_model as usize, cfg.d_ff as usize);
    let hd = d / cfg.n_heads as usize;
    let norm = 2 * d + 4; // mean, variance, then the rsqrt and scale
    let layer = norm + d + (hd + t + 4 + t) + d + 1 + norm + d + 10 + ff + 1;
    let k = cfg.n_layers as usize * layer + norm + d;

    let max_abs = |name: &str| init[name].iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    let h = (d as f64).sqrt() * max_abs("ln.weight") + max_abs("ln.bias");
    init["tok.weight"]
        .chunks(d)
        .map(|row| brain_testutil::rounding::agreement_bound(k, h * row.iter().map(|w| f64::from(w.abs())).sum::<f64>()) as f32)
        .collect()
}
