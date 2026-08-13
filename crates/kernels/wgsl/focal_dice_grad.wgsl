// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of the SAM-style segmentation objective w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Gradient of the SAM-style segmentation objective w.r.t. the mask LOGITS.
// The matching forward reduction is `focal_dice_stats.wgsl`; this kernel
// re-reads the four per-mask sums it wrote, because the dice term's denominator
// is a whole-mask reduction and every pixel's derivative depends on it.
//
// The objective, per mask m of `hw` pixels (`mw[m]` is a frozen per-mask weight:
// 1 for a supervised mask, 0 for one the reference's best-mask `argmin`
// deselected — keeping the SELECTION out of the differentiated graph, which is
// what makes it finite-difference-checkable):
//
//   L_m = w_focal * (1/hw) * sum_i a_t,i * ce_i * (1 - p_t,i)^gamma
//       + w_dice  * (1 - (2*S_pt + 1) / (S_p + S_t + 1))
//   L   = sum_m mw[m] * L_m
//
// with p = sigmoid(z), ce = max(z,0) - z*t + log(1+exp(-|z|)),
// p_t = p*t + (1-p)*(1-t), a_t = alpha*t + (1-alpha)*(1-t)  (alpha < 0 = off),
// S_p = sum_i p_i, S_pt = sum_i p_i*t_i, S_t = sum_i t_i  (= stats[4m+1..3]).
//
// Derivatives, all elementwise in z once the three sums are known:
//   dce/dz   = p - t
//   dp_t/dz  = (2t - 1) * p*(1-p)
//   d(1-p_t)^g/dz = -g * (1-p_t)^(g-1) * (2t-1) * p*(1-p)
//   dL_dice/dp_i  = -2*t_i / D + N / D^2,   N = 2*S_pt + 1,  D = S_p + S_t + 1
//
// ASSIGNS `dlogits` (one invocation per (mask, pixel) — every output element is
// written by exactly one thread), so no atomics and no pre-zeroing.
struct Params {
    n_masks: u32,
    hw: u32,
    alpha: f32,     // < 0 disables the alpha_t class weighting
    gamma: f32,     // 0 disables the (1 - p_t)^gamma modulating factor
    w_focal: f32,
    w_dice: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits:  array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:     array<f32>;
@group(0) @binding(3) var<storage, read>       stats:   array<f32>;  // [n_masks*4]
@group(0) @binding(4) var<storage, read>       mw:      array<f32>;  // [n_masks]
@group(0) @binding(5) var<storage, read_write> dlogits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n_masks * p.hw;
    if (gidx >= total) { return; }

    let m = gidx / p.hw;
    let z = logits[gidx];
    let t = tgt[gidx];

    var pr = 0.0;
    if (z >= 0.0) {
        pr = 1.0 / (1.0 + exp(-z));
    } else {
        let e = exp(z);
        pr = e / (1.0 + e);
    }
    let dp = pr * (1.0 - pr);
    let ce = max(z, 0.0) - z * t + log(1.0 + exp(-abs(z)));
    let pt = pr * t + (1.0 - pr) * (1.0 - t);
    let sgn = 2.0 * t - 1.0;

    // ---- focal ----
    var mo = 1.0;
    var dmo = 0.0;
    if (p.gamma != 0.0) {
        let u = max(1.0 - pt, 0.0);
        mo = pow(u, p.gamma);
        dmo = -p.gamma * pow(u, p.gamma - 1.0) * sgn * dp;
    }
    var df = (pr - t) * mo + ce * dmo;
    if (p.alpha >= 0.0) {
        df = df * (p.alpha * t + (1.0 - p.alpha) * (1.0 - t));
    }

    // ---- dice ----
    let s_p = stats[4u * m + 1u];
    let s_pt = stats[4u * m + 2u];
    let s_t = stats[4u * m + 3u];
    let den = s_p + s_t + 1.0;
    let num = 2.0 * s_pt + 1.0;
    let ddice = (-2.0 * t / den + num / (den * den)) * dp;

    dlogits[gidx] = mw[m] * (p.w_focal * df / f32(p.hw) + p.w_dice * ddice);
}
