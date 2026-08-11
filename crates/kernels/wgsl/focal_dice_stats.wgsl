// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-mask reductions for the SAM-style segmentation objective (sigmoid focal loss + dice loss), over `n_masks` masks of `hw` pixels each (row-major [n_masks, hw])
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Per-mask reductions for the SAM-style segmentation objective (sigmoid focal
// loss + dice loss), over `n_masks` masks of `hw` pixels each (row-major
// [n_masks, hw]).
//
// For each mask m this writes four partial sums into `stats[4*m + k]`:
//   k=0  sum_i focal_i      focal_i = a_t * ce_i * (1 - p_t,i)^gamma
//   k=1  sum_i p_i          p = sigmoid(z)
//   k=2  sum_i p_i * t_i
//   k=3  sum_i t_i
// with  ce = max(z,0) - z*t + log(1 + exp(-|z|))   (the stable BCE-with-logits
// form `bce_logits.wgsl` uses — the two MUST agree),
//       p_t = p*t + (1-p)*(1-t),
//       a_t = alpha*t + (1-alpha)*(1-t)            (skipped when alpha < 0).
//
// The HOST composes the scalars the reference computes from these four sums:
//   focal_m = stats[4m+0] / hw                       (`loss.mean(-1)`)
//   dice_m  = 1 - (2*stats[4m+2] + 1) / (stats[4m+1] + stats[4m+3] + 1)
// which is exactly `training/loss_fns.py::sigmoid_focal_loss` /`dice_loss` with
// `loss_on_multimask=True`. Only 4*n_masks floats cross the bus (16 for SAM 2's
// four mask tokens), so the per-mask combination is host arithmetic on a
// constant-size buffer, not host math on a hot path.
//
// The MATCHING gradient is `focal_dice_grad.wgsl`, which re-reads these sums
// (the dice denominator is a whole-mask reduction, so its per-pixel derivative
// needs them). Gather-based and atomic-free: one invocation per MASK, serial
// over `hw`, which is the same shape as `add_chan_bcast_dv` / `bn_dgamma`. It is
// barrier-free on purpose — the reduction is over `hw`, and a cooperative twin
// would be a second kernel needing its own caps gate (see
// .agents/rules/kernels.md §C.2); at SAM 2's four masks a workgroup-per-mask
// variant is a perf item, not a correctness one.
struct Params {
    n_masks: u32,
    hw: u32,
    alpha: f32,   // < 0 disables the alpha_t class weighting
    gamma: f32,   // 0 disables the (1 - p_t)^gamma modulating factor
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:    array<f32>;
@group(0) @binding(3) var<storage, read_write> stats:  array<f32>;  // [n_masks*4]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let m = gidx;
    if (m >= p.n_masks) { return; }

    let base = m * p.hw;
    var s_focal = 0.0;
    var s_p = 0.0;
    var s_pt = 0.0;
    var s_t = 0.0;
    for (var i: u32 = 0u; i < p.hw; i = i + 1u) {
        let z = logits[base + i];
        let t = tgt[base + i];
        // Stable sigmoid, two-branch (matches bce_logits_grad.wgsl).
        var pr = 0.0;
        if (z >= 0.0) {
            pr = 1.0 / (1.0 + exp(-z));
        } else {
            let e = exp(z);
            pr = e / (1.0 + e);
        }
        let ce = max(z, 0.0) - z * t + log(1.0 + exp(-abs(z)));
        let pt = pr * t + (1.0 - pr) * (1.0 - t);
        var mo = 1.0;
        if (p.gamma != 0.0) {
            mo = pow(max(1.0 - pt, 0.0), p.gamma);
        }
        var f = ce * mo;
        if (p.alpha >= 0.0) {
            f = f * (p.alpha * t + (1.0 - p.alpha) * (1.0 - t));
        }
        s_focal = s_focal + f;
        s_p = s_p + pr;
        s_pt = s_pt + pr * t;
        s_t = s_t + t;
    }
    stats[4u * m + 0u] = s_focal;
    stats[4u * m + 1u] = s_p;
    stats[4u * m + 2u] = s_pt;
    stats[4u * m + 3u] = s_t;
}
