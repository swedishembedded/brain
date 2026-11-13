// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of the SSI + gradient-matching loss wrt the prediction, from per-image folded weights
// @how   one thread per pixel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
//   d_pred_i = m_i ( w_r r_i + w_g g_i + w_t t_i + w_p p_i + w_0 )
//
// with five weights per image, w = [N, 5] = (w_r, w_g, w_t, w_p, w_0), that `zipdepth::loss` folds on the host from
// the solved fit and the per-image reductions (r from ssi_resid.wgsl, g from grad_match.wgsl). The loss's gradient
// is the sum of the SSI part (s r_i / |M|, the envelope theorem) and the gradient-matching part, which flows through
// the fit as well: s g_i + (ds/dp_i) sum_j g_j p_j + (db/dp_i) sum_j g_j, where ds/dp_i and db/dp_i are affine in
// (p_i, t_i); folding them is what leaves the per-pixel work this kernel does.
//
//   pred, tgt, mask, resid, g : [N * hw]     w : [N, 5]
//   d_pred : [N * hw]  read_write

struct Params {
    total: u32,   // N * hw
    hw: u32,      // pixels per image
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:    array<f32>;
@group(0) @binding(3) var<storage, read>       mask:   array<f32>;
@group(0) @binding(4) var<storage, read>       resid:  array<f32>;
@group(0) @binding(5) var<storage, read>       g:      array<f32>;
@group(0) @binding(6) var<storage, read>       w:      array<f32>;
@group(0) @binding(7) var<storage, read_write> d_pred: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let b = (idx / p.hw) * 5u;
    d_pred[idx] = mask[idx] * (w[b] * resid[idx] + w[b + 1u] * g[idx] + w[b + 2u] * tgt[idx] + w[b + 3u] * pred[idx] + w[b + 4u]);
}
