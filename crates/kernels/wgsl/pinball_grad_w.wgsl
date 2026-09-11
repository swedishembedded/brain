// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row weighted pinball (quantile) loss gradient - spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Exact gradient of L = scale * Σ_n out[n] (out from pinball_value_w) w.r.t.
// pred. [N, H, Q] row-major, total = N*H*Q, one invocation per ELEMENT i:
//   k = i / (H*Q); rem = i % (H*Q); t = rem / Q; j = rem % Q
//   y = tgt[k*H+t]; tau = levels[j]; pr = pred[i]
//   dpinball/dpr = -tau        if pr < y
//                   1-tau       if pr > y
//                   tau - 0.5   if pr == y  (subgradient midpoint, matching
//                                            forecast::metrics::pinball_grad)
//   dpred[i] = w[k] * dpinball/dpr / f32(H*Q) * bitcast<f32>(scale)
// `scale` is the upstream scale (e.g. 1/N batch mean) as f32 bits in params
// (gpu_core::f convention) - it appears in THIS kernel only. Divides by the
// SAME H*Q as pinball_value_w (mirrored reduction convention). Exactly 5
// storage buffers.
//

struct Params {
    total: u32,
    h: u32,
    q: u32,
    scale: u32,  // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:    array<f32>;
@group(0) @binding(3) var<storage, read>       levels: array<f32>;
@group(0) @binding(4) var<storage, read>       w:      array<f32>;
@group(0) @binding(5) var<storage, read_write> dpred:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let hq = p.h * p.q;
    let k = i / hq;
    let rem = i % hq;
    let t = rem / p.q;
    let j = rem % p.q;

    let y = tgt[k * p.h + t];
    let tau = levels[j];
    let pr = pred[i];

    var g: f32;
    if (pr < y) {
        g = -tau;
    } else if (pr > y) {
        g = 1.0 - tau;
    } else {
        g = tau - 0.5;
    }
    dpred[i] = w[k] * g / f32(hq) * bitcast<f32>(p.scale);
}
