// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row weighted pinball (quantile) loss partial sums - spec
// @how   one thread per output row, serial inner reduction over (h, q)
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-row weighted mean pinball loss, mirroring mse_value_w's reduction
// convention. pred is [N, H, Q] row-major (n = rows N, h = horizon steps,
// q = quantile levels); tgt is [N, H] (one actual per step, broadcast over
// quantiles); levels is [Q] (the tau values, shared by every row); w is [N].
// ONE THREAD PER ROW k (n threads):
//   out[k] = w[k] * ( Σ_{t=0..h-1} Σ_{j=0..q-1} pinball(pred[(k*h+t)*q+j], tgt[k*h+t], levels[j]) ) / f32(h*q)
// where pinball(pr, y, tau) = max(tau*(y-pr), (tau-1)*(y-pr)).
// Divides by the element count (h*q) in-kernel so the host reduction is a
// PLAIN SUM: L = scale * Σ_k out[k] (upstream `scale` lives in
// pinball_grad_w's params / on the host, NOT here, mirroring mse_value_w).
// Loop ascending t then j - a determinism contract. Gradient: pinball_grad_w.
// No dtgt (targets are data), no dlevels (quantile levels are architecture,
// not trained), no dw (a loss weight is a constant, not trained). Exactly 5
// storage buffers.
//

struct Params {
    n: u32,
    h: u32,
    q: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:    array<f32>;
@group(0) @binding(3) var<storage, read>       levels: array<f32>;
@group(0) @binding(4) var<storage, read>       w:      array<f32>;
@group(0) @binding(5) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let k = gid.y * (nwg.x * 64u) + gid.x;
    if (k >= p.n) { return; }
    var acc = 0.0;
    for (var t = 0u; t < p.h; t = t + 1u) {
        let y = tgt[k * p.h + t];
        for (var j = 0u; j < p.q; j = j + 1u) {
            let pr = pred[(k * p.h + t) * p.q + j];
            let tau = levels[j];
            let e = y - pr;
            acc = acc + max(tau * e, (tau - 1.0) * e);
        }
    }
    out[k] = w[k] * acc / f32(p.h * p.q);
}
