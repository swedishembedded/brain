// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Per-sample weighted MSE partial sums — spec:
// docs/world-models/specs/P1.glue.md §3.4/§4.4. pred/tgt are [N, M] row-major
// (n = samples N, m = M); w is [N]. ONE THREAD PER SAMPLE k (n threads):
//   out[k] = w[k] * ( Σ_{j=0..m-1} (pred[k*m+j] - tgt[k*m+j])^2 ) / f32(m)
// Reduction convention mirrors mse_value.wgsl: divide by the element count
// in-kernel so the host reduction is a PLAIN SUM: L = scale * Σ_k out[k]
// (upstream `scale` lives in mse_grad_w's params / on the host, NOT here).
// Loop ascending j, single division after the loop — determinism contract
// (spec §10). Gradient: mse_grad_w. No dtgt (targets are data), no dw
// (lambda(sigma) is a constant, not trained). Exactly 4 storage buffers.
//

struct Params {
    n: u32,
    m: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:  array<f32>;
@group(0) @binding(3) var<storage, read>       w:    array<f32>;
@group(0) @binding(4) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let k = gid.y * (nwg.x * 64u) + gid.x;
    if (k >= p.n) { return; }
    var acc = 0.0;
    for (var j = 0u; j < p.m; j = j + 1u) {
        let d = pred[k * p.m + j] - tgt[k * p.m + j];
        acc = acc + d * d;
    }
    out[k] = w[k] * acc / f32(p.m);
}
