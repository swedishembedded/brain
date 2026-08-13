// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Vector-quantization nearest-codebook assignment (Euclidean)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Vector-quantization nearest-codebook assignment (Euclidean). For each of M
// query vectors x[M,D], find the codebook entry cb[K,D] minimising squared
// L2 distance. One invocation per query m (M threads); inner loop over K*D.
//   out[2m]   = f32(argmin_k ||x[m] - cb[k]||^2)   (index, tie -> LOWEST k)
//   out[2m+1] = the minimum squared distance
// The straight-through estimator + commitment loss compose HOST-side from the
// existing `embed` (codebook gather by idx) and `emb_bwd` (scatter-grad):
// this kernel only produces the assignment. fp32, wg64, no barriers.

struct Params {
    m: u32,
    k: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       cb:  array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let m = gid.y * (nwg.x * 64u) + gid.x;
    if (m >= p.m) { return; }
    let xb = m * p.d;
    var best_k = 0u;
    var best_d = 3.402823e38; // f32::MAX
    for (var k: u32 = 0u; k < p.k; k = k + 1u) {
        let cbb = k * p.d;
        var dist = 0.0;
        for (var i: u32 = 0u; i < p.d; i = i + 1u) {
            let diff = x[xb + i] - cb[cbb + i];
            dist = dist + diff * diff;
        }
        // Strictly-less keeps the LOWEST index on ties (deterministic).
        if (dist < best_d) {
            best_d = dist;
            best_k = k;
        }
    }
    out[2u * m] = f32(best_k);
    out[2u * m + 1u] = best_d;
}
