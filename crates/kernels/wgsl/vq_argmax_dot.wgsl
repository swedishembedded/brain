// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Vector-quantization nearest-codebook assignment (COSINE similarity), used by
// GenieRedux-style tokenizers with `use_cosine_sim=True`. Queries x[M,D] and
// codebook cb[K,D] are assumed L2-NORMALISED per row by the caller (project-in
// + rmsnorm, the constant sqrt(D) factor cancels in the argmax); this kernel
// picks the entry maximising the dot product = cosine similarity.
//   out[2m]   = f32(argmax_k <x[m], cb[k]>)   (index, tie -> LOWEST k)
//   out[2m+1] = the maximum dot product
// Same packing/host-composition contract as vq_argmin. fp32, wg64, no barriers.

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
    var best_s = -3.402823e38; // -f32::MAX
    for (var k: u32 = 0u; k < p.k; k = k + 1u) {
        let cbb = k * p.d;
        var s = 0.0;
        for (var i: u32 = 0u; i < p.d; i = i + 1u) {
            s = s + x[xb + i] * cb[cbb + i];
        }
        if (s > best_s) {
            best_s = s;
            best_k = k;
        }
    }
    out[2u * m] = f32(best_k);
    out[2u * m + 1u] = best_s;
}
