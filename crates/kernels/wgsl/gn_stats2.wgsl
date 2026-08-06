// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm statistics combine (stage 2 of 2, after gn_part)
// @how   one thread per output element, serial inner reduction
// @opt   4
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
//
// GroupNorm statistics combine (stage 2 of 2, after gn_part): one invocation
// per (n,g) group folds its P partial (sum, sumsq) pairs into
//   stats[2k]   = mean = S/M
//   stats[2k+1] = rstd = 1/sqrt(S2/M - mean^2 + eps)
// Population variance via E[x^2] - mean^2 (see gn_part.wgsl). Output layout
// identical to gn_stats.wgsl, so gn_apply/gn_dsum/gn_dx consume it unchanged.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    P: u32,
    eps: u32,  // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part:  array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let k = gid.y * (nwg.x * 64u) + gid.x;
    if (k >= p.N * p.G) { return; }
    let cpg = p.C / p.G;
    let m = f32(cpg * p.H * p.W);
    var s = 0.0;
    var s2 = 0.0;
    for (var t: u32 = 0u; t < p.P; t = t + 1u) {
        s = s + part[(k * p.P + t) * 2u];
        s2 = s2 + part[(k * p.P + t) * 2u + 1u];
    }
    let mean = s / m;
    let va = max(s2 / m - mean * mean, 0.0);
    stats[2u * k] = mean;
    stats[2u * k + 1u] = inverseSqrt(va + bitcast<f32>(p.eps));
}
