// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GroupNorm statistics for an NCHW tensor x[N,C,H,W] — spec:
// docs/world-models/specs/P1.gn.md §4.1. One invocation per (n,g) group
// (N*G threads): thread k has n = k/G, g = k%G, cpg = C/G, M = cpg*H*W.
//   mean_k = mean over channels [g*cpg,(g+1)*cpg) x (h,w) of x   (population)
//   rstd_k = 1/sqrt(var_k + eps)   -- eps INSIDE the sqrt, from params (f32 bits)
// Output packing: stats[2k] = mean_k, stats[2k+1] = rstd_k.
// Reduction order: ascending (c,h,w) — determinism contract (spec §11).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    eps: u32,  // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let k = gidx;
    if (k >= p.N * p.G) { return; }
    let n = k / p.G;
    let g = k % p.G;
    let cpg = p.C / p.G;
    let c0 = g * cpg;
    let c1 = c0 + cpg;
    let M = f32(cpg * p.H * p.W);

    // Pass 1: mean, ascending (c,h,w).
    var m = 0.0;
    for (var c: u32 = c0; c < c1; c = c + 1u) {
        for (var h: u32 = 0u; h < p.H; h = h + 1u) {
            for (var w: u32 = 0u; w < p.W; w = w + 1u) {
                m = m + x[((n * p.C + c) * p.H + h) * p.W + w];
            }
        }
    }
    m = m / M;

    // Pass 2: population variance, same order.
    var va = 0.0;
    for (var c: u32 = c0; c < c1; c = c + 1u) {
        for (var h: u32 = 0u; h < p.H; h = h + 1u) {
            for (var w: u32 = 0u; w < p.W; w = w + 1u) {
                let d = x[((n * p.C + c) * p.H + h) * p.W + w] - m;
                va = va + d * d;
            }
        }
    }
    stats[2u * k] = m;
    stats[2u * k + 1u] = inverseSqrt(va / M + bitcast<f32>(p.eps));
}
