// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GroupNorm backward per-group reductions, NCHW — spec:
// docs/world-models/specs/P1.gn.md §4.5. One invocation per (n,g) group
// (N*G threads): n = k/G, g = k%G, cpg = C/G. `dyg` = dy*gamma_c, produced
// by the EXISTING scale_chan kernel (bufs [dy, gb, dyg], params
// [N*C*H*W, C, H*W]). With xhat = (x-mean_k)*rstd_k:
//   S1_k = sum_group dyg
//   S2_k = sum_group dyg * xhat
// Output packing (consumed by gn_dx, keeps it at 4 storage buffers):
//   sums[4k+0] = mean_k (copied from stats)   sums[4k+2] = S1_k
//   sums[4k+1] = rstd_k (copied from stats)   sums[4k+3] = S2_k
// Reduction order: ascending (c,h,w) — determinism contract (spec §11).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       dyg:   array<f32>;
@group(0) @binding(3) var<storage, read>       stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> sums:  array<f32>;

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
    let mean = stats[2u * k];
    let rstd = stats[2u * k + 1u];

    // Single pass over the group, ascending (c,h,w).
    var s1 = 0.0;
    var s2 = 0.0;
    for (var c: u32 = c0; c < c1; c = c + 1u) {
        for (var h: u32 = 0u; h < p.H; h = h + 1u) {
            for (var w: u32 = 0u; w < p.W; w = w + 1u) {
                let i = ((n * p.C + c) * p.H + h) * p.W + w;
                let d = dyg[i];
                s1 = s1 + d;
                s2 = s2 + d * (x[i] - mean) * rstd;
            }
        }
    }
    sums[4u * k + 0u] = mean;
    sums[4u * k + 1u] = rstd;
    sums[4u * k + 2u] = s1;
    sums[4u * k + 3u] = s2;
}
