// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm backward per-group reductions, STAGE 2 of 2 - fold the partials
// @how   one thread per output element, serial fold over P partials
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// GroupNorm backward per-group reductions, STAGE 2 of 2 — fold the partials.
//
// One invocation per (n,g) group, summing the `P` partials `gn_dsum_part` wrote
// and emitting the packing `gn_dx` consumes — unchanged from `gn_dsum`, so this
// pair is a drop-in for it:
//
//   sums[4k+0] = mean_k (copied from stats)   sums[4k+2] = S1_k
//   sums[4k+1] = rstd_k (copied from stats)   sums[4k+3] = S2_k
//
// P is small (64), so `N*G` lanes each walking 64 values is not the pathology
// stage 1 exists to fix — the same reason `gn_stats2` is a plain per-group loop.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    P: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part:  array<f32>;
@group(0) @binding(2) var<storage, read>       stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> sums:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let k = gidx;
    if (k >= p.N * p.G) { return; }

    // Ascending partial order — the determinism contract (spec §11) is that a
    // run reproduces, and a fixed fold order is what gives that.
    var s1 = 0.0;
    var s2 = 0.0;
    for (var t: u32 = 0u; t < p.P; t = t + 1u) {
        s1 = s1 + part[(k * p.P + t) * 2u + 0u];
        s2 = s2 + part[(k * p.P + t) * 2u + 1u];
    }
    sums[4u * k + 0u] = stats[2u * k];
    sums[4u * k + 1u] = stats[2u * k + 1u];
    sums[4u * k + 2u] = s1;
    sums[4u * k + 3u] = s2;
}
