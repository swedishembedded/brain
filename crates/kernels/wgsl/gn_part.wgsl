// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GroupNorm partial reduction (stage 1 of 2) — the parallel replacement for
// gn_stats' serial per-group loop on wide GPUs. P threads per (n,g) group
// (P = threads_per_group in params, typically 64): thread t sums its
// contiguous chunk of the group's M = (C/G)*H*W elements, writing
//   part[(k*P + t)*2]     = sum(x_chunk)
//   part[(k*P + t)*2 + 1] = sum(x_chunk^2)
// gn_stats2 combines the partials into (mean, rstd). Population variance via
// E[x^2] - mean^2 (algebraically equal to gn_stats' two-pass; fp32 rounding
// differs within tolerance). One invocation per (k, t); barrier-free.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    P: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.N * p.G * p.P;
    if (gidx >= total) { return; }
    let t = gidx % p.P;
    let k = gidx / p.P;
    let n = k / p.G;
    let g = k % p.G;

    let cpg = p.C / p.G;
    let m = cpg * p.H * p.W;
    let base = (n * p.C + g * cpg) * p.H * p.W;
    // Contiguous chunk [t*chunk, min((t+1)*chunk, m)).
    let chunk = (m + p.P - 1u) / p.P;
    let lo = t * chunk;
    let hi = min(lo + chunk, m);
    var s = 0.0;
    var s2 = 0.0;
    for (var i: u32 = lo; i < hi; i = i + 1u) {
        let v = x[base + i];
        s = s + v;
        s2 = s2 + v * v;
    }
    part[gidx * 2u] = s;
    part[gidx * 2u + 1u] = s2;
}
