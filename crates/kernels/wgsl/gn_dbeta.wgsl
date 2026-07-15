// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GroupNorm backward w.r.t. beta, NCHW — spec:
// docs/world-models/specs/P1.gn.md §4.4. One invocation per channel
// (C threads):
//   dgb[C+c] = dgb[C+c] + sum_{n,h,w} dy       (ACCUMULATES, beta half)
// Writes ONLY dgb[C+c]; gn_dgamma owns the gamma half dgb[c]. The two
// dispatches have disjoint write sets and may share one submit.
// Params keep the uniform {N,C,H,W,G} layout of the gn_* family; G is unused.
// Reduction order: ascending (n,h,w) — determinism contract (spec §11).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dgb: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let c = gidx;
    if (c >= p.C) { return; }

    var s = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        for (var h: u32 = 0u; h < p.H; h = h + 1u) {
            for (var w: u32 = 0u; w < p.W; w = w + 1u) {
                s = s + dy[((n * p.C + c) * p.H + h) * p.W + w];
            }
        }
    }
    dgb[p.C + c] = dgb[p.C + c] + s;
}
