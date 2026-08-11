// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm backward w.r.t. gamma, NCHW — spec
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// GroupNorm backward w.r.t. gamma, NCHW. One invocation per channel
// (C threads). With cpg = C/G, k = n*G + c/cpg, xhat = (x-mean_k)*rstd_k:
//   dgb[c] = dgb[c] + sum_{n,h,w} dy * xhat        (ACCUMULATES, gamma half)
// Writes ONLY dgb[c] (c < C); gn_dbeta owns the beta half dgb[C+c].
// `stats` is gn_stats output ([2*N*G]: mean|rstd interleaved per (n,g)).
// Reduction order: ascending (n,h,w) — a determinism contract.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       dy:    array<f32>;
@group(0) @binding(3) var<storage, read>       stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgb:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let c = gidx;
    if (c >= p.C) { return; }
    let cpg = p.C / p.G;
    let g = c / cpg;

    var s = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        let k = n * p.G + g;
        let mean = stats[2u * k];
        let rstd = stats[2u * k + 1u];
        for (var h: u32 = 0u; h < p.H; h = h + 1u) {
            for (var w: u32 = 0u; w < p.W; w = w + 1u) {
                let i = ((n * p.C + c) * p.H + h) * p.W + w;
                s = s + dy[i] * (x[i] - mean) * rstd;
            }
        }
    }
    dgb[c] = dgb[c] + s;
}
