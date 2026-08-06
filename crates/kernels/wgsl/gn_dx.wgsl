// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm backward w.r.t. x, NCHW — spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// GroupNorm backward w.r.t. x, NCHW — spec:
// docs/world-models/specs/P1.gn.md §4.6. One invocation per element
// (N*C*H*W threads). With hw = H*W, c = (idx/hw) % C, n = idx/(C*hw),
// cpg = C/G, k = n*G + c/cpg, M = f32(cpg*H*W), `dyg` = dy*gamma (scale_chan),
// `sums` the gn_dsum output ([4*N*G]: mean|rstd|S1|S2 per (n,g)):
//   xhat    = (x[idx] - mean_k) * rstd_k
//   dx[idx] = rstd_k * (dyg[idx] - S1_k/M - xhat * S2_k/M)     (OVERWRITES)

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       dyg:  array<f32>;
@group(0) @binding(3) var<storage, read>       sums: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.N * p.C * p.H * p.W) { return; }
    let hw = p.H * p.W;
    let c = (idx / hw) % p.C;
    let n = idx / (p.C * hw);
    let cpg = p.C / p.G;
    let k = n * p.G + c / cpg;
    let mean = sums[4u * k + 0u];
    let rstd = sums[4u * k + 1u];
    let m = f32(cpg * hw);
    let s1m = sums[4u * k + 2u] / m;
    let s2m = sums[4u * k + 3u] / m;
    let xhat = (x[idx] - mean) * rstd;
    dx[idx] = rstd * (dyg[idx] - s1m - xhat * s2m);
}
