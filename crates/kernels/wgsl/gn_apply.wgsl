// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm forward apply, NCHW x[N,C,H,W] - spec
// @how   one thread per output element
// @opt   3
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// GroupNorm forward apply, NCHW x[N,C,H,W]. One invocation per element
// (N*C*H*W threads). With hw = H*W, c = (idx/hw) % C, n = idx/(C*hw),
// cpg = C/G, g = c/cpg, k = n*G + g:
//   y[idx] = gamma_c * (x[idx] - mean_k) * rstd_k + beta_c
// `stats` is gn_stats output: stats[2k]=mean, stats[2k+1]=rstd (eps baked in).
// `gb` is the CONCATENATED [2C] affine buffer: gb[c]=gamma_c, gb[C+c]=beta_c
// (NOT bn's interleaved packing — the concat layout lets scale_chan read
// gamma directly in the backward chain).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       stats: array<f32>;
@group(0) @binding(3) var<storage, read>       gb:    array<f32>;
@group(0) @binding(4) var<storage, read_write> y:     array<f32>;

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
    let xhat = (x[idx] - stats[2u * k]) * stats[2u * k + 1u];
    y[idx] = gb[c] * xhat + gb[p.C + c];
}
