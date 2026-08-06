// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nearest-neighbour resize INPUT gradient, NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Nearest-neighbour resize INPUT gradient, NCHW.
//   dy : [N, C, Ho, Wo]
//   dx : [N, C, H,  W ]   read_write (one invocation per INPUT element)
//
// The adjoint of resize_nearest.wgsl: each input pixel accumulates the gradient
// of every output that selected it. Inverting `src = floor(o*in/out)` exactly is
// avoided — the candidate range is computed loosely and each candidate re-tests
// the FORWARD's own rule, so the adjoint holds by construction (the same
// discipline as resize_bilinear_dx / avgpool2d_dx).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    Ho: u32,
    Wo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }
    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let nc = t1 / p.H;

    // Outputs mapping to input `hi` lie in [hi*Ho/H, (hi+1)*Ho/H]; widen by 1 and
    // clamp, then filter exactly.
    var ho_lo = (hi * p.Ho) / p.H;
    if (ho_lo > 0u) { ho_lo = ho_lo - 1u; }
    let ho_hi = min(((hi + 1u) * p.Ho) / p.H + 1u, p.Ho - 1u);
    var wo_lo = (wi * p.Wo) / p.W;
    if (wo_lo > 0u) { wo_lo = wo_lo - 1u; }
    let wo_hi = min(((wi + 1u) * p.Wo) / p.W + 1u, p.Wo - 1u);

    var acc = 0.0;
    for (var ho: u32 = ho_lo; ho <= ho_hi; ho = ho + 1u) {
        if (min((ho * p.H) / p.Ho, p.H - 1u) == hi) {
            for (var wo: u32 = wo_lo; wo <= wo_hi; wo = wo + 1u) {
                if (min((wo * p.W) / p.Wo, p.W - 1u) == wi) {
                    acc = acc + dy[(nc * p.Ho + ho) * p.Wo + wo];
                }
            }
        }
    }
    dx[idx] = acc;
}
