// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Transposed 2D convolution WEIGHT gradient
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Transposed 2D convolution WEIGHT gradient. ACCUMULATES into a pre-zeroed
// buffer (same contract as convtr1d_dw.wgsl / conv2d_gd_dw.wgsl).
//   dy : [N, Cout,      Ho, Wo]  idx = ((n*Cout + co)*Ho + ho)*Wo + wo
//   x  : [N, Cin,       H,  W]   idx = ((n*Cin + ci)*H + hi)*W + wi
//   dw : [Cin, Cout/G,  K,  K]   read_write, one invocation per WEIGHT element.
//
// dw[ci,co_local,kh,kw] = sum over n,hi,wi of dy[n,co,ho,wo] * x[n,ci,hi,wi]
// with the convtr2d forward relation
//   ho = hi*stride - pad + kh*dilation,  wo = wi*stride - pad + kw*dilation
// (bounds-checked; taps landing outside the output map are the ones `pad` cropped
// away and contribute nothing), and
//   co = g*(Cout/G) + co_local  where  g = ci/(Cin/G).
//
// GROUPING: unlike conv2d_gd_dw, the weight's FIRST axis is the absolute input
// channel ci and the SECOND is co_local in [0, Cout/G) — so the group is derived
// from ci, and co_local selects the output channel within it. Reading the two
// axes the other way round still runs, still has the right shape, and computes
// the wrong gradient.
//
// A weight element is touched by every (n, hi, wi), which is exactly why the
// other parallelization would need an atomic; owning the weight element
// serialises that sum inside one invocation instead. Each invocation owns a
// distinct weight element and then accumulates, so the pass composes with a
// prior dw buffer — the caller MUST zero dw (submit's clear list) before the
// first dispatch of a step.

struct Params {
    N: u32,
    Cin: u32,
    H: u32,
    W: u32,
    Cout: u32,
    K: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    groups: u32,
    Ho: u32,
    Wo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let cout_g = p.Cout / p.groups;
    let total = p.Cin * cout_g * p.K * p.K;
    if (idx >= total) { return; }

    // Decode weight coordinate (ci, co_local, kh, kw) from the linear index.
    let kw = idx % p.K;
    let t1 = idx / p.K;
    let kh = t1 % p.K;
    let t2 = t1 / p.K;
    let co_local = t2 % cout_g;
    let ci = t2 / cout_g;

    // The absolute output channel this weight touches: its group comes from ci.
    let cin_g = p.Cin / p.groups;
    let g  = ci / cin_g;
    let co = g * cout_g + co_local;

    var acc = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        for (var hi: u32 = 0u; hi < p.H; hi = hi + 1u) {
            let ho_b = hi * p.stride + kh * p.dilation;
            if (ho_b >= p.pad) {
                let ho = ho_b - p.pad;
                if (ho < p.Ho) {
                    for (var wi: u32 = 0u; wi < p.W; wi = wi + 1u) {
                        let wo_b = wi * p.stride + kw * p.dilation;
                        if (wo_b >= p.pad) {
                            let wo = wo_b - p.pad;
                            if (wo < p.Wo) {
                                let dy_idx = ((n * p.Cout + co) * p.Ho + ho) * p.Wo + wo;
                                let x_idx = ((n * p.Cin + ci) * p.H + hi) * p.W + wi;
                                acc = acc + dy[dy_idx] * x[x_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    dw[idx] = dw[idx] + acc;
}
