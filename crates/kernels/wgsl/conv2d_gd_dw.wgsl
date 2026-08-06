// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped+dilated 2D convolution WEIGHT gradient
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Grouped+dilated 2D convolution WEIGHT gradient. ACCUMULATES into a pre-zeroed
// buffer (same contract as conv2d_dw.wgsl).
//   dy : [N, Cout,      Ho, Wo]
//   x  : [N, Cin,       H,  W]
//   dw : [Cout, Cin/G,  K,  K]   read_write (one invocation per WEIGHT element)
//
// dw[co,cl,kh,kw] = sum over n,ho,wo of dy[n,co,ho,wo] * x[n,ci,hi,wi]
// with hi = ho*stride - pad + kh*dilation (bounds-checked; taps outside the input
// map are skipped = implicit zero pad).
//
// GROUPING: the weight's second axis is cl in [0, Cin/G), NOT the absolute input
// channel. Recover the input channel as ci = (co/cout_g)*cin_g + cl — i.e. the
// group is determined by the OUTPUT channel co, and cl selects within it. Getting
// this backwards is the classic grouped-conv bug: it still runs, still has the
// right shape, and computes the wrong gradient.
//
// Each invocation owns a distinct weight element, then accumulates, so the pass
// composes with a prior dw buffer.

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
    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let total = p.Cout * cin_g * p.K * p.K;
    if (idx >= total) { return; }

    // Decode weight coordinate (co, cl, kh, kw) from the linear index.
    let kw = idx % p.K;
    let t1 = idx / p.K;
    let kh = t1 % p.K;
    let t2 = t1 / p.K;
    let cl = t2 % cin_g;
    let co = t2 / cin_g;

    // The absolute input channel this weight touches: its group comes from co.
    let g  = co / cout_g;
    let ci = g * cin_g + cl;

    var acc = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        for (var ho: u32 = 0u; ho < p.Ho; ho = ho + 1u) {
            let hi_b = ho * p.stride + kh * p.dilation;
            if (hi_b >= p.pad) {
                let hi = hi_b - p.pad;
                if (hi < p.H) {
                    for (var wo: u32 = 0u; wo < p.Wo; wo = wo + 1u) {
                        let wi_b = wo * p.stride + kw * p.dilation;
                        if (wi_b >= p.pad) {
                            let wi = wi_b - p.pad;
                            if (wi < p.W) {
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
