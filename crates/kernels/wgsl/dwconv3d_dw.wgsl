// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Depthwise 3D convolution, WEIGHT gradient
// @how   one thread per output element, 4 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Depthwise 3D convolution, WEIGHT gradient. One invocation per weight element
// wt[c,kt,kh,kw]; sum over all output positions (and batch) of the product of
// the upstream grad and the input it multiplied in the forward:
//   dw[c,kt,kh,kw] = sum_{n,ot,oh,ow} dy[n,c,ot,oh,ow]
//                       * x[n,c, ot+kt-pt, oh+kh-ps, ow+kw-ps]   (in-range only)
// ACCUMULATES into a pre-zeroed dw (matching conv2d_dw). Independent spatial pad
// `ps` and temporal low-pad `pt`. Bias grad = sum of dy over (n,t,h,w) per
// channel: use the existing bias-grad path after a permute. fp32, wg64.

struct Params {
    N: u32, C: u32, T: u32, H: u32, W: u32,
    K: u32, ps: u32, pt: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let kk = p.K * p.K * p.K;
    if (idx >= p.C * kk) { return; }
    let kw = idx % p.K;
    let kh = (idx / p.K) % p.K;
    let kt = (idx / (p.K * p.K)) % p.K;
    let c = idx / kk;
    let thw = p.T * p.H * p.W;
    var acc = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        for (var ot: u32 = 0u; ot < p.T; ot = ot + 1u) {
            let it = ot + kt;
            if (it >= p.pt && it - p.pt < p.T) {
                let ti = it - p.pt;
                for (var oh: u32 = 0u; oh < p.H; oh = oh + 1u) {
                    let ih = oh + kh;
                    if (ih >= p.ps && ih - p.ps < p.H) {
                        let hi = ih - p.ps;
                        for (var ow: u32 = 0u; ow < p.W; ow = ow + 1u) {
                            let iw = ow + kw;
                            if (iw >= p.ps && iw - p.ps < p.W) {
                                let wi = iw - p.ps;
                                let di = ((((n * p.C + c) * p.T + ot) * p.H + oh) * p.W) + ow;
                                let xi = ((((n * p.C + c) * p.T + ti) * p.H + hi) * p.W) + wi;
                                acc = acc + dy[di] * x[xi];
                            }
                        }
                    }
                }
            }
        }
    }
    dw[idx] = dw[idx] + acc;
}
