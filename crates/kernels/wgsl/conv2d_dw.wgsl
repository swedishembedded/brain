// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// 2D convolution weight gradient. ACCUMULATES into a pre-zeroed buffer.
//   dy : [N, Cout, Ho, Wo]
//   x  : [N, Cin,  H,  W]
//   dw : [Cout, Cin, K, K]   read_write (one invocation per WEIGHT element)
//
// dw[co,ci,kh,kw] = sum over n,ho,wo of dy[n,co,ho,wo] * x[n,ci,hi,wi]
// with the forward index relation hi = ho*stride - pad + kh (bounds-checked,
// taps outside the input map are skipped = implicit zero pad). Each invocation
// owns a distinct weight element, then accumulates like bias_grad.wgsl so the
// pass composes with a prior dw buffer.

struct Params {
    N: u32,
    Cin: u32,
    H: u32,
    W: u32,
    Cout: u32,
    K: u32,
    stride: u32,
    pad: u32,
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
    let total = p.Cout * p.Cin * p.K * p.K;
    if (idx >= total) { return; }

    // Decode weight coordinate (co, ci, kh, kw) from the linear index.
    let kw = idx % p.K;
    let t1 = idx / p.K;
    let kh = t1 % p.K;
    let t2 = t1 / p.K;
    let ci = t2 % p.Cin;
    let co = t2 / p.Cin;

    var acc = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        for (var ho: u32 = 0u; ho < p.Ho; ho = ho + 1u) {
            let hi_b = ho * p.stride + kh;
            if (hi_b >= p.pad) {
                let hi = hi_b - p.pad;
                if (hi < p.H) {
                    for (var wo: u32 = 0u; wo < p.Wo; wo = wo + 1u) {
                        let wi_b = wo * p.stride + kw;
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
