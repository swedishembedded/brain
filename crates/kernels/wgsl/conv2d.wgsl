// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// 2D convolution forward (bias-free), NCHW layout, square KxK kernel.
//   x : [N, Cin,  H,  W]   row-major   idx = ((n*Cin + ci)*H + hi)*W + wi
//   w : [Cout, Cin, K, K]  row-major   idx = ((co*Cin + ci)*K + kh)*K + kw
//   y : [N, Cout, Ho, Wo]  row-major   idx = ((n*Cout + co)*Ho + ho)*Wo + wo
//
// One invocation per OUTPUT element. Generic stride & zero-pad (implicit):
// taps whose input coordinate falls outside [0,H)/[0,W) are skipped, which is
// exactly the contribution of a zero-padded border.
//   Ho = (H + 2*pad - K)/stride + 1   (likewise Wo)

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
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cout * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (n, co, ho, wo) from the linear index.
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let co = t2 % p.Cout;
    let n  = t2 / p.Cout;

    var acc = 0.0;
    for (var ci: u32 = 0u; ci < p.Cin; ci = ci + 1u) {
        for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
            // hi = ho*stride - pad + kh, computed in signed-ish form via u32.
            let hi_b = ho * p.stride + kh;   // base before subtracting pad
            if (hi_b >= p.pad) {
                let hi = hi_b - p.pad;
                if (hi < p.H) {
                    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                        let wi_b = wo * p.stride + kw;
                        if (wi_b >= p.pad) {
                            let wi = wi_b - p.pad;
                            if (wi < p.W) {
                                let x_idx = ((n * p.Cin + ci) * p.H + hi) * p.W + wi;
                                let w_idx = ((co * p.Cin + ci) * p.K + kh) * p.K + kw;
                                acc = acc + x[x_idx] * w[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    y[idx] = acc;
}
