// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Fused conv2d -> per-channel affine (BatchNorm-eval collapsed) -> SiLU.
// Identical convolution to conv2d.wgsl (bias-free, NCHW, square KxK, generic
// stride & implicit zero-pad), then in the SAME pass applies a per-OUTPUT-channel
// affine and the SiLU activation, so the activation is produced without the
// extra bn_eval + silu memory round-trips:
//   z = conv(x,w) * scale[co] + bias[co]
//   y = z / (1 + exp(-z))                       // SiLU
// where (scale,bias) is the BatchNorm-eval transform pre-collapsed per channel:
//   scale[c] = gamma[c] / sqrt(run_var[c] + 1e-5)
//   bias[c]  = beta[c] - run_mean[c] * scale[c]
// packed as sb[2c]=scale[c], sb[2c+1]=bias[c]. Four storage buffers (x,w,sb,y).

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
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read>       sb: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cout * p.Ho * p.Wo;
    if (idx >= total) { return; }

    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let co = t2 % p.Cout;
    let n  = t2 / p.Cout;

    var acc = 0.0;
    for (var ci: u32 = 0u; ci < p.Cin; ci = ci + 1u) {
        for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
            let hi_b = ho * p.stride + kh;
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
    let z = acc * sb[2u * co] + sb[2u * co + 1u];
    y[idx] = z / (1.0 + exp(-z));
}
