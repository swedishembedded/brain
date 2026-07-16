// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Depthwise 3D convolution (forward) — the PEG (position-encoding generator)
// of ST-ViViT tokenizers: a per-channel Conv3d over (T,H,W) with a K^3 kernel,
// stride 1, zero-pad. Layout [N,C,T,H,W]; weights [C,K,K,K] (one KxKxK kernel
// per channel, no Cin mixing) + bias[C]. Output size == input size.
//   y[n,c,t,h,w] = bias[c] + sum_{kt,kh,kw} x[n,c,t+kt-pt,h+kh-ps,w+kw-ps]
//                                            * wt[c,kt,kh,kw]   (zero outside)
// The spatial pad `ps` and TEMPORAL low-pad `pt` are independent, so the causal
// PEG (temporal pad (2,0) with K=3: pt=2) and the non-causal PEG (pt=ps=1) both
// map to this kernel. One invocation per OUTPUT element. fp32, wg64, no barriers.

struct Params {
    N: u32, C: u32, T: u32, H: u32, W: u32,
    K: u32, ps: u32, pt: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       wt:   array<f32>;
@group(0) @binding(3) var<storage, read>       bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let thw = p.T * p.H * p.W;
    if (idx >= p.N * p.C * thw) { return; }
    let w = idx % p.W;
    let h = (idx / p.W) % p.H;
    let t = (idx / (p.W * p.H)) % p.T;
    let c = (idx / thw) % p.C;
    let n = idx / (p.C * thw);
    let kk = p.K * p.K * p.K;
    var acc = bias[c];
    for (var kt: u32 = 0u; kt < p.K; kt = kt + 1u) {
        let it = t + kt;
        if (it >= p.pt && it - p.pt < p.T) {
            let ti = it - p.pt;
            for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
                let ih = h + kh;
                if (ih >= p.ps && ih - p.ps < p.H) {
                    let hi = ih - p.ps;
                    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                        let iw = w + kw;
                        if (iw >= p.ps && iw - p.ps < p.W) {
                            let wi = iw - p.ps;
                            let xi = ((((n * p.C + c) * p.T + ti) * p.H + hi) * p.W) + wi;
                            let wti = ((c * p.K + kt) * p.K + kh) * p.K + kw;
                            acc = acc + x[xi] * wt[wti];
                        }
                    }
                }
            }
        }
    }
    y[idx] = acc;
}
