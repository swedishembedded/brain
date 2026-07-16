// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Depthwise 3D convolution, INPUT gradient (adjoint of dwconv3d). One
// invocation per INPUT element x[n,c,t,h,w]; scatter-gather over the output
// positions whose receptive field covers this input:
//   dx[n,c,t,h,w] = sum_{kt,kh,kw} dy[n,c, t-kt+P, h-kh+P, w-kw+P] * wt[c,kt,kh,kw]
// (output index valid only when in range). Per-channel (no Cin sum). fp32.

struct Params {
    N: u32, C: u32, T: u32, H: u32, W: u32,
    K: u32, pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

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
    var acc = 0.0;
    // output ot with ot + kt - P == t  =>  ot = t - kt + P.
    for (var kt: u32 = 0u; kt < p.K; kt = kt + 1u) {
        let otp = t + p.pad;
        if (otp >= kt) {
            let ot = otp - kt;
            if (ot < p.T) {
                for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
                    let ohp = h + p.pad;
                    if (ohp >= kh) {
                        let oh = ohp - kh;
                        if (oh < p.H) {
                            for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                                let owp = w + p.pad;
                                if (owp >= kw) {
                                    let ow = owp - kw;
                                    if (ow < p.W) {
                                        let di = ((((n * p.C + c) * p.T + ot) * p.H + oh) * p.W) + ow;
                                        let wti = ((c * p.K + kt) * p.K + kh) * p.K + kw;
                                        acc = acc + dy[di] * wt[wti];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    dx[idx] = acc;
}
