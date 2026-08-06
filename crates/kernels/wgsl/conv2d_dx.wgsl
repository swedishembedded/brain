// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  2D convolution input gradient (transposed-conv GATHER form, no scatter)
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// 2D convolution input gradient (transposed-conv GATHER form, no scatter).
//   dy : [N, Cout, Ho, Wo]
//   w  : [Cout, Cin, K, K]
//   dx : [N, Cin,  H,  W]   read_write (one invocation per INPUT element)
//
// The forward reads input (hi,wi) into output (ho,wo) with the relation
//   ho*stride - pad + kh == hi  =>  ho = (hi + pad - kh)/stride
// valid only when (hi + pad - kh) is non-negative, divisible by stride, and the
// resulting ho lies in [0,Ho). Each (co,kh,kw) that satisfies this contributes
//   dx += dy[n,co,ho,wo] * w[co,ci,kh,kw].
// Gathering per input element means every dx[idx] is written by exactly one
// invocation, so no atomics / scatter are needed.

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
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cin * p.H * p.W;
    if (idx >= total) { return; }

    // Decode input coordinate (n, ci, hi, wi) from the linear index.
    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let ci = t2 % p.Cin;
    let n  = t2 / p.Cin;

    var acc = 0.0;
    for (var co: u32 = 0u; co < p.Cout; co = co + 1u) {
        for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
            // num_h = hi + pad - kh ; need >=0 and divisible by stride.
            let hi_pad = hi + p.pad;
            if (hi_pad >= kh) {
                let num_h = hi_pad - kh;
                if ((num_h % p.stride) == 0u) {
                    let ho = num_h / p.stride;
                    if (ho < p.Ho) {
                        for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                            let wi_pad = wi + p.pad;
                            if (wi_pad >= kw) {
                                let num_w = wi_pad - kw;
                                if ((num_w % p.stride) == 0u) {
                                    let wo = num_w / p.stride;
                                    if (wo < p.Wo) {
                                        let dy_idx = ((n * p.Cout + co) * p.Ho + ho) * p.Wo + wo;
                                        let w_idx = ((co * p.Cin + ci) * p.K + kh) * p.K + kw;
                                        acc = acc + dy[dy_idx] * w[w_idx];
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
