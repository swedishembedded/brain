// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped+dilated 2D convolution INPUT gradient (transposed-conv GATHER form)
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Grouped+dilated 2D convolution INPUT gradient (transposed-conv GATHER form).
//   dy : [N, Cout,      Ho, Wo]
//   w  : [Cout, Cin/G,  K,  K]
//   dx : [N, Cin,       H,  W]   read_write (one invocation per INPUT element)
//
// The backward partner of conv2d_gd.wgsl; conv2d_dx.wgsl generalized.
//
// The forward reads input (hi,wi) into output (ho,wo) with the relation
//   ho*stride - pad + kh*dilation == hi  =>  ho = (hi + pad - kh*dilation)/stride
// valid only when (hi + pad - kh*dilation) is non-negative, divisible by stride,
// and the resulting ho lies in [0,Ho). Each (co,kh,kw) that satisfies this
// contributes dx += dy[n,co,ho,wo] * w[co,cl,kh,kw].
//
// GROUPING: input channel ci only feeds the output channels of ITS OWN group, so
// the co loop runs over [g*cout_g, (g+1)*cout_g) rather than all of Cout — g is
// derived from ci, and cl = ci - g*cin_g is ci's index within the group, which is
// what indexes w's second axis.
//
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
    dilation: u32,
    groups: u32,
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

    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let g   = ci / cin_g;        // group this INPUT channel belongs to
    let cl  = ci - g * cin_g;    // its index within the group -> w's 2nd axis
    let co0 = g * cout_g;        // only this group's output channels see it

    var acc = 0.0;
    for (var cg: u32 = 0u; cg < cout_g; cg = cg + 1u) {
        let co = co0 + cg;
        for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
            // num_h = hi + pad - kh*dilation ; need >=0 and divisible by stride.
            let hi_pad = hi + p.pad;
            let dh = kh * p.dilation;
            if (hi_pad >= dh) {
                let num_h = hi_pad - dh;
                if ((num_h % p.stride) == 0u) {
                    let ho = num_h / p.stride;
                    if (ho < p.Ho) {
                        for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                            let wi_pad = wi + p.pad;
                            let dw_ = kw * p.dilation;
                            if (wi_pad >= dw_) {
                                let num_w = wi_pad - dw_;
                                if ((num_w % p.stride) == 0u) {
                                    let wo = num_w / p.stride;
                                    if (wo < p.Wo) {
                                        let dy_idx = ((n * p.Cout + co) * p.Ho + ho) * p.Wo + wo;
                                        let w_idx = ((co * cin_g + cl) * p.K + kh) * p.K + kw;
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
