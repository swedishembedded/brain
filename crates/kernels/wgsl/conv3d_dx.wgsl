// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D convolution INPUT gradient (adjoint of conv3d, GATHER form, no scatter)
// @how   one thread per output element, 4 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// 3D convolution input gradient - conv2d_dx lifted to NCTHW, sharing conv3d's
// Params word-for-word.
//   dy : [N, Cout,     To, Ho, Wo]
//   wt : [Cout, Cin/G, KT, KH, KW]
//   dx : [N, Cin,      T,  H,  W]   read_write (one invocation per INPUT element)
//
// The forward reads input ti into output to via  to*st - pt + kt == ti, so
//   to = (ti + pt - kt)/st
// contributes only when (ti + pt - kt) is non-negative, divisible by `st`, and
// the quotient lies in [0,To). Space inverts identically with ph/sh and pw/sw.
// Gathering per input element means each dx[idx] has exactly one writer, so no
// atomics and no scatter are needed.
//
// The causal asymmetry survives transposition unchanged: `pt` appears only as
// the low-side offset, so an input frame can only push gradient BACKWARD into
// output frames at or after it. A dx that used a symmetric time pad would train
// the forward to leak the future even while the forward kernel is correct.
// Only output channels of ci's own group are summed.

struct Params {
    N: u32, Cin: u32, T: u32, H: u32, W: u32,
    Cout: u32, KT: u32, KH: u32, KW: u32,
    st: u32, sh: u32, sw: u32,
    pt: u32, ph: u32, pw: u32, groups: u32,
    To: u32, Ho: u32, Wo: u32,
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
    if (idx >= p.N * p.Cin * thw) { return; }

    // Decode input coordinate (n, ci, ti, hi, wi) from the linear index.
    let wi = idx % p.W;
    let hi = (idx / p.W) % p.H;
    let ti = (idx / (p.W * p.H)) % p.T;
    let ci = (idx / thw) % p.Cin;
    let n  = idx / (p.Cin * thw);

    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let g  = ci / cin_g;
    let cl = ci - g * cin_g;   // ci's slot inside wt's Cin/G axis
    let co0 = g * cout_g;

    var acc = 0.0;
    for (var l: u32 = 0u; l < cout_g; l = l + 1u) {
        let co = co0 + l;
        for (var kt: u32 = 0u; kt < p.KT; kt = kt + 1u) {
            let tp = ti + p.pt;
            if (tp >= kt) {
                let num_t = tp - kt;
                if ((num_t % p.st) == 0u) {
                    let to = num_t / p.st;
                    if (to < p.To) {
                        for (var kh: u32 = 0u; kh < p.KH; kh = kh + 1u) {
                            let hp = hi + p.ph;
                            if (hp >= kh) {
                                let num_h = hp - kh;
                                if ((num_h % p.sh) == 0u) {
                                    let ho = num_h / p.sh;
                                    if (ho < p.Ho) {
                                        for (var kw: u32 = 0u; kw < p.KW; kw = kw + 1u) {
                                            let wp = wi + p.pw;
                                            if (wp >= kw) {
                                                let num_w = wp - kw;
                                                if ((num_w % p.sw) == 0u) {
                                                    let wo = num_w / p.sw;
                                                    if (wo < p.Wo) {
                                                        let di = (((n * p.Cout + co) * p.To + to) * p.Ho + ho) * p.Wo + wo;
                                                        let wti = (((co * cin_g + cl) * p.KT + kt) * p.KH + kh) * p.KW + kw;
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
                }
            }
        }
    }
    dx[idx] = acc;
}
