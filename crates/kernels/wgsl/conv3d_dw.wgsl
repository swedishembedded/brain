// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D convolution WEIGHT gradient (accumulating), NCTHW
// @how   one thread per output element, 4 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// 3D convolution weight gradient. ACCUMULATES into a pre-zeroed buffer, like
// conv2d_dw and dwconv3d_dw, so a pass composes with a prior dw.
//   dy : [N, Cout,     To, Ho, Wo]
//   x  : [N, Cin,      T,  H,  W]
//   dw : [Cout, Cin/G, KT, KH, KW]   read_write (one invocation per WEIGHT element)
//
// dw[co,cl,kt,kh,kw] = sum over n, to, ho, wo of dy[n,co,to,ho,wo] * x[n,ci,ti,hi,wi]
// with the forward relation ti = to*st - pt + kt (bounds-checked; a tap outside
// the input map is skipped, which is the zero pad contributing nothing). `ci` is
// recovered from co's group: ci = (co/cout_g)*cin_g + cl.
//
// Params is conv3d's, unchanged, including the one-sided temporal `pt` - a tap
// at kt is never allowed to reach forward past the low pad, so an output frame
// that the forward could not see cannot contribute gradient here either.
//
// BIAS grad is not this kernel's job: it is the sum of dy over (n,to,ho,wo) per
// output channel, which the existing bias-grad path covers once dy is viewed as
// [N, Cout, To*Ho*Wo].

struct Params {
    N: u32, Cin: u32, T: u32, H: u32, W: u32,
    Cout: u32, KT: u32, KH: u32, KW: u32,
    st: u32, sh: u32, sw: u32,
    pt: u32, ph: u32, pw: u32, groups: u32,
    To: u32, Ho: u32, Wo: u32,
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
    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let ktk = p.KT * p.KH * p.KW;
    if (idx >= p.Cout * cin_g * ktk) { return; }

    // Decode weight coordinate (co, cl, kt, kh, kw) from the linear index.
    let kw = idx % p.KW;
    let kh = (idx / p.KW) % p.KH;
    let kt = (idx / (p.KW * p.KH)) % p.KT;
    let cl = (idx / ktk) % cin_g;
    let co = idx / (cin_g * ktk);
    let ci = (co / cout_g) * cin_g + cl;

    var acc = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        for (var to: u32 = 0u; to < p.To; to = to + 1u) {
            let it = to * p.st + kt;
            if (it >= p.pt && it - p.pt < p.T) {
                let ti = it - p.pt;
                for (var ho: u32 = 0u; ho < p.Ho; ho = ho + 1u) {
                    let ih = ho * p.sh + kh;
                    if (ih >= p.ph && ih - p.ph < p.H) {
                        let hi = ih - p.ph;
                        for (var wo: u32 = 0u; wo < p.Wo; wo = wo + 1u) {
                            let iw = wo * p.sw + kw;
                            if (iw >= p.pw && iw - p.pw < p.W) {
                                let wi = iw - p.pw;
                                let di = (((n * p.Cout + co) * p.To + to) * p.Ho + ho) * p.Wo + wo;
                                let xi = (((n * p.Cin + ci) * p.T + ti) * p.H + hi) * p.W + wi;
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
