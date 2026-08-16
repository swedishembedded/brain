// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D convolution forward (with bias), NCTHW, per-axis kernel/stride, CAUSAL temporal pad
// @how   one thread per output element, 4 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// 3D convolution forward, NCTHW - conv2d_gd's index math lifted once more, to
// the time axis. Every extent is per-axis, so a temporal-only (3,1,1) conv and a
// full (3,3,3) conv are the same kernel at different Params.
//   x    : [N, Cin,      T,  H,  W]   idx = (((n*Cin  + ci)*T  + ti)*H  + hi)*W  + wi
//   wt   : [Cout, Cin/G, KT, KH, KW]  idx = (((co*Cin/G + cl)*KT + kt)*KH + kh)*KW + kw
//   bias : [Cout]
//   y    : [N, Cout,     To, Ho, Wo]  idx = (((n*Cout + co)*To + to)*Ho + ho)*Wo + wo
//
// THE TIME PAD IS ONE-SIDED AND THAT IS THE WHOLE POINT. `pt` pads only the low
// (past) side; the high side gets nothing, so output frame `to` reads at most
// input frame `to*st + KT-1 - pt`. With the causal-conv convention pt = 2*pad_t,
// KT = 3, st = 1 that upper bound is exactly `to`: no output frame can see a
// future input frame. A symmetric pad here would still produce plausible video,
// just video that has read frames it is supposed to predict, so `pt` is the
// already-DOUBLED low pad (same meaning as dwconv3d's `pt`), never `pad_t`.
// Space is ordinary symmetric padding via `ph`/`pw`. Taps whose input
// coordinate falls outside the volume are skipped, which is exactly the
// contribution of a zero-padded border.
//   To = (T +   pt  - KT)/st + 1
//   Ho = (H + 2*ph  - KH)/sh + 1        (likewise Wo)
// The host computes all three and passes them in.
//
// Grouping costs two divisions here, the same way it does in conv2d_gd, so it
// is kept: G == Cin == Cout is depthwise, and dwconv3d remains the narrower
// (and cheaper, weights [C,K,K,K]) kernel for that case.
//
// Only the natural tensors are bound - unlike a lowered conv there is no im2col
// operand that can exceed `max_storage_buffer_binding_size` on its own. When x
// or y is too large to bind, N is the split that stays correct: it is the
// outermost axis of both, so a batch range is contiguous in each and the kernel
// is separable over it (wt/bias are shared unsliced).

struct Params {
    N: u32, Cin: u32, T: u32, H: u32, W: u32,
    Cout: u32, KT: u32, KH: u32, KW: u32,
    st: u32, sh: u32, sw: u32,
    pt: u32, ph: u32, pw: u32, groups: u32,
    To: u32, Ho: u32, Wo: u32,
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
    let thw = p.To * p.Ho * p.Wo;
    if (idx >= p.N * p.Cout * thw) { return; }

    // Decode output coordinate (n, co, to, ho, wo) from the linear index.
    let wo = idx % p.Wo;
    let ho = (idx / p.Wo) % p.Ho;
    let to = (idx / (p.Wo * p.Ho)) % p.To;
    let co = (idx / thw) % p.Cout;
    let n  = idx / (p.Cout * thw);

    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let ci0 = (co / cout_g) * cin_g;   // first input channel of co's group

    var acc = bias[co];
    for (var cl: u32 = 0u; cl < cin_g; cl = cl + 1u) {
        let ci = ci0 + cl;
        for (var kt: u32 = 0u; kt < p.KT; kt = kt + 1u) {
            let it = to * p.st + kt;
            if (it >= p.pt && it - p.pt < p.T) {
                let ti = it - p.pt;
                for (var kh: u32 = 0u; kh < p.KH; kh = kh + 1u) {
                    let ih = ho * p.sh + kh;
                    if (ih >= p.ph && ih - p.ph < p.H) {
                        let hi = ih - p.ph;
                        for (var kw: u32 = 0u; kw < p.KW; kw = kw + 1u) {
                            let iw = wo * p.sw + kw;
                            if (iw >= p.pw && iw - p.pw < p.W) {
                                let wi = iw - p.pw;
                                let xi = (((n * p.Cin + ci) * p.T + ti) * p.H + hi) * p.W + wi;
                                let wti = (((co * cin_g + cl) * p.KT + kt) * p.KH + kh) * p.KW + kw;
                                acc = acc + x[xi] * wt[wti];
                            }
                        }
                    }
                }
            }
        }
    }
    y[idx] = acc;
}
