// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  1D convolution forward (bias-free), NCL layout, with grouping + dilation
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// 1D convolution forward (bias-free), NCL layout, with grouping + dilation.
// The workhorse for the TTS audio stack (codec conv encoder/decoder, ECAPA
// speaker encoder, GAN vocoder). Causal convs are expressed by the caller as a
// LEFT pad of `dilation*(K-1)` with `Lo == L` (stride 1); `pad` is the low-side
// pad and the high side is implicit (taps past the end are skipped = zero pad).
//   x : [N, Cin,        L ]   idx = (n*Cin + ci)*L + li
//   w : [Cout, Cin/G,   K ]   idx = (co*(Cin/G) + ci_local)*K + kw
//   y : [N, Cout,       Lo]   idx = (n*Cout + co)*Lo + lo
// One invocation per OUTPUT element.  li = lo*stride + kw*dilation - pad.

struct Params {
    N: u32,
    Cin: u32,
    L: u32,
    Cout: u32,
    K: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    groups: u32,
    Lo: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cout * p.Lo;
    if (idx >= total) { return; }

    // Decode output coordinate (n, co, lo).
    let lo = idx % p.Lo;
    let t1 = idx / p.Lo;
    let co = t1 % p.Cout;
    let n  = t1 / p.Cout;

    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let g = co / cout_g;              // group this output channel belongs to
    let ci0 = g * cin_g;             // first input channel of the group

    var acc = 0.0;
    for (var cl: u32 = 0u; cl < cin_g; cl = cl + 1u) {
        let ci = ci0 + cl;
        for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
            let li_b = lo * p.stride + kw * p.dilation;
            if (li_b >= p.pad) {
                let li = li_b - p.pad;
                if (li < p.L) {
                    let x_idx = (n * p.Cin + ci) * p.L + li;
                    let w_idx = (co * cin_g + cl) * p.K + kw;
                    acc = acc + x[x_idx] * w[w_idx];
                }
            }
        }
    }
    y[idx] = acc;
}
