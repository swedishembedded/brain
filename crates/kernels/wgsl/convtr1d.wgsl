// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Transposed 1D convolution forward (ConvTranspose1d), NCL layout, grouping + dilation
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Transposed 1D convolution forward (ConvTranspose1d), NCL layout, grouping +
// dilation. Used for codec-decoder / vocoder upsampling. PyTorch weight layout
//   w : [Cin, Cout/G, K]   idx = (ci*(Cout/G) + co_local)*K + kw
//   x : [N, Cin,  L ]      idx = (n*Cin + ci)*L + li
//   y : [N, Cout, Lo]      idx = (n*Cout + co)*Lo + lo
// Forward maps input li into output lo = li*stride - pad + kw*dilation. One
// invocation per OUTPUT element gathers the inputs that land on it:
//   li = (lo + pad - kw*dilation)/stride  (integer, in [0,L)).
//   Lo = (L-1)*stride - 2*pad + dilation*(K-1) + out_pad + 1  (caller-computed).

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
    let g = co / cout_g;
    let co_local = co - g * cout_g;
    let ci0 = g * cin_g;

    var acc = 0.0;
    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
        let num = lo + p.pad;
        let sub = kw * p.dilation;
        if (num >= sub) {
            let num2 = num - sub;
            if ((num2 % p.stride) == 0u) {
                let li = num2 / p.stride;
                if (li < p.L) {
                    for (var cl: u32 = 0u; cl < cin_g; cl = cl + 1u) {
                        let ci = ci0 + cl;
                        let x_idx = (n * p.Cin + ci) * p.L + li;
                        let w_idx = (ci * cout_g + co_local) * p.K + kw;
                        acc = acc + x[x_idx] * w[w_idx];
                    }
                }
            }
        }
    }
    y[idx] = acc;
}
