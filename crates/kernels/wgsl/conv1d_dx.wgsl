// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  1D convolution input gradient (gather form, no scatter / no atomics)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// 1D convolution input gradient (gather form, no scatter / no atomics).
//   dy : [N, Cout,     Lo]
//   w  : [Cout, Cin/G, K ]
//   dx : [N, Cin,      L ]   read_write, one invocation per INPUT element.
// Forward: li = lo*stride + kw*dilation - pad. Invert for fixed li:
//   lo = (li + pad - kw*dilation) / stride, valid when the numerator is
//   non-negative, divisible by stride, and lo in [0,Lo). Each (co,kw) hit adds
//   dy[n,co,lo] * w[co,ci_local,kw].

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
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Cin * p.L;
    if (idx >= total) { return; }

    // Decode input coordinate (n, ci, li).
    let li = idx % p.L;
    let t1 = idx / p.L;
    let ci = t1 % p.Cin;
    let n  = t1 / p.Cin;

    let cin_g  = p.Cin / p.groups;
    let cout_g = p.Cout / p.groups;
    let g  = ci / cin_g;             // group of this input channel
    let cl = ci - g * cin_g;         // local input-channel index within group
    let co0 = g * cout_g;

    var acc = 0.0;
    for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
        let num = li + p.pad;
        let sub = kw * p.dilation;
        if (num >= sub) {
            let num2 = num - sub;
            if ((num2 % p.stride) == 0u) {
                let lo = num2 / p.stride;
                if (lo < p.Lo) {
                    for (var c: u32 = 0u; c < cout_g; c = c + 1u) {
                        let co = co0 + c;
                        let dy_idx = (n * p.Cout + co) * p.Lo + lo;
                        let w_idx = (co * cin_g + cl) * p.K + kw;
                        acc = acc + dy[dy_idx] * w[w_idx];
                    }
                }
            }
        }
    }
    dx[idx] = acc;
}
