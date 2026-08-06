// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Transposed 1D convolution weight gradient
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Transposed 1D convolution weight gradient. ACCUMULATES into a pre-zeroed
// buffer.
//   dy : [N, Cout, Lo]
//   x  : [N, Cin,  L ]
//   dw : [Cin, Cout/G, K]   read_write, one invocation per WEIGHT element.
// dw[ci,co_local,kw] = sum over n,li of dy[n,co,lo] * x[n,ci,li] with the
// forward relation lo = li*stride - pad + kw*dilation (bounds-checked) and
// co = g*(Cout/G) + co_local where g = ci/(Cin/G).

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
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let cout_g = p.Cout / p.groups;
    let total = p.Cin * cout_g * p.K;
    if (idx >= total) { return; }

    // Decode weight coordinate (ci, co_local, kw).
    let kw = idx % p.K;
    let t1 = idx / p.K;
    let co_local = t1 % cout_g;
    let ci = t1 / cout_g;

    let cin_g = p.Cin / p.groups;
    let g = ci / cin_g;
    let co = g * cout_g + co_local;

    var acc = 0.0;
    for (var n: u32 = 0u; n < p.N; n = n + 1u) {
        for (var li: u32 = 0u; li < p.L; li = li + 1u) {
            let lo_b = li * p.stride + kw * p.dilation;
            if (lo_b >= p.pad) {
                let lo = lo_b - p.pad;
                if (lo < p.Lo) {
                    let dy_idx = (n * p.Cout + co) * p.Lo + lo;
                    let x_idx = (n * p.Cin + ci) * p.L + li;
                    acc = acc + dy[dy_idx] * x[x_idx];
                }
            }
        }
    }
    dw[idx] = dw[idx] + acc;
}
