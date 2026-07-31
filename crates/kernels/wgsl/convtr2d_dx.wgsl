// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Transposed 2D convolution INPUT gradient (gather form, no scatter/atomics).
// The adjoint of convtr2d.wgsl.
//   dy : [N, Cout,      Ho, Wo]  idx = ((n*Cout + co)*Ho + ho)*Wo + wo
//   w  : [Cin, Cout/G,  K,  K]   idx = ((ci*(Cout/G) + c)*K + kh)*K + kw
//   dx : [N, Cin,       H,  W]   read_write, one invocation per INPUT element.
//
// convtr2d's forward is already a gather over an INVERTED map, so its `_dx` is
// the DIRECT map and is the simpler of the two: the input (hi,wi) contributed to
//   ho = hi*stride + kh*dilation - pad,  wo = wi*stride + kw*dilation - pad
// for every tap (kh,kw) whose (ho,wo) lands inside [0,Ho)x[0,Wo). There is no
// divisibility test here — the multiplication by `stride` is on the input side.
//
// Gathering per input element means every dx[idx] is written by exactly one
// invocation, so no atomics / scatter are needed. Terminal write is a plain
// overwrite.
//
// GROUPING: the invocation owns an INPUT channel, so the group comes from ci
// (g = ci/(Cin/G)) and the co loop is confined to [g*Cout/G, (g+1)*Cout/G).
// The weight's second axis is that loop's LOCAL index c, not the absolute co.

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
    let g   = ci / cin_g;   // group this INPUT channel belongs to
    let co0 = g * cout_g;   // first output channel of that group

    var acc = 0.0;
    for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
        let ho_b = hi * p.stride + kh * p.dilation;
        if (ho_b >= p.pad) {
            let ho = ho_b - p.pad;
            if (ho < p.Ho) {
                for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                    let wo_b = wi * p.stride + kw * p.dilation;
                    if (wo_b >= p.pad) {
                        let wo = wo_b - p.pad;
                        if (wo < p.Wo) {
                            for (var c: u32 = 0u; c < cout_g; c = c + 1u) {
                                let co = co0 + c;
                                let dy_idx = ((n * p.Cout + co) * p.Ho + ho) * p.Wo + wo;
                                let w_idx = ((ci * cout_g + c) * p.K + kh) * p.K + kw;
                                acc = acc + dy[dy_idx] * w[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    dx[idx] = acc;
}
