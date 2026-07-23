// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// im2col: lower a conv input into a GEMM operand.
//   x   : [N=1, Cin, H, W]  (NCHW)
//   col : [Ho*Wo, Cin*K*K]  row-major — col[hw, (ci*K + kh)*K + kw] = the input
//         pixel feeding output position hw for tap (ci,kh,kw), 0 outside padding.
//
// With this, a conv is a plain matmul:  y[Cout, Ho*Wo] = W[Cout, Cin*K*K] ·
// colᵀ, i.e. `matmul_reg2(x=W, w=col)` → y[Cout, HW]. On a compute-bound GPU
// (P40: 34 FLOP/byte) the register GEMM's ~34% of peak dwarfs the collapse the
// direct register-tiled conv (`conv_act_reg`) suffers on deep small-spatial
// layers — the trade `docs/PERFORMANCE.md` flagged as "worth it on a
// compute-bound discrete GPU". im2col's extra [HW, Cin*K*K] write+read is the
// cost; the arithmetic-intensity win pays for it here.
//
// One invocation per (hw, cinkk) element of `col`.

struct Params {
    cin: u32, h: u32, w: u32, k: u32,
    stride: u32, pad: u32, ho: u32, wo: u32,
    cinkk: u32,      // Cin*K*K (col row stride)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let hw = p.ho * p.wo;
    let total = hw * p.cinkk;
    if (idx >= total) { return; }

    let cinkk = idx % p.cinkk;
    let pos = idx / p.cinkk;          // output spatial index hw
    let ho = pos / p.wo;
    let wo = pos % p.wo;
    let kk = p.k * p.k;
    let ci = cinkk / kk;
    let r = cinkk % kk;
    let kh = r / p.k;
    let kw = r % p.k;

    // signed source coords (i32 to catch the negative-padding region)
    let ih = i32(ho * p.stride + kh) - i32(p.pad);
    let iw = i32(wo * p.stride + kw) - i32(p.pad);
    var v = 0.0;
    if (ih >= 0 && iw >= 0 && ih < i32(p.h) && iw < i32(p.w)) {
        v = x[(ci * p.h + u32(ih)) * p.w + u32(iw)];
    }
    col[idx] = v;
}
