// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D im2col over a RANGE of output positions - `im2col_at.wgsl` lifted to the time axis, so a conv3d can be lowered to a GEMM in chunks
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// 3D im2col over a RANGE of output positions - `im2col_at.wgsl` with the time
// axis added, so a `conv3d` can be lowered to `matmul_reg3` in spatial chunks:
//
//   x   : [N=1, Cin, T, H, W]        (NCTHW)
//   col : [cnt, Cin*KT*KH*KW]        row-major -
//         col[i, ((ci*KT + kt)*KH + kh)*KW + kw] = the input voxel feeding
//         output position `pos0 + i` for tap (ci,kt,kh,kw), 0 outside padding.
//
// That tap order is NOT a free choice: it is exactly `conv3d.wgsl`'s weight
// index `(((co*Cin/G + cl)*KT + kt)*KH + kh)*KW + kw`, so the SAME weight
// tensor, viewed as `[Cout, Cin*KT*KH*KW]`, is the GEMM's B operand with no
// repacking. `y[To*Ho*Wo, Cout] = col . Wt`, then `nlc_bias_nchw` adds the bias
// and transposes to the `[Cout, To, Ho, Wo]` that `conv3d` would have written.
//
// Why direct `conv3d` is not enough: it is one thread per output with four
// nested serial reductions and no operand reuse, which measured a FLAT low
// single-digit percent of a P40's fp32 peak across every shape of the Wan-VAE
// decode, while `matmul_reg3` runs the same arithmetic orders of magnitude
// closer to peak. A rate that
// does not move with shape is structural, so the fix is the lowering, not
// tuning.
//
// THE TIME PAD IS ONE-SIDED, exactly as in `conv3d.wgsl`: `pt` is the
// already-DOUBLED low (past) pad, so output frame `to` reads at most input
// frame `to*st + KT-1 - pt`, and with the causal convention that is `to`. A
// symmetric pad here would let a frame read frames it is supposed to predict.
// Space is ordinary symmetric padding via `ph`/`pw`.
//
// Why the window exists: the un-windowed operand for the Wan-VAE's 96-channel
// 240x416 (3,3,3) conv is `[399360, 2592]` f32 = 4.1 GB, twice over the P40's
// 2047 MiB `max_storage_buffer_binding_size`. Chunking the GEMM's ROWS keeps
// one bounded scratch and leaves the arithmetic identical: a position chunk is
// a contiguous row range of both `col` and the output.
//
// One invocation per (i, cinkkk) element of the window.

struct Params {
    cin: u32, t: u32, h: u32, w: u32,
    kt: u32, kh: u32, kw: u32,
    st: u32, sh: u32, sw: u32,
    pt: u32, ph: u32, pw: u32,
    to: u32, ho: u32, wo: u32,
    cinkkk: u32,     // Cin*KT*KH*KW (col row stride)
    pos0: u32,       // first output spatial index of this window
    cnt: u32,        // number of positions in this window
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.cnt * p.cinkkk;
    if (idx >= total) { return; }

    let tap = idx % p.cinkkk;
    let pos = p.pos0 + idx / p.cinkkk;   // absolute output spatial index

    // Decode (to, ho, wo) - `wo` fastest, matching conv3d's y layout for a
    // fixed output channel.
    let owo = pos % p.wo;
    let oho = (pos / p.wo) % p.ho;
    let oto = pos / (p.wo * p.ho);

    // Decode the tap (ci, kt, kh, kw) from conv3d's weight ordering.
    let khw = p.kh * p.kw;
    let ci = tap / (p.kt * khw);
    let r  = tap % (p.kt * khw);
    let kt = r / khw;
    let r2 = r % khw;
    let kh = r2 / p.kw;
    let kw = r2 % p.kw;

    var v = 0.0;
    // Time: one-sided low pad, unsigned compare exactly as conv3d does it.
    let it = oto * p.st + kt;
    if (it >= p.pt && it - p.pt < p.t) {
        let ti = it - p.pt;
        // Space: symmetric pad, signed coords to catch the negative region.
        let ih = i32(oho * p.sh + kh) - i32(p.ph);
        let iw = i32(owo * p.sw + kw) - i32(p.pw);
        if (ih >= 0 && iw >= 0 && ih < i32(p.h) && iw < i32(p.w)) {
            v = x[(((ci * p.t + ti) * p.h + u32(ih)) * p.w + u32(iw))];
        }
    }
    col[idx] = v;
}
