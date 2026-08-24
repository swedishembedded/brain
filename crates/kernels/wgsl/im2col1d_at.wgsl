// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  1D im2col over a RANGE of output positions - `im2col_at`'s windowing for NCL convolutions, so a 1D conv can be lowered to a GEMM in length chunks
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// 1D im2col over a RANGE of output positions - `im2col_at`'s windowing for NCL
// convolutions, so a 1D conv can be lowered to a GEMM in length chunks.
//
//   x   : [Cin, L]        one NCL batch row (the caller slices N)
//   col : [cnt, Cin*K]    row-major - col[i, ci*K + kw] is the input sample
//         feeding output position `pos0 + i` for tap (ci, kw), 0 outside the
//         padding. Tap coordinate: `li = (pos0+i)*stride + kw*dilation - pad`,
//         which is `conv1d.wgsl`'s own mapping (low pad explicit, high side
//         implicit) so the lowering is the same arithmetic, not an equivalent
//         one.
//
// With that operand, `y[Lo, Cout] = col[Lo, Cin*K] · Wᵀ` is `matmul_reg3` over
// the conv's NATIVE `[Cout, Cin/G, K]` weight - which is `[Cout, Cin*K]`
// row-major at `G = 1`, so no permute of the checkpoint tensor is needed. The
// epilogue is the shared `nlc_bias_nchw` (transpose + bias), the same one the
// 2D lowering uses.
//
// Why the window exists, and why the GEMM is in THIS orientation: the
// un-windowed operand for one vocoder stage (Cin = 96, K = 7, L = 352768) is
// `[352768, 672]` f32 = **948 MB** per batch row - 1.9 GB for a stereo pair,
// over a P40's 2047 MiB `max_storage_buffer_binding_size`, so the whole-signal
// im2col is not even bindable. With positions as the GEMM's ROWS a length
// chunk is a contiguous row range of both `col` and the output, so both
// bindings stay plain sub-ranges and one bounded scratch serves every chunk.
// Exactly `im2col_at.wgsl`'s argument, one spatial axis down.
//
// `K = 1` does NOT come here: at `stride = 1, pad = 0` the col operand would be
// x transposed, and the transpose-free lowering is instead
// `matmul_dx_reg`'s NN form straight over the native NCL input (see
// `audio::conv`).
//
// Threads are indexed by the col element, so the WRITE is fully coalesced and
// the read gathers x in `K`-float runs - the same trade `im2col_at` measured
// (its workgroup-staged tile was 273 -> 311 ms, i.e. slower, because the
// uncoalesced side is only ~K-fold amplified and the shared memory costs more
// occupancy than it buys). Kept element-indexed for the same reason.
//
// One invocation per (i, cink) element of the window.

struct Params {
    cin: u32,
    l: u32,
    k: u32,
    stride: u32,
    pad: u32,
    dilation: u32,
    cink: u32,       // Cin*K (col row stride)
    pos0: u32,       // first output position of this window
    cnt: u32,        // number of positions in this window
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.cnt * p.cink;
    if (idx >= total) { return; }

    let cink = idx % p.cink;
    let pos = p.pos0 + idx / p.cink;   // absolute output position
    let ci = cink / p.k;
    let kw = cink % p.k;

    // signed source coordinate (i32 to catch the negative-padding region)
    let li = i32(pos * p.stride + kw * p.dilation) - i32(p.pad);
    var v = 0.0;
    if (li >= 0 && li < i32(p.l)) {
        v = x[ci * p.l + u32(li)];
    }
    col[idx] = v;
}
