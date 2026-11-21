// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped causal depthwise Conv1d whose taps are PER TOKEN (data-dependent) on top of a static per-channel base, or purely static, with an optional fused residual and scale, document-boundary segments and a carried history
// @how   one thread per (t,c) output element, serial reduction over K taps
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// A causal depthwise conv over a short block whose weight is not a weight:
//
//   y[t,c] = sum_{j=0}^{K-1} (base[j,c] + dyn[t,j,group(c)]) * x[t-j,c]
//
// with `x[t-j,c] = 0` for `t < j` (the same implicit left zero-pad
// `conv1d.wgsl` applies via `pad = K-1`), `group(c) = c / group_size`, and
// `dyn` PRODUCED BY A PROJECTION OF THE ACTIVATIONS - a different set of taps
// for every token of every call.
//
// This is what separates it from `causal_conv1d_step.wgsl`, which is
// otherwise the same shape: there, `w` is indexed by channel alone, and a
// data-dependent weight would have to be materialised as `N` broadcast copies
// of it. Here the per-token part is already compact - one value per (token,
// tap, GROUP) rather than per channel, `group_size` channels sharing it - so
// it is read directly and added to the static base, and the whole convolution
// stays one dispatch with no broadcast pass.
//
// It also takes the whole block at once rather than one token per call. The
// consumer (a block-diffusion drafter, `dflash2`) denoises all its rows in a
// single non-autoregressive forward, so the conv's history is entirely inside
// the block and there is no ring-buffer state to carry: the first row of a
// block convolves against zeros, by construction and not by initialisation.
// Several blocks (one per sequence) ride one dispatch: `block` is the rows of
// one, and the history restarts at every multiple of it.
//
// Buffers:
//   x    : `[n, c]`  the block's activations, token-major
//   dyn   : `[n, dyn_stride]` the projection's output; this conv reads the
//           `taps*groups` window starting at `dyn_off` of each row, laid out
//           `[tap][group]`. One projection feeds TWO convolutions (one before
//           the sublayer, one after) out of one row, which is what `dyn_off`
//           selects between - the alternative, slicing the projection into two
//           buffers, would be a copy of the same size as the thing copied.
//           Unread with FLAG_NO_DYN.
//   base : `[*, taps, c]` the static per-channel kernel, at `base_off`; same
//           two-halves-in-one-tensor reason for the offset. With FLAG_BASE_CK it
//           is `[c, taps]` with the LAST tap on the current token (the layout of
//           a depthwise `Conv1d` weight `[c, 1, taps]`, as `causal_conv1d_step`
//           reads it), so a published convolution weight is read as it is.
//   y    : `[n, c]`
//   seg  : `[n]` u32 document id of each row; read only with FLAG_SEG
//   hist : `[c, taps-1]` the previous `taps-1` inputs of each channel, oldest
//          first (`causal_conv1d_step`'s `hist`); read only with FLAG_HIST
//
// `n` is a whole number of blocks of `block` rows; pass `block = n` for one.
//
// `group_size` must divide `c`, and `groups = c / group_size` must be the same
// `groups` the projection was laid out with - the host is what knows both, so
// they are passed rather than derived twice.
//
// ## Optional behaviours (all off when `flags == 0`: the kernel then computes exactly what it always did)
//
//   FLAG_RES_SCALE (1)  y = (conv(x) + x) * res_scale: the residual and its scale fused into the output, so a
//                       convolution that feeds a normalised sum needs no extra passes.
//   FLAG_SEG (2)        a tap reads row `t-j` only when `seg[t-j] == seg[t]`: rows packed from different documents
//                       never convolve with each other. (`block` restarts history at fixed boundaries; `seg` at data
//                       dependent ones; both may be on.)
//   FLAG_HIST (4)       taps that reach before the first row read `hist` instead of zero (a chunk that continues a
//                       sequence). Requires `block == n`: the history belongs to the one sequence the call holds. With
//                       FLAG_SEG the history counts as belonging to the document of row 0.
//   FLAG_NO_DYN (8)     a purely static convolution: the per-token taps (and the `dyn` binding) are not read.
//   FLAG_BASE_CK (16)   `base` is `[c, taps]`, last tap current (see above).
// The backward is dyn_group_conv1d_dx / _dbase / _ddyn (no history input: a history is an inference carry-over); a
// single token with a ring state is dyn_group_conv1d_step.

const FLAG_RES_SCALE: u32 = 1u;
const FLAG_SEG: u32 = 2u;
const FLAG_HIST: u32 = 4u;
const FLAG_NO_DYN: u32 = 8u;
const FLAG_BASE_CK: u32 = 16u;

struct Params {
    n: u32,
    c: u32,
    taps: u32,
    group_size: u32,
    groups: u32,
    dyn_stride: u32,
    dyn_off: u32,
    base_off: u32,
    block: u32,
    flags: u32,
    res_scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       dyn:  array<f32>;
@group(0) @binding(3) var<storage, read>       base: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:    array<f32>;
@group(0) @binding(5) var<storage, read>       seg:  array<u32>;
@group(0) @binding(6) var<storage, read>       hist: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n * p.c) { return; }
    let t = idx / p.c;
    let ci = idx % p.c;
    let in_block = t % p.block;
    let g = ci / p.group_size;
    let drow = t * p.dyn_stride + p.dyn_off;
    let use_seg = (p.flags & FLAG_SEG) != 0u;
    let use_hist = (p.flags & FLAG_HIST) != 0u;

    var acc = 0.0;
    for (var j: u32 = 0u; j < p.taps; j = j + 1u) {
        var xv = 0.0;
        if (j <= in_block) {
            let s = t - j;
            if (!use_seg || seg[s] == seg[t]) { xv = x[s * p.c + ci]; }
        } else if (use_hist) {
            // `block == n`, so `t == in_block` and the tap reaches `j - t` rows before the first row.
            if (!use_seg || seg[0] == seg[t]) { xv = hist[ci * (p.taps - 1u) + (p.taps - 1u - j + t)]; }
        } else {
            break;
        }
        var w = 0.0;
        if ((p.flags & FLAG_BASE_CK) != 0u) { w = base[p.base_off + ci * p.taps + (p.taps - 1u - j)]; }
        else { w = base[p.base_off + j * p.c + ci]; }
        if ((p.flags & FLAG_NO_DYN) == 0u) { w = w + dyn[drow + j * p.groups + g]; }
        acc = acc + w * xv;
    }
    if ((p.flags & FLAG_RES_SCALE) != 0u) { acc = (acc + x[idx]) * p.res_scale; }
    y[idx] = acc;
}
