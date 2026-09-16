// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Grouped causal depthwise Conv1d whose taps are PER TOKEN (data-dependent), on top of a static per-channel base
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
// consumer (a block-diffusion drafter, `qwen35::dflash2`) denoises all its
// rows in a single non-autoregressive forward, so the conv's history is
// entirely inside the block and there is no ring-buffer state to carry: row 0
// convolves against zeros, by construction and not by initialisation.
//
// Buffers:
//   x    : `[n, c]`  the block's activations, token-major
//   dyn   : `[n, dyn_stride]` the projection's output; this conv reads the
//           `taps*groups` window starting at `dyn_off` of each row, laid out
//           `[tap][group]`. One projection feeds TWO convolutions (one before
//           the sublayer, one after) out of one row, which is what `dyn_off`
//           selects between - the alternative, slicing the projection into two
//           buffers, would be a copy of the same size as the thing copied.
//   base : `[*, taps, c]` the static per-channel kernel, at `base_off`; same
//           two-halves-in-one-tensor reason for the offset.
//   y    : `[n, c]`
//
// `group_size` must divide `c`, and `groups = c / group_size` must be the same
// `groups` the projection was laid out with - the host is what knows both, so
// they are passed rather than derived twice.

struct Params {
    n: u32,
    c: u32,
    taps: u32,
    group_size: u32,
    groups: u32,
    dyn_stride: u32,
    dyn_off: u32,
    base_off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       dyn:  array<f32>;
@group(0) @binding(3) var<storage, read>       base: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n * p.c) { return; }
    let t = idx / p.c;
    let ci = idx % p.c;
    let g = ci / p.group_size;
    let drow = t * p.dyn_stride + p.dyn_off;

    var acc = 0.0;
    for (var j: u32 = 0u; j < p.taps; j = j + 1u) {
        if (j > t) { break; }
        let w = base[p.base_off + j * p.c + ci] + dyn[drow + j * p.groups + g];
        acc = acc + w * x[(t - j) * p.c + ci];
    }
    y[idx] = acc;
}
