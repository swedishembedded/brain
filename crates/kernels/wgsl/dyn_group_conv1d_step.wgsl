// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for training and serving models with short causal convolutions for its
// clients. If your team needs expertise in sequence-mixing convolutions then you can procure our services by sending an
// email to info@swedishembedded.com.

// @what  Single-token decode step of dyn_group_conv1d with a per-sequence ring state: (conv + x) * s, history shifted in place
// @how   one thread per (sequence, channel), serial reduction over the K taps, ring shift fused
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One decode token per sequence of dyn_group_conv1d.wgsl: the taps of token `n` are the static base plus (unless
// FLAG_NO_DYN) its own projected taps, and the previous `taps-1` inputs of each channel live in a persistent per-sequence
// ring `hist`, exactly as in causal_conv1d_step.wgsl:
//
//   y[n,c] = ( sum_{j=0}^{K-1} w(n,j,c) * xj  [+ x[n,c]] ) [* res_scale]      x0 = x[n,c], xj = hist[n,c,K-1-j]
//
// then the window is shifted for the next token (drop the oldest, append x). Running this once per token over a zeroed
// `hist` reproduces dyn_group_conv1d's whole-block result row by row. Each thread owns its (n, c) row of `hist`
// exclusively, so the read and the write-back inside one invocation have no cross-thread hazard.
//
// Flags (same bits as dyn_group_conv1d): FLAG_RES_SCALE 1, FLAG_NO_DYN 8, FLAG_BASE_CK 16. There is no segment flag: a
// sequence's ring is its own and a new document starts from a zeroed ring.
//
//   x, y : [n, c]   dyn : [n, dyn_stride]   base : see dyn_group_conv1d   hist : [n, c, taps-1] read_write
//   bindings: 1 x, 2 dyn, 3 base, 4 hist (read_write), 5 y (read_write)

const FLAG_RES_SCALE: u32 = 1u;
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
    flags: u32,
    res_scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read>       dyn:  array<f32>;
@group(0) @binding(3) var<storage, read>       base: array<f32>;
@group(0) @binding(4) var<storage, read_write> hist: array<f32>;
@group(0) @binding(5) var<storage, read_write> y:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n * p.c) { return; }
    let sq = idx / p.c;
    let ci = idx % p.c;
    let g = ci / p.group_size;
    let drow = sq * p.dyn_stride + p.dyn_off;
    let hb = idx * (p.taps - 1u);
    let xi = x[idx];

    var acc = 0.0;
    for (var j: u32 = 0u; j < p.taps; j = j + 1u) {
        var xv = xi;
        if (j > 0u) { xv = hist[hb + (p.taps - 1u - j)]; }
        var w = 0.0;
        if ((p.flags & FLAG_BASE_CK) != 0u) { w = base[p.base_off + ci * p.taps + (p.taps - 1u - j)]; }
        else { w = base[p.base_off + j * p.c + ci]; }
        if ((p.flags & FLAG_NO_DYN) == 0u) { w = w + dyn[drow + j * p.groups + g]; }
        acc = acc + w * xv;
    }
    if ((p.flags & FLAG_RES_SCALE) != 0u) { acc = (acc + xi) * p.res_scale; }
    y[idx] = acc;

    // Shift the window: drop the oldest input, append this token. Guarded so a single-tap convolution has no history.
    if (p.taps >= 2u) {
        for (var i: u32 = 0u; i + 2u < p.taps; i = i + 1u) { hist[hb + i] = hist[hb + i + 1u]; }
        hist[hb + p.taps - 2u] = xi;
    }
}
