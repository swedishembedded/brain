// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal tap convolution over a per-row history: out[b, c] = sum_j taps[j, c] * hist[b, pos - j*dilation, c] (positions below zero are absent); the current row comes from `src` and is also stored into `hist[b, pos]`
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// src: [rows, w] the current row, hist: [rows, maxt, w] (positions before `pos`; this kernel writes `pos`), taps: [ntaps, w].
// Each thread owns one (row, channel), so reading and writing its own history cell cannot race.
// The position is read from `pos[0]`. `src` rows may sit inside a wider fused projection: element (b, c) is src[b * src_stride + src_off + c].
// Params: rows, w, maxt, ntaps, dilation, src_stride, src_off. Dispatch: rows*w threads.

struct Params { rows: u32, w: u32, maxt: u32, ntaps: u32, dilation: u32, src_stride: u32, src_off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<f32>;
@group(0) @binding(3) var<storage, read>       taps: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read>       pos_buf: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.w) { return; }
    let c = idx % p.w;
    let b = idx / p.w;
    let pos = pos_buf[0];
    let cur = src[b * p.src_stride + p.src_off + c];
    hist[(b * p.maxt + pos) * p.w + c] = cur;
    var acc = taps[c] * cur;
    for (var j = 1u; j < p.ntaps; j = j + 1u) {
        let back = j * p.dilation;
        if (back > pos) { break; }
        acc = acc + taps[j * p.w + c] * hist[(b * p.maxt + pos - back) * p.w + c];
    }
    out[idx] = acc;
}
