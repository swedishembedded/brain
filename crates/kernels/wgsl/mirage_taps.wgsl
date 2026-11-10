// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal tap convolution over a per-row history: out[b, c] = sum_j taps[j, c] * hist[b, pos - j*dilation, c] (positions below zero are absent)
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// hist: [rows, maxt, w] (the current position already written), taps: [ntaps, w].
// Params: rows, w, maxt, pos, ntaps, dilation. Dispatch: rows*w threads.

struct Params { rows: u32, w: u32, maxt: u32, pos: u32, ntaps: u32, dilation: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       hist: array<f32>;
@group(0) @binding(2) var<storage, read>       taps: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.w) { return; }
    let c = idx % p.w;
    let b = idx / p.w;
    var acc = 0.0;
    for (var j = 0u; j < p.ntaps; j = j + 1u) {
        let back = j * p.dilation;
        if (back > p.pos) { break; }
        acc = acc + taps[j * p.w + c] * hist[(b * p.maxt + p.pos - back) * p.w + c];
    }
    out[idx] = acc;
}
