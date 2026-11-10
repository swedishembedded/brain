// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Write one row per sequence into a history table: hist[b, pos, :] = src[b, :] (KV cache and tap histories)
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: rows, w, maxt, pos. Dispatch: rows*w threads.

struct Params { rows: u32, w: u32, maxt: u32, pos: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.w) { return; }
    let c = idx % p.w;
    let b = idx / p.w;
    hist[(b * p.maxt + p.pos) * p.w + c] = src[idx];
}
