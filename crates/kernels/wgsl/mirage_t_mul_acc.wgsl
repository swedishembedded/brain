// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training backward of a product: dst[i] += g[i] * o[k(i)] with a periodic or per-row factor
// @how   one thread per output element; accumulates, one owner per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// k(i) = i % period (mode 0: elementwise product, per-channel and scalar scale)
//      = i / period (mode 1: one factor per row of width `period`).
// dst must not be g or o. Params: total, period, mode. Dispatch: total threads.

struct Params { total: u32, period: u32, mode: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       g: array<f32>;
@group(0) @binding(2) var<storage, read>       o: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    var k = idx % p.period;
    if (p.mode == 1u) { k = idx / p.period; }
    dst[idx] = dst[idx] + g[idx] * o[k];
}
