// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-ROW (per-token) max/x/ → int8 scale, for outlier-robust activation quantization
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Per-ROW (per-token) max|x| → int8 scale, for outlier-robust activation
// quantization: sx[m] = max|x[m,:]| / 127. One thread per row. Per-token scales
// (vs one per tensor) are what keep a deep int8 activation path accurate — a
// single outlier token no longer crushes every other token's resolution.

struct Params { m: u32, k: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read_write> sx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let m = gid.y * (nwg.x * 64u) + gid.x;
    if (m >= p.m) { return; }
    let base = m * p.k;
    var a = 0.0;
    for (var c: u32 = 0u; c < p.k; c = c + 1u) {
        a = max(a, abs(x[base + c]));
    }
    sx[m] = max(a, 1e-8) / 127.0;
}
