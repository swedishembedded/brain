// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training RMS-norm input gradient: dx[r, j] += inv[r] * w[j] * g[r, j] - x[r, j] * k[r]
// @how   one thread per element; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// stats from mirage_t_rms_stats. Params: total, d. Dispatch: total threads.

struct Params { total: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       g: array<f32>;
@group(0) @binding(3) var<storage, read>       w: array<f32>;
@group(0) @binding(4) var<storage, read>       stats: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let r = idx / p.d;
    let j = idx % p.d;
    dx[idx] = dx[idx] + stats[r * 2u] * w[j] * g[idx] - x[idx] * stats[r * 2u + 1u];
}
