// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training mask loss backward: dx[i] += g (bw (p - t) / n + dw dDice/dp p (1 - p))
// @how   one thread per element; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// stats from mirage_t_bce_dice. Params: n, bw, dw (f32 bits). Dispatch: n threads.

struct Params { n: u32, bw: f32, dw: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<f32>;
@group(0) @binding(3) var<storage, read>       stats: array<f32>;
@group(0) @binding(4) var<storage, read>       g: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    let den = stats[0] + stats[2] + 1.0;
    let num = 2.0 * stats[1] + 1.0;
    let tg = tgt[idx];
    let pr = 1.0 / (1.0 + exp(-x[idx]));
    let ddice = -(2.0 * tg * den - num) / (den * den);
    dx[idx] = dx[idx] + (p.bw * (pr - tg) / f32(p.n) + p.dw * ddice * pr * (1.0 - pr)) * g[0];
}
