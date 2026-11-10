// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training elementwise forward: add, multiply or scale with a periodic or per-row operand
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// out[i] = f(a[i], b) with b indexed by the period:
//   mode 0  a[i] + b[i % period]      (add, bias add)
//   mode 1  a[i] * b[i % period]      (multiply, per-channel scale, scalar scale with period 1)
//   mode 2  a[i] * c                  (constant scale; b is not read)
//   mode 3  a[i] + c                  (constant offset; b is not read)
//   mode 4  a[i] * b[i / period]      (one factor per row of width `period`)
// Params: total, period, mode, c (f32 bits). Dispatch: total threads.

struct Params { total: u32, period: u32, mode: u32, c: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a: array<f32>;
@group(0) @binding(2) var<storage, read>       b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let x = a[idx];
    var y = 0.0;
    if (p.mode == 0u) {
        y = x + b[idx % p.period];
    } else if (p.mode == 1u) {
        y = x * b[idx % p.period];
    } else if (p.mode == 2u) {
        y = x * p.c;
    } else if (p.mode == 3u) {
        y = x + p.c;
    } else {
        y = x * b[idx / p.period];
    }
    out[idx] = y;
}
