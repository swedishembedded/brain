// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training banded channel convolution weight gradient: dw[j, t] += sum_r gy[r, j] u[r, j + t - k/2]
// @how   one thread per weight, looping over the rows; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-channel banded convolution along the channel axis: y[r, j] = sum_t w[j, t] * u[r, j + t - rad], rad = k / 2,
// zero outside [0, n). w is [n, k].
// The row loop is untuned (a row-split two-pass reduction is the follow-up for tall inputs).
// Params: rows, n, k. Dispatch: n*k threads.

struct Params { rows: u32, n: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  u: array<f32>;
@group(0) @binding(2) var<storage, read>  gy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n * p.k) { return; }
    let j = idx / p.k;
    let t = idx % p.k;
    let s = i32(j) + i32(t) - i32(p.k / 2u);
    if (s < 0 || s >= i32(p.n)) { return; }
    var acc = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        acc = acc + gy[r * p.n + j] * u[r * p.n + u32(s)];
    }
    dw[idx] = dw[idx] + acc;
}
