// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training banded channel convolution forward: y[r, j] = sum_t w[j, t] u[r, j + t - k/2]
// @how   one thread per output element, looping over the k taps
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-channel banded convolution along the channel axis: y[r, j] = sum_t w[j, t] * u[r, j + t - rad], rad = k / 2,
// zero outside [0, n). w is [n, k].
// Params: rows, n, k. Dispatch: rows*n threads.

struct Params { rows: u32, n: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  u: array<f32>;
@group(0) @binding(2) var<storage, read>  w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.n) { return; }
    let j = idx % p.n;
    let r = idx / p.n;
    let rad = i32(p.k / 2u);
    var acc = 0.0;
    for (var t = 0u; t < p.k; t = t + 1u) {
        let s = i32(j) + i32(t) - rad;
        if (s < 0 || s >= i32(p.n)) { continue; }
        acc = acc + w[j * p.k + t] * u[r * p.n + u32(s)];
    }
    y[idx] = acc;
}
