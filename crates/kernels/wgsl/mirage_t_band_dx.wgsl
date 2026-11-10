// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training banded channel convolution input gradient: du[r, m] += sum_j gy[r, j] w[j, m - j + k/2]
// @how   one thread per input element, looping over the taps that read it; accumulates
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
@group(0) @binding(1) var<storage, read>  gy: array<f32>;
@group(0) @binding(2) var<storage, read>  w: array<f32>;
@group(0) @binding(3) var<storage, read_write> du: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.n) { return; }
    let m = idx % p.n;
    let r = idx / p.n;
    let rad = i32(p.k / 2u);
    var acc = 0.0;
    for (var t = 0u; t < p.k; t = t + 1u) {
        // u[m] feeds y[j] through tap t when j + t - rad = m.
        let j = i32(m) + rad - i32(t);
        if (j < 0 || j >= i32(p.n)) { continue; }
        acc = acc + gy[r * p.n + u32(j)] * w[u32(j) * p.k + t];
    }
    du[idx] = du[idx] + acc;
}
