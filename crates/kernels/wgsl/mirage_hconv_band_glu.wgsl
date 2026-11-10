// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  HConvGLU gate stage on the original parameters: z = gelu_tanh(band_gate(u) + bias) * band_value(u)
// @how   one thread per output element, looping over the band taps; taps stored tap-major so neighbouring threads read neighbouring words
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Per-channel banded convolution along the channel axis, out[r, j] = sum_t w[t, j] * u[r, j + t - k/2], zero outside
// [0, n), for the gate and the value branch at once. gw and vw are [k, n] (tap-major), so the 65-tap band costs
// 2 * 65 multiply-adds per element where the folded 2n x n matrix costs 2n. Params: rows, n, k. Dispatch: rows*n threads.

struct Params { rows: u32, n: u32, k: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       u:    array<f32>;
@group(0) @binding(2) var<storage, read>       gw:   array<f32>;
@group(0) @binding(3) var<storage, read>       vw:   array<f32>;
@group(0) @binding(4) var<storage, read>       bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> z:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.n) { return; }
    let j = idx % p.n;
    let r = idx / p.n;
    let rad = i32(p.k / 2u);
    var a = bias[j];
    var v = 0.0;
    for (var t = 0u; t < p.k; t = t + 1u) {
        let s = i32(j) + i32(t) - rad;
        if (s < 0 || s >= i32(p.n)) { continue; }
        let x = u[r * p.n + u32(s)];
        a = a + gw[t * p.n + j] * x;
        v = v + vw[t * p.n + j] * x;
    }
    let k = 0.7978845608028654;
    let g = 0.5 * a * (1.0 + tanh(k * (a + 0.044715 * a * a * a)));
    z[idx] = g * v;
}
