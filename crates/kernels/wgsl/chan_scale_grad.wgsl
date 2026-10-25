// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Filter gradient of a per-channel feature scaling:  df[c,k] += sum_r x[r,k] * g[r,c,k]
// @how   one thread per filter element, serial over rows
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
//   x: [rows, m]   g: [rows, C, m]   df: [C, m]
//   params: rows, channels, m        dispatch: C * m threads
//
// The weight gradient shared by `chan_scale_expand` (g = d out, x = its input)
// and `chan_scale_contract` (x = d y, g = its channel input t). It ADDS into
// `df`, the convention of every parameter-gradient kernel: the trainer zeroes
// gradients once per step and every use of a parameter accumulates. Adjacent
// threads take adjacent k, so each row's loads of x and g are coalesced.

struct Params {
    rows: u32,
    channels: u32,
    m: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       g:  array<f32>;
@group(0) @binding(3) var<storage, read_write> df: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let cm = p.channels * p.m;
    if (idx >= cm) { return; }
    let k = idx % p.m;
    var acc = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        acc = acc + x[r * p.m + k] * g[r * cm + idx];
    }
    df[idx] = df[idx] + acc;
}
