// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-channel feature scaling summed over C channels:  y[r,k] (+)= sum_c f[c,k] * t[r,c,k]
// @how   one thread per output element, serial over channels
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
//   t: [rows, C, m]   f: [C, m]   y: [rows, m]
//   params: rows, channels, m, accumulate (0 = assign, 1 = add)
//   dispatch: rows * m threads
//
// The transpose of `chan_scale_expand`: it weights each channel per feature
// and sums the channels, and it is also `chan_scale_expand`'s input gradient
// (dx[r,k] = sum_c f[c,k] * d out[r,c,k]). `accumulate = 1` adds into `y`, so
// two filter banks reading the same input can sum their input gradients into
// one buffer.

struct Params {
    rows: u32,
    channels: u32,
    m: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       t: array<f32>;
@group(0) @binding(2) var<storage, read>       f: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.m) { return; }
    let r = idx / p.m;
    let k = idx % p.m;
    let base = r * p.channels * p.m + k;
    var acc = 0.0;
    for (var c = 0u; c < p.channels; c = c + 1u) {
        acc = acc + f[c * p.m + k] * t[base + c * p.m];
    }
    if (p.accumulate != 0u) {
        y[idx] = y[idx] + acc;
    } else {
        y[idx] = acc;
    }
}
