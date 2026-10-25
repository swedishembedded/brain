// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-channel feature scaling that fans one row out to C channels:  out[r,c,k] = f[c,k] * x[r,k]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
//   x: [rows, m]   f: [C, m]   out: [rows, C, m]
//   params: rows, channels, m        dispatch: rows * C * m threads
//
// A bank of C diagonal operators applied to the same input, which is what a
// per-mode spectral filter is in a basis where the operator is diagonal. The
// same kernel is the input-gradient of `chan_scale_contract` (whose transpose
// it is): d t[r,c,k] = f[c,k] * dy[r,k].

struct Params {
    rows: u32,
    channels: u32,
    m: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       f:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let cm = p.channels * p.m;
    if (idx >= p.rows * cm) { return; }
    let r = idx / cm;
    let ck = idx % cm;
    let k = ck % p.m;
    out[idx] = f[ck] * x[r * p.m + k];
}
