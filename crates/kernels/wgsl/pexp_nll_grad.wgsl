// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of pexp_nll_value with respect to the log-hazard
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// d out[e] / d loglam[e] = w[i] * inv_wsum * (exp(loglam[e]) * expo[e] - event[e])
// Same layout and parameters as pexp_nll_value. Writes (does not accumulate)
// d_loglam: the log-hazard table has exactly one consumer, the head that
// produced it.

struct Params {
    rows: u32,
    k: u32,
    pieces: u32,
    inv_wsum: u32, // f32 bits
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       loglam:   array<f32>;
@group(0) @binding(2) var<storage, read>       event:    array<f32>;
@group(0) @binding(3) var<storage, read>       expo:     array<f32>;
@group(0) @binding(4) var<storage, read>       w:        array<f32>;
@group(0) @binding(5) var<storage, read_write> d_loglam: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let e = gid.y * (nwg.x * 64u) + gid.x;
    if (e >= p.rows * p.k) { return; }
    let subj = (e / p.k) / p.pieces;
    let scale = w[subj] * bitcast<f32>(p.inv_wsum);
    d_loglam[e] = scale * (exp(loglam[e]) * expo[e] - event[e]);
}
