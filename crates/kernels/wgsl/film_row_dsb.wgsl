// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  FiLM per-row-group modulation, scale/shift gradient — spec
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// FiLM per-row-group modulation, scale/shift gradient. One invocation per (cond,d)
// (NC*D threads, NC = R/rows_per_cond): thread t has k = t/D, d = t%D.
// Sequential loop i in 0..rows_per_cond (ascending — a determinism contract),
// r = k*rows_per_cond + i:
//   ds += dy[r*D+d] * x[r*D+d]      db += dy[r*D+d]
// then writes BOTH halves of the packed dsb[NC,2D]:
//   dsb[k*2D + d] = ds              dsb[k*2D + D + d] = db
// OVERWRITES dsb (=, never += — s,b are activations here).
//

struct Params {
    R: u32,
    D: u32,
    rows_per_cond: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       dy:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dsb: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= (p.R / p.rows_per_cond) * p.D) { return; }
    let k = t / p.D;
    let d = t % p.D;
    var ds = 0.0;
    var db = 0.0;
    for (var i = 0u; i < p.rows_per_cond; i = i + 1u) {
        let r = k * p.rows_per_cond + i;
        ds = ds + dy[r * p.D + d] * x[r * p.D + d];
        db = db + dy[r * p.D + d];
    }
    dsb[k * 2u * p.D + d] = ds;
    dsb[k * 2u * p.D + p.D + d] = db;
}
