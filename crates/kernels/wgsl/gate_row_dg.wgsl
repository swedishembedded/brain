// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  adaLN gated residual, gate gradient — spec
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// adaLN gated residual, gate gradient — spec:
// docs/world-models/specs/P1.film.md §4.9. One invocation per (cond,d)
// (NC*D threads, NC = R/rows_per_cond): thread t has k = t/D, d = t%D.
// Sequential loop i in 0..rows_per_cond (ascending — determinism contract,
// spec §11), r = k*rows_per_cond + i:
//   s += dy[r*D+d] * h[r*D+d]
// writes dg[k*D + d] = s over the gate-shaped dg[NC,D].
// OVERWRITES dg (=, never += — g is an activation here, spec §1).
//

struct Params {
    R: u32,
    D: u32,
    rows_per_cond: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       h:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dg: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= (p.R / p.rows_per_cond) * p.D) { return; }
    let k = t / p.D;
    let d = t % p.D;
    var s = 0.0;
    for (var i = 0u; i < p.rows_per_cond; i = i + 1u) {
        let r = k * p.rows_per_cond + i;
        s = s + dy[r * p.D + d] * h[r * p.D + d];
    }
    dg[t] = s;
}
