// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  FiLM per-channel modulation, scale/shift gradient - spec
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// FiLM per-channel modulation, scale/shift gradient. One invocation per (n,c) pair
// (N*C threads): thread t has n = t/C, c = t%C. ONE sequential loop over
// (h,w) in ascending order (a determinism contract) accumulating
//   ds += dy[i] * x[i]      db += dy[i]
// then writing BOTH halves of the packed dsb[N,2C]:
//   dsb[n*2C + c] = ds      dsb[n*2C + C + c] = db
// OVERWRITES dsb (=, never += — s,b are activations here).
//

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
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
    if (t >= p.N * p.C) { return; }
    let n = t / p.C;
    let c = t % p.C;
    let hw = p.H * p.W;
    let base = (n * p.C + c) * hw;
    var ds = 0.0;
    var db = 0.0;
    for (var i = 0u; i < hw; i = i + 1u) {
        ds = ds + dy[base + i] * x[base + i];
        db = db + dy[base + i];
    }
    dsb[n * 2u * p.C + c] = ds;
    dsb[n * 2u * p.C + p.C + c] = db;
}
