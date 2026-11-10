// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training column gather: out[r, i] (+)= z[r, idx[i]] (a permutation forward, its inverse backward)
// @how   one thread per output element; assigns or accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: total, n, acc. Dispatch: total threads.

struct Params { total: u32, n: u32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       z: array<f32>;
@group(0) @binding(2) var<storage, read>       cols: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let i = idx % p.n;
    let r = idx / p.n;
    let v = z[r * p.n + cols[i]];
    if (p.acc != 0u) {
        out[idx] = out[idx] + v;
    } else {
        out[idx] = v;
    }
}
