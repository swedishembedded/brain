// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  weighted_gap gradient wrt the WEIGHT MAP
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// weighted_gap gradient wrt the WEIGHT MAP.
//   dy : [N, C]
//   x  : [N, C, H*W]
//   dm : [N, 1, H*W]   read_write (one invocation per WEIGHT element)
//
//   dm[n,hw] = sum_c dy[n,c] * x[n,c,hw]
//
// Unlike dx this contracts over C, because one weight touches every channel.
// Still a gather — one invocation owns each dm element and sums C terms.

struct Params {
    N: u32,
    C: u32,
    HW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       x:  array<f32>;
@group(0) @binding(3) var<storage, read_write> dm: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.HW;
    if (idx >= total) { return; }
    let i = idx % p.HW;
    let n = idx / p.HW;
    var acc = 0.0;
    for (var c: u32 = 0u; c < p.C; c = c + 1u) {
        acc = acc + dy[n * p.C + c] * x[(n * p.C + c) * p.HW + i];
    }
    dm[idx] = acc;
}
