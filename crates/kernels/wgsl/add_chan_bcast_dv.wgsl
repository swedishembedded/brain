// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gradient of add_chan_bcast wrt the per-(image, channel) scalar
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Gradient of add_chan_bcast wrt the per-(image, channel) scalar: sum the map.
//   dy : [N, C, H, W]
//   dv : [N, C]   read_write   dv[n,c] = sum_hw dy[n,c,hw]
//
// (The gradient wrt `x` is dy itself — the caller reuses the buffer, no kernel.)
// The adjoint of a broadcast is a sum over the broadcast axes. One invocation per
// (n,c), serial over HW: no atomics.

struct Params {
    N: u32,
    C: u32,
    HW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dv: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C;
    if (idx >= total) { return; }
    let base = idx * p.HW;
    var acc = 0.0;
    for (var i: u32 = 0u; i < p.HW; i = i + 1u) { acc = acc + dy[base + i]; }
    dv[idx] = acc;
}
