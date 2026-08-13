// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row-wise argmax, FINAL stage
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Row-wise argmax, FINAL stage: fold each row's P partial (value, index) pairs.
//
//   part: [M, P, 2]  — from argmax_part
//   out : [M]        — winning column index as f32 (exact below 2^24)
//
// One thread per row over only P pairs (P ~ vocab/1024), so this stage is
// trivial. Tie-break: on equal values the LOWER index wins, matching a serial
// scan — partials arrive in ascending chunk order, so strict '>' preserves it.

struct Params {
    m: u32,
    p: u32,
};

@group(0) @binding(0) var<uniform> pr: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = gid.y * (nwg.x * 64u) + gid.x;
    if (row >= pr.m) { return; }
    var best_v = -3.402823e38;
    var best_n = 0.0;
    for (var c = 0u; c < pr.p; c = c + 1u) {
        let v = part[2u * (row * pr.p + c)];
        if (v > best_v) {
            best_v = v;
            best_n = part[2u * (row * pr.p + c) + 1u];
        }
    }
    out[row] = best_n;
}
