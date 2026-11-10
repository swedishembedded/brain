// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training RMS-norm scale gradient: ds[j] += sum over rows of g[r, j] * x[r, j] * inv[r]
// @how   64-thread workgroup per 8 channels (8 row lanes), 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Tiled column reduction: a 64-thread workgroup owns 8 adjacent columns; thread (rl, cl) sums rows rl, rl+8, ... of
// column cl, then the 8 row lanes are folded in a fixed order (one barrier), so the sum is deterministic and the
// loads of a row lane touch 8 adjacent columns.
// Params: rows, d. Dispatch: ceil(d / 8) workgroups.

struct Params { rows: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  x: array<f32>;
@group(0) @binding(2) var<storage, read>  g: array<f32>;
@group(0) @binding(3) var<storage, read>  stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> ds: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let col0 = row * 8u;
    let cl = t % 8u;
    let rl = t / 8u;
    let c = col0 + cl;
    var s = 0.0;
    if (c < p.d) {
        for (var r = rl; r < p.rows; r = r + 8u) {
            s = s + g[r * p.d + c] * x[r * p.d + c] * stats[r * 2u];
        }
    }
    partial[t] = s;
    workgroupBarrier();
    if (t < 8u) {
        let c2 = col0 + t;
        if (c2 < p.d) {
            var tot = 0.0;
            for (var k = 0u; k < 8u; k = k + 1u) { tot = tot + partial[k * 8u + t]; }
            ds[c2] = ds[c2] + tot;
        }
    }
}
