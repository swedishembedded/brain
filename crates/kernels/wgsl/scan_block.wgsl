// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Exclusive prefix scan, stage 1 of the generic multi-pass scan
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Exclusive prefix scan, stage 1 of the generic multi-pass scan: each
// invocation owns one `block`-sized run of `data`, replaces it with its
// exclusive scan in place, and writes the run's total to `sums[block_idx]`.
// Host orchestration recursively scans `sums` and then applies `scan_add`.
// u32 payload (counts/offsets); no atomics, no barriers — CPU-JIT safe.

struct Params {
    n: u32,      // element count in data
    block: u32,  // run length per invocation (256 by convention)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let n_blocks = (p.n + p.block - 1u) / p.block;
    if (idx >= n_blocks) { return; }
    let start = idx * p.block;
    let end = min(start + p.block, p.n);
    var running = 0u;
    for (var i = start; i < end; i = i + 1u) {
        let v = data[i];
        data[i] = running;
        running = running + v;
    }
    sums[idx] = running;
}
