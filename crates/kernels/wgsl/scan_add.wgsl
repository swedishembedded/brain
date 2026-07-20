// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Exclusive prefix scan, stage 2: add the (already exclusively scanned)
// per-block totals back onto every element of the corresponding block.
// One invocation per element.

struct Params {
    n: u32,      // element count in data
    block: u32,  // run length used by scan_block
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read>       sums: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    data[idx] = data[idx] + sums[idx / p.block];
}
