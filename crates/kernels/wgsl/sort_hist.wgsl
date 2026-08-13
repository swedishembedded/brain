// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  LSD radix sort, stage 1: per-chunk 256-bin digit histogram
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype n/a
//
// LSD radix sort, stage 1: per-chunk 256-bin digit histogram. Each invocation
// owns one `chunk`-sized run of `keys` and counts the 8-bit digit at `shift`
// into a private local table, then writes it COLUMN-MAJOR into `hist`
// (`hist[digit * n_chunks + chunk_idx]`), so a single exclusive scan of the
// whole hist array yields the combined global-digit-base + per-chunk offsets
// that sort_scatter consumes. No atomics, no barriers.

struct Params {
    n: u32,        // total key count
    shift: u32,    // digit bit offset (0, 8, 16, 24)
    n_chunks: u32, // ceil(n / chunk)
    chunk: u32,    // keys per invocation
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n_chunks) { return; }
    var counts: array<u32, 256>;
    for (var d = 0u; d < 256u; d = d + 1u) { counts[d] = 0u; }
    let start = idx * p.chunk;
    let end = min(start + p.chunk, p.n);
    for (var i = start; i < end; i = i + 1u) {
        let d = (keys[i] >> p.shift) & 255u;
        counts[d] = counts[d] + 1u;
    }
    for (var d = 0u; d < 256u; d = d + 1u) {
        hist[d * p.n_chunks + idx] = counts[d];
    }
}
