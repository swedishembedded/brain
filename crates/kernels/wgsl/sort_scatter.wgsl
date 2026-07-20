// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// LSD radix sort, stage 2: stable per-chunk scatter. `hist` holds the
// exclusive scan of the column-major histograms (sort_hist), so
// `hist[digit * n_chunks + chunk_idx]` is exactly where this chunk's first
// key with that digit belongs globally. Each invocation walks its chunk in
// order, bumping a private copy of its 256 offsets — stable within the chunk
// and disjoint across chunks, hence no atomics.

struct Params {
    n: u32,        // total key count
    shift: u32,    // digit bit offset (0, 8, 16, 24)
    n_chunks: u32, // ceil(n / chunk)
    chunk: u32,    // keys per invocation
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       keys_in:  array<u32>;
@group(0) @binding(2) var<storage, read>       vals_in:  array<u32>;
@group(0) @binding(3) var<storage, read>       hist:     array<u32>;
@group(0) @binding(4) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals_out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n_chunks) { return; }
    var off: array<u32, 256>;
    for (var d = 0u; d < 256u; d = d + 1u) {
        off[d] = hist[d * p.n_chunks + idx];
    }
    let start = idx * p.chunk;
    let end = min(start + p.chunk, p.n);
    for (var i = start; i < end; i = i + 1u) {
        let k = keys_in[i];
        let d = (k >> p.shift) & 255u;
        let pos = off[d];
        off[d] = pos + 1u;
        keys_out[pos] = k;
        vals_out[pos] = vals_in[i];
    }
}
