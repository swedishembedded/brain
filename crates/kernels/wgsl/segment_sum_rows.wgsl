// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Masked sum of each fixed-length segment of rows: pooling of contiguous per-item rows
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// x is [segments * len, d] row-major: the rows of segment s are s*len ..
// (s+1)*len - 1, contiguous. For every segment s and column c:
//   out[s, c] = sum_{j < len} mask[s*len + j] * x[(s*len + j)*d + c]
// with mask a [segments * len] u32 of 0/1 (a row that does not belong to the
// sum - padding, a summary row - is 0). One thread per (s, c); adjacent
// threads read adjacent columns of the same row, so every step of the serial
// walk is coalesced, and the walk is `len` long, not the whole tensor - the
// dense formulation (a [segments, segments*len] membership matrix times x)
// spends segments times the work on zeros. Writes (does not accumulate).
// Backward: segment_bcast_rows.

struct Params {
    segments: u32,
    len: u32,
    d: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       mask: array<u32>;
@group(0) @binding(2) var<storage, read>       x:    array<f32>;
@group(0) @binding(3) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.segments * p.d) { return; }
    let s = i / p.d;
    let c = i % p.d;
    var acc = 0.0;
    for (var j = 0u; j < p.len; j = j + 1u) {
        let row = s * p.len + j;
        if (mask[row] != 0u) {
            acc = acc + x[row * p.d + c];
        }
    }
    out[i] = acc;
}
