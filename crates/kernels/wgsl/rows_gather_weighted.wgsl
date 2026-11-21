// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Weighted row gather: out[r] = sum over taps t of w[r, t] * table[idx[r, t]], the resampling of a learned table (bilinear position embeddings)
// @how   one thread per output element, serial sum over the taps
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// For r < rows and c < width:
//
//   out[r * width + c] = sum_{t < taps} w[r * taps + t] * table[idx[r * taps + t] * width + c]
//
// The taps of one row are summed in order t = 0, 1, ... in f32, the reference's `(embedding(idx) * w).sum(1)` for a
// handful of taps. `idx` holds table row numbers (a bilinear resample of a learned position grid has 4 taps per
// output row; any tap count works). The adjoint with respect to the table is `rows_gather_weighted_bwd.wgsl`.
//
//   params: rows, taps, width

struct Params {
    rows: u32,
    taps: u32,
    width: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       idx:   array<u32>;
@group(0) @binding(2) var<storage, read>       w:     array<f32>;
@group(0) @binding(3) var<storage, read>       table: array<f32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= p.rows * p.width) { return; }
    let r = t / p.width;
    let c = t % p.width;
    var acc = 0.0;
    for (var k = 0u; k < p.taps; k = k + 1u) {
        acc = acc + w[r * p.taps + k] * table[idx[r * p.taps + k] * p.width + c];
    }
    out[t] = acc;
}
