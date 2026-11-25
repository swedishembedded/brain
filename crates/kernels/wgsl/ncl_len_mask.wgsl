// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Zero the positions past each batch item's own length in a padded NCL batch, in place
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// A batch of sequences of different lengths padded to one `[n, c, l]` NCL
// tensor: item `b` is valid on `[0, lens[b])`. This writes 0 to every position
// at or past that length, so a conv that reads to the right of its output
// (a strided conv's last window) sees the zeros a single unpadded sequence
// would be padded with, not whatever the padding computed to:
//
//   b = idx / (c*l); t = idx % l;  x[idx] = 0 if t >= lens[b]
//
// `lens` is u32 per item. Params: total = n*c*l, c, l. Dispatch: total
// invocations. Valid positions are untouched (bit for bit).

struct Params {
    total: u32,
    c: u32,
    l: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x:    array<f32>;
@group(0) @binding(2) var<storage, read>       lens: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let b = idx / (p.c * p.l);
    let t = idx % p.l;
    if (t >= lens[b]) {
        x[idx] = 0.0;
    }
}
