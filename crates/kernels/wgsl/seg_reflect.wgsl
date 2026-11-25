// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Refill the margins of a segment-packed batch with each item's own reflect padding (gather)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// A batch of sequences packed along the time axis: a `[c, n, s]` tensor whose
// row `(ch, b)` holds item `b`'s channel `ch` on `[off, off + lens[b])` with
// margins on both sides. This writes the whole row again with every position
// mirrored into the item's own valid range (PyTorch `mode="reflect"`, no edge
// repeat), so a "same" conv with pad `<= off` over the flattened `c * (n*s)`
// sequence reads, at every valid output, exactly the reflect-padded item:
//
//   row = idx / s; j = idx % s; b = row % n; L = lens[b]; i = j - off
//   i < 0 -> -i;  i >= L -> 2(L-1) - i;  then clamped into [0, L)
//   y[idx] = x[row*s + off + i]
//
// The clamp only touches positions farther out than one reflection, which no
// valid output of a conv with pad `<= off <= L - 1` reads. Params: total =
// c*n*s, n, s, off. Dispatch: total invocations. Pure movement.

struct Params {
    total: u32,
    n: u32,
    s: u32,
    off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> y:    array<f32>;
@group(0) @binding(3) var<storage, read>       lens: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let row = idx / p.s;
    let j = i32(idx % p.s);
    let len = i32(lens[row % p.n]);
    var i = j - i32(p.off);
    if (i < 0) { i = -i; }
    if (i >= len) { i = 2 * (len - 1) - i; }
    if (i < 0) { i = 0; }
    if (i >= len) { i = len - 1; }
    y[idx] = x[row * p.s + p.off + u32(i)];
}
