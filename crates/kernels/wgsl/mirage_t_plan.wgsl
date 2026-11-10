// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training weighted segment plan, owner side: out[o] += sum over covering segments of w * other[...]
// @how   one thread per owned element; binary search of the owner interval, then its cover list
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// A plan is a set of weighted copies `out[dst..dst+len] += w * src[src..src+len]` whose destinations may overlap.
// The host splits the owner axis at every segment boundary into elementary intervals and lists, per interval, the
// segments covering it in segment order. Thread t owns one covered element: it finds its interval by binary search
// on the interval prefix counts and sums that interval's list in order, so each output element has one writer and
// the sum order is fixed. The forward uses the segments' destinations as the owner axis; the backward uses their
// sources (the transposed plan), which is why overlap on either side is race free.
//
// ivals: 4 words per interval: first owner element, prefix count of covered elements, list begin, list end.
// list: 3 words per entry: owner offset of the segment start, other offset of the segment start, weight (f32 bits).
// Params: total (covered elements), n_ivals. Dispatch: total threads.

struct Params { total: u32, n_ivals: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       ivals: array<u32>;
@group(0) @binding(2) var<storage, read>       list: array<u32>;
@group(0) @binding(3) var<storage, read>       other: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    var lo = 0u;
    var hi = p.n_ivals;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (ivals[mid * 4u + 1u] <= idx) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let o = ivals[lo * 4u] + (idx - ivals[lo * 4u + 1u]);
    var s = 0.0;
    for (var e = ivals[lo * 4u + 2u]; e < ivals[lo * 4u + 3u]; e = e + 1u) {
        s = s + bitcast<f32>(list[e * 3u + 2u]) * other[list[e * 3u + 1u] + (o - list[e * 3u])];
    }
    out[o] = out[o] + s;
}
