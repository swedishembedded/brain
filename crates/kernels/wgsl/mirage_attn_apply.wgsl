// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Key-parallel masked GQA attention, pass 3: out[b, q, h, i] = sum_key probs[b, q, h, key] * v[b, key, kv, i]
// @how   64-thread workgroup per row block, keys split across thread groups, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// probs: [batch, nq, heads, nk]; v: [batch, nk, kvh, vd]; out: [batch, nq, heads, vd]; `visible` 0xFFFFFFFF means
// pos[0] + 1. A 64-thread workgroup owns `lanes` output channels of one (image, query, head) row and splits the
// keys over 64 / lanes groups of threads; the partial sums meet in shared memory behind one barrier.
// Params: batch, nq, heads, kvh, vd, nk, visible, lanes (a divisor of 64 and of vd).
// Dispatch: batch * nq * heads * (vd / lanes) workgroups.

struct Params { batch: u32, nq: u32, heads: u32, kvh: u32, vd: u32, nk: u32, visible: u32, lanes: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read>       pos: array<u32>;

var<workgroup> part: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wg.y * nwg.x + wg.x;
    let blocks = p.vd / p.lanes;
    if (w >= p.batch * p.nq * p.heads * blocks) { return; }
    let t = li.x;
    let lane = t % p.lanes;
    let g = t / p.lanes;
    let groups = 64u / p.lanes;
    let row = w / blocks;
    let i = (w % blocks) * p.lanes + lane;
    let h = row % p.heads;
    let b = row / (p.heads * p.nq);
    let kv = h / (p.heads / p.kvh);
    var visible = p.visible;
    if (visible == 0xFFFFFFFFu) { visible = pos[0] + 1u; }
    var acc = 0.0;
    for (var s = g; s < visible; s = s + groups) {
        acc = acc + probs[row * p.nk + s] * v[((b * p.nk + s) * p.kvh + kv) * p.vd + i];
    }
    part[t] = acc;
    workgroupBarrier();
    if (g == 0u) {
        var r = 0.0;
        for (var q = 0u; q < groups; q = q + 1u) { r = r + part[q * p.lanes + lane]; }
        out[row * p.vd + i] = r;
    }
}
