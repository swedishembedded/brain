// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read, split over the bank's slots: per 32-slot piece the logits, the softmax statistics and the unnormalised weighted values
// @how   one workgroup per (row, KV group, 32-slot split); 8 threads share a key row (coalesced 4-wide loads), a thread per channel accumulates the values; `anchor_read_merge` folds the splits
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The inference read of the bank (`anchor_read_scores_group` + `anchor_read_softmax_apply`) as flash-decoding: the
// bank's slots are cut into splits of 32, each its own workgroup, so a single sequence's read fills the card instead
// of a handful of multiprocessors. A split writes, for each of the group's heads, the record of `head_dim + 4` words
//   [max logit, sum of exp(logit - max), 0, 0, acc[head_dim]]    acc[d] = sum_c exp(logit_c - max) * value_c[d]
// over its live slots (a split with none writes max = -3.4e38, sum = 0, acc = 0). The merge adds the sink (logit 0,
// no value) and normalises. Same `refs`, `slots`, key and value layouts as `anchor_read_scores_group`.
// Requires head_dim % 4 == 0 and at most 8 heads per group.
// Dispatch: n_seq * t * groups * ceil(capacity / 32) workgroups.

struct Params {
    rows: u32,       // n_seq * t
    t: u32,
    n_heads: u32,
    groups: u32,
    capacity: u32,
    head_dim: u32,
    chunk: u32,
    kv_rows: u32,
    slotted: u32,
    splits: u32,     // ceil(capacity / 32)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       key:   array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       value: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       refs:  array<u32>;
@group(0) @binding(5) var<storage, read>       slots: array<u32>;
@group(0) @binding(6) var<storage, read_write> part:  array<vec4<f32>>;

var<workgroup> dots: array<f32, 2048>;   // [head 8][slot 32][part 8]
var<workgroup> lg: array<f32, 256>;      // [head 8][slot 32]
var<workgroup> pe: array<f32, 256>;
var<workgroup> sl: array<u32, 32>;
var<workgroup> sred: array<vec4<f32>, 1536>;   // [lane 1..3][head 8][channel quad 64]

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let wg = wid.y * nwg.x + wid.x;
    if (wg >= p.rows * p.groups * p.splits) { return; }
    let t = lid.x;
    let sp = wg % p.splits;
    let rg = wg / p.splits;
    let g = rg % p.groups;
    let row = rg / p.groups;
    let hpg = p.n_heads / p.groups;
    let hd = p.head_dim;
    let hd4 = hd / 4u;
    let first = row * p.n_heads + g * hpg;
    let sq = row / p.t;
    let pos = row % p.t;
    let n_seq = p.rows / p.t;
    let bank = (((pos / p.chunk) * n_seq + sq) * p.groups + g) * p.capacity;
    var base = sq * p.kv_rows;
    if (p.slotted != 0u) { base = slots[sq] * p.kv_rows; }

    // Phase 1: partial dot products, 8 threads per slot.
    let i = t / 8u;
    let k = t % 8u;
    let c = sp * 32u + i;
    var slot = 0u;
    if (c < p.capacity) { slot = refs[bank + c]; }
    if (k == 0u) { sl[i] = slot; }
    var dot: array<f32, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) { dot[j] = 0.0; }
    if (slot != 0u) {
        let k_base = ((base + slot - 1u) * p.groups + g) * hd4;
        for (var d = k; d < hd4; d = d + 8u) {
            let kv = key[k_base + d];
            for (var j = 0u; j < 8u; j = j + 1u) {
                if (j < hpg) {
                    let qv = q[(first + j) * hd4 + d];
                    var a = dot[j];
                    a = a + qv.x * kv.x;
                    a = a + qv.y * kv.y;
                    a = a + qv.z * kv.z;
                    a = a + qv.w * kv.w;
                    dot[j] = a;
                }
            }
        }
    }
    for (var j = 0u; j < 8u; j = j + 1u) { dots[(j * 32u + i) * 8u + k] = dot[j]; }
    workgroupBarrier();

    // Phase 2: logits, then each head's maximum and the exponentials.
    let scale = inverseSqrt(f32(hd));
    {
        let j = t / 32u;
        let ci = t % 32u;
        var l = -3.4e38;
        if (j < hpg && sl[ci] != 0u) {
            var s = 0.0;
            for (var u = 0u; u < 8u; u = u + 1u) { s = s + dots[(j * 32u + ci) * 8u + u]; }
            l = s * scale;
        }
        lg[t] = l;
    }
    workgroupBarrier();
    {
        let j = t / 32u;
        let ci = t % 32u;
        var m = -3.4e38;
        for (var u = 0u; u < 32u; u = u + 1u) { m = max(m, lg[j * 32u + u]); }
        var e = 0.0;
        if (j < hpg && sl[ci] != 0u) { e = exp(lg[t] - m); }
        pe[t] = e;
        if (ci == 0u && j < hpg) {
            var s = 0.0;
            for (var u = 0u; u < 32u; u = u + 1u) { s = s + exp(lg[j * 32u + u] - m) * select(0.0, 1.0, sl[u] != 0u); }
            let rec = (((rg * p.splits + sp) * hpg + j) * (hd + 4u)) / 4u;
            part[rec] = vec4<f32>(m, s, 0.0, 0.0);
        }
    }
    workgroupBarrier();

    // Phase 3: the weighted values. Four lanes of 64 threads take every fourth slot; a thread owns a quad of channels.
    let lane = t / 64u;
    let quad = t % 64u;
    for (var d0 = 0u; d0 < hd4; d0 = d0 + 64u) {
        let d4 = d0 + quad;
        let live = d4 < hd4;
        var acc: array<vec4<f32>, 8>;
        for (var j = 0u; j < 8u; j = j + 1u) { acc[j] = vec4<f32>(0.0); }
        for (var u = lane; u < 32u; u = u + 4u) {
            let sv = sl[u];
            if (sv == 0u || !live) { continue; }
            let v = value[((base + sv - 1u) * p.groups + g) * hd4 + d4];
            for (var j = 0u; j < 8u; j = j + 1u) {
                if (j < hpg) { acc[j] = acc[j] + pe[j * 32u + u] * v; }
            }
        }
        if (lane != 0u) {
            for (var j = 0u; j < 8u; j = j + 1u) { sred[((lane - 1u) * 8u + j) * 64u + quad] = acc[j]; }
        }
        workgroupBarrier();
        if (lane == 0u && live) {
            for (var j = 0u; j < hpg; j = j + 1u) {
                let sum = ((acc[j] + sred[j * 64u + quad]) + sred[(8u + j) * 64u + quad]) + sred[(16u + j) * 64u + quad];
                part[(((rg * p.splits + sp) * hpg + j) * (hd + 4u)) / 4u + 1u + d4] = sum;
            }
        }
        workgroupBarrier();
    }
}
