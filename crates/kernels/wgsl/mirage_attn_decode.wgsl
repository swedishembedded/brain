// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Single-query attention over a key slice: scores, softmax and weighted values of one (row, split) in one workgroup
// @how   64-thread workgroup per (row, key split): scores into workgroup memory, ONE barrier, then a lane per value column
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The decoder's attention has one query row per sequence and head, so the three-kernel path (scores, row softmax,
// weighted sum) is three dependent launches over a few kilobytes: latency, not bandwidth. This kernel does the
// three stages for a slice of up to `chunk` keys inside one workgroup:
//   phase 1: thread t scores keys lo + t, lo + t + 64, ... (the masked and not-yet-written keys score -1e30) and keeps
//            the maximum of its keys; every score goes to workgroup memory;
//   ONE barrier;
//   phase 2: every thread folds the 64 maxima, then lane t < vd walks the slice once accumulating
//            p_s = exp(score_s - max) and p_s * v[s, t]. The sum of the p_s is the same in every lane.
// With one split the output row is final (acc / sum, zero when no key is live). With S > 1 splits (a row's keys
// spread over S workgroups so a batch of one still fills the card) each workgroup writes its partial
// (acc[0..vd), max, sum) to `o`, and mirage_attn_merge folds the S partials of a row.
//
//   q: [batch * heads, hd]   k: [batch, nk, kvh, hd]   v: [batch, nk, kvh, vd]   key_mask: [batch, nk]
//   pos: one word; visible == 0xFFFFFFFF means pos[0] + 1 keys.
//   o: [batch * heads, vd] for one split, else [batch * heads, splits, vd + 2].
// Requires vd <= 64 and chunk <= 128. Dispatch: batch * heads * splits workgroups.

struct Params { batch: u32, heads: u32, kvh: u32, hd: u32, vd: u32, nk: u32, visible: u32, scale: f32, splits: u32, chunk: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q: array<f32>;
@group(0) @binding(2) var<storage, read>       k: array<f32>;
@group(0) @binding(3) var<storage, read>       v: array<f32>;
@group(0) @binding(4) var<storage, read>       key_mask: array<f32>;
@group(0) @binding(5) var<storage, read>       pos: array<u32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;

var<workgroup> sc: array<f32, 128>;
var<workgroup> pm: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wg.y * nwg.x + wg.x;
    if (w >= p.batch * p.heads * p.splits) { return; }
    let t = li.x;
    let row = w / p.splits;
    let sp = w % p.splits;
    let h = row % p.heads;
    let b = row / p.heads;
    let kv = h / (p.heads / p.kvh);
    let visible = select(p.visible, pos[0] + 1u, p.visible == 0xFFFFFFFFu);
    let lo = sp * p.chunk;
    let hi = min(min(lo + p.chunk, p.nk), visible);
    let n = select(0u, hi - lo, hi > lo);
    var m = -1.0e30;
    for (var i = t; i < n; i = i + 64u) {
        let s = lo + i;
        var r = -1.0e30;
        if (key_mask[b * p.nk + s] != 0.0) {
            let kb = ((b * p.nk + s) * p.kvh + kv) * p.hd;
            var dot = 0.0;
            for (var d = 0u; d < p.hd; d = d + 1u) { dot = dot + q[row * p.hd + d] * k[kb + d]; }
            r = dot * p.scale;
        }
        sc[i] = r;
        m = max(m, r);
    }
    pm[t] = m;
    workgroupBarrier();
    var gm = -1.0e30;
    for (var i = 0u; i < 64u; i = i + 1u) { gm = max(gm, pm[i]); }
    var acc = 0.0;
    var l = 0.0;
    if (gm > -1.0e29) {
        for (var i = 0u; i < n; i = i + 1u) {
            let e = exp(sc[i] - gm);
            l = l + e;
            if (t < p.vd) { acc = acc + e * v[((b * p.nk + lo + i) * p.kvh + kv) * p.vd + t]; }
        }
    }
    if (p.splits == 1u) {
        if (t < p.vd) {
            var r = 0.0;
            if (l > 0.0) { r = acc / l; }
            o[row * p.vd + t] = r;
        }
    } else {
        let base = w * (p.vd + 2u);
        if (t < p.vd) { o[base + t] = acc; }
        if (t == 0u) {
            o[base + p.vd] = gm;
            o[base + p.vd + 1u] = l;
        }
    }
}
