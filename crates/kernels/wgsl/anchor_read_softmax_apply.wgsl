// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read steps 2 and 3 in one kernel: row softmax over the logits (with the sink) and the weighted sum of values
// @how   one workgroup per (row, KV group, 32-channel tile), all the group's heads at once; online softmax statistics merged across threads, the bank slots split over 8 lanes
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// out[row, h, :] = sum_c softmax(logits[row, h, :])[c] * value[s * t + ref(c), g, :]
//
// exactly `row softmax` followed by `anchor_read_apply` (see those kernels), without writing the probabilities
// or reading them back: the inference form of the read, for decode steps and chunks that keep no backward.
// The softmax statistics are an online (max, sum) pair per thread, merged by every thread after one barrier;
// the slots `c` are split over 8 lanes of 32 channels, whose partial sums one more barrier folds.
//
//   logits : [n_seq * t * n_heads, capacity + 1]   the sink is column `capacity` and carries no value
//   value, refs, slots, out : as `anchor_read_apply`
// Each value row is read once for the `n_heads / groups` (at most 8) heads that read it.
// Dispatch: n_seq * t * groups * ceil(head_dim / 32) workgroups.

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
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       logits: array<f32>;
@group(0) @binding(2) var<storage, read>       value: array<f32>;
@group(0) @binding(3) var<storage, read>       refs:  array<u32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;
@group(0) @binding(5) var<storage, read>       slots: array<u32>;

var<workgroup> stat_m: array<f32, 256>;
var<workgroup> stat_s: array<f32, 256>;
var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let tiles = (hd + 31u) / 32u;
    let wg = wid.y * nwg.x + wid.x;
    if (wg >= p.rows * p.groups * tiles) { return; }
    let t = lid.x;
    let rg = wg / tiles;
    let tile = wg % tiles;
    let g = rg % p.groups;
    let row = rg / p.groups;
    let hpg = p.n_heads / p.groups;
    let cols = p.capacity + 1u;
    let first = row * p.n_heads + g * hpg;   // the group's first (row, head)

    // Online softmax statistics: thread (head j, partial k) covers the columns k, k + 32, ... of head j.
    {
        let j = t / 32u;
        let k = t % 32u;
        var m = -3.0e38;
        var s = 0.0;
        if (j < hpg) {
            let lrow = (first + j) * cols;
            for (var c = k; c < cols; c = c + 32u) {
                let l = logits[lrow + c];
                if (l > m) {
                    s = s * exp(m - l);
                    m = l;
                }
                s = s + exp(l - m);
            }
        }
        stat_m[t] = m;
        stat_s[t] = s;
    }
    workgroupBarrier();
    var big: array<f32, 8>;
    var total: array<f32, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) {
        if (j < hpg) {
            var b = -3.0e38;
            for (var i = 0u; i < 32u; i = i + 1u) { b = max(b, stat_m[j * 32u + i]); }
            var sum = 0.0;
            for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + stat_s[j * 32u + i] * exp(stat_m[j * 32u + i] - b); }
            big[j] = b;
            total[j] = sum;
        }
    }

    // The weighted sums of this tile's channels over the slots, 8 lanes apart.
    let d = tile * 32u + (t % 32u);
    let lane = t / 32u;
    let sq = row / p.t;
    let pos = row % p.t;
    let n_seq = p.rows / p.t;
    let bank = (((pos / p.chunk) * n_seq + sq) * p.groups + g) * p.capacity;
    var base = sq * p.kv_rows;
    if (p.slotted != 0u) { base = slots[sq] * p.kv_rows; }
    var acc: array<f32, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) { acc[j] = 0.0; }
    if (d < hd) {
        for (var c = lane; c < p.capacity; c = c + 8u) {
            let slot = refs[bank + c];
            if (slot == 0u) { continue; }
            let v = value[(base + slot - 1u) * p.groups * hd + g * hd + d];
            for (var j = 0u; j < 8u; j = j + 1u) {
                if (j < hpg) { acc[j] = acc[j] + exp(logits[(first + j) * cols + c] - big[j]) * v; }
            }
        }
    }
    for (var j = 0u; j < 8u; j = j + 1u) { part[j * 256u + t] = acc[j]; }
    workgroupBarrier();
    if (lane == 0u && d < hd) {
        for (var j = 0u; j < hpg; j = j + 1u) {
            var sum = 0.0;
            for (var k = 0u; k < 8u; k = k + 1u) { sum = sum + part[j * 256u + k * 32u + t]; }
            out[(first + j) * hd + d] = sum / total[j];
        }
    }
}
