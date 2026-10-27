// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read steps 2 and 3 in one kernel: row softmax over the logits (with the sink) and the weighted sum of values
// @how   one workgroup per (row, head, 32-channel tile); online softmax statistics merged across threads, the bank slots split over 8 lanes
// @opt   2
// @cpu   no
// @gpu   yes
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
// Dispatch: n_seq * t * n_heads * ceil(head_dim / 32) workgroups.

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
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let tiles = (hd + 31u) / 32u;
    let wg = wid.y * nwg.x + wid.x;
    if (wg >= p.rows * p.n_heads * tiles) { return; }
    let t = lid.x;
    let rh = wg / tiles;
    let tile = wg % tiles;
    let cols = p.capacity + 1u;
    let lrow = rh * cols;

    // Online softmax statistics over this thread's columns.
    var m = -3.0e38;
    var s = 0.0;
    for (var c = t; c < cols; c = c + 256u) {
        let l = logits[lrow + c];
        if (l > m) {
            s = s * exp(m - l);
            m = l;
        }
        s = s + exp(l - m);
    }
    stat_m[t] = m;
    stat_s[t] = s;
    workgroupBarrier();
    var big = -3.0e38;
    for (var i = 0u; i < 256u; i = i + 1u) { big = max(big, stat_m[i]); }
    var total = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) { total = total + stat_s[i] * exp(stat_m[i] - big); }

    // The weighted sum of this tile's channels over the slots, 8 lanes apart.
    let d = tile * 32u + (t % 32u);
    let lane = t / 32u;
    let h = rh % p.n_heads;
    let row = rh / p.n_heads;
    let sq = row / p.t;
    let pos = row % p.t;
    let n_seq = p.rows / p.t;
    let g = h / (p.n_heads / p.groups);
    let bank = (((pos / p.chunk) * n_seq + sq) * p.groups + g) * p.capacity;
    var base = sq * p.kv_rows;
    if (p.slotted != 0u) { base = slots[sq] * p.kv_rows; }
    var acc = 0.0;
    if (d < hd) {
        for (var c = lane; c < p.capacity; c = c + 8u) {
            let slot = refs[bank + c];
            if (slot == 0u) { continue; }
            acc = acc + exp(logits[lrow + c] - big) * value[(base + slot - 1u) * p.groups * hd + g * hd + d];
        }
    }
    part[t] = acc;
    workgroupBarrier();
    if (lane == 0u && d < hd) {
        var sum = 0.0;
        for (var k = 0u; k < 8u; k = k + 1u) { sum = sum + part[k * 32u + t]; }
        out[rh * hd + d] = sum / total;
    }
}
