// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Merge of the split anchor-memory read: fold the splits' softmax records and add the sink
// @how   one workgroup per (row, query head); four lanes fold every fourth split of a channel quad; weights from the records' maxima first
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// out[row, h, :] = sum_sp acc_sp * exp(m_sp - M) / (exp(0 - M) + sum_sp s_sp * exp(m_sp - M)),  M = max(0, max_sp m_sp)
// over the records `anchor_read_split` wrote (`head_dim + 4` words each: m, s, two pad words, acc); the sink is the zero
// logit that carries no value. At most 256 splits.
// Dispatch: n_seq * t * n_heads workgroups.

struct Params {
    rows: u32,
    t: u32,
    n_heads: u32,
    groups: u32,
    capacity: u32,
    head_dim: u32,
    chunk: u32,
    kv_rows: u32,
    slotted: u32,
    splits: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out:  array<vec4<f32>>;

var<workgroup> wts: array<f32, 256>;
var<workgroup> sred: array<vec4<f32>, 192>;
var<workgroup> tot: array<f32, 2>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let wg = wid.y * nwg.x + wid.x;
    if (wg >= p.rows * p.n_heads) { return; }
    let t = lid.x;
    let row = wg / p.n_heads;
    let h = wg % p.n_heads;
    let hpg = p.n_heads / p.groups;
    let g = h / hpg;
    let j = h % hpg;
    let hd = p.head_dim;
    let hd4 = hd / 4u;
    let rg = row * p.groups + g;
    let stride4 = (hd + 4u) / 4u;
    let rec0 = (rg * p.splits * hpg + j) * stride4;        // split 0's record; split sp is `sp * hpg * stride4` further

    // Weights: exp(m_sp - M), and the normaliser with the sink.
    var big = 0.0;
    for (var sp = 0u; sp < p.splits; sp = sp + 1u) { big = max(big, part[rec0 + sp * hpg * stride4].x); }
    if (t < p.splits) { wts[t] = exp(part[rec0 + t * hpg * stride4].x - big); }
    workgroupBarrier();
    if (t == 0u) {
        var total = exp(0.0 - big);
        for (var sp = 0u; sp < p.splits; sp = sp + 1u) { total = total + part[rec0 + sp * hpg * stride4].y * wts[sp]; }
        tot[0] = total;
    }
    workgroupBarrier();

    let lane = t / 64u;
    let quad = t % 64u;
    for (var d0 = 0u; d0 < hd4; d0 = d0 + 64u) {
        let d4 = d0 + quad;
        let live = d4 < hd4;
        var acc = vec4<f32>(0.0);
        for (var sp = lane; sp < p.splits && live; sp = sp + 4u) {
            acc = acc + part[rec0 + sp * hpg * stride4 + 1u + d4] * wts[sp];
        }
        if (lane != 0u) { sred[(lane - 1u) * 64u + quad] = acc; }
        workgroupBarrier();
        if (lane == 0u && live) {
            out[(row * p.n_heads + h) * hd4 + d4] = (((acc + sred[quad]) + sred[64u + quad]) + sred[128u + quad]) / tot[0];
        }
        workgroupBarrier();
    }
}
