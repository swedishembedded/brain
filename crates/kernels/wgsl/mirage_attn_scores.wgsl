// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Key-parallel masked GQA attention, pass 1: scaled scores s[b, q, h, key] (masked or invisible keys get -1e30)
// @how   one thread per (image, query, head, key)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// q: [batch, nq, heads, hd]; k: [batch, nk, kvh, hd]; key_mask: [batch, nk]; scores: [batch, nq, heads, nk].
// Together with mirage_attn_softmax and mirage_attn_apply this is mirage_attn split so every key and output
// element has its own thread. Params: batch, nq, heads, kvh, hd, nk, visible, scale (f32 bits).
// Dispatch: batch*nq*heads*nk threads.

struct Params { batch: u32, nq: u32, heads: u32, kvh: u32, hd: u32, nk: u32, visible: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q: array<f32>;
@group(0) @binding(2) var<storage, read>       k: array<f32>;
@group(0) @binding(3) var<storage, read>       key_mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.nq * p.heads * p.nk) { return; }
    let s = idx % p.nk;
    let row = idx / p.nk;
    let h = row % p.heads;
    let b = row / (p.heads * p.nq);
    var r = -1.0e30;
    if (s < p.visible && key_mask[b * p.nk + s] != 0.0) {
        let kv = h / (p.heads / p.kvh);
        let kb = ((b * p.nk + s) * p.kvh + kv) * p.hd;
        var dot = 0.0;
        for (var i = 0u; i < p.hd; i = i + 1u) { dot = dot + q[row * p.hd + i] * k[kb + i]; }
        r = dot * p.scale;
    }
    scores[idx] = r;
}
