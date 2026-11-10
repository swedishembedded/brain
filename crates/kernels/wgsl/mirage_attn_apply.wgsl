// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Key-parallel masked GQA attention, pass 3: out[b, q, h, i] = sum_key probs[b, q, h, key] * v[b, key, kv, i]
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// probs: [batch, nq, heads, nk]; v: [batch, nk, kvh, vd]; out: [batch, nq, heads, vd].
// `visible` 0xFFFFFFFF means pos[0] + 1. Params: batch, nq, heads, kvh, vd, nk, visible. Dispatch: batch*nq*heads*vd threads.

struct Params { batch: u32, nq: u32, heads: u32, kvh: u32, vd: u32, nk: u32, visible: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read>       pos: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.nq * p.heads * p.vd) { return; }
    let i = idx % p.vd;
    let row = idx / p.vd;
    let h = row % p.heads;
    let b = row / (p.heads * p.nq);
    let kv = h / (p.heads / p.kvh);
    var visible = p.visible;
    if (visible == 0xFFFFFFFFu) { visible = pos[0] + 1u; }
    // Four independent accumulators hide the load latency of a serial walk over the keys.
    let pb = row * p.nk;
    let vb = (b * p.nk * p.kvh + kv) * p.vd + i;
    let vs = p.kvh * p.vd;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    var s = 0u;
    for (; s + 4u <= visible; s = s + 4u) {
        a0 = a0 + probs[pb + s] * v[vb + s * vs];
        a1 = a1 + probs[pb + s + 1u] * v[vb + (s + 1u) * vs];
        a2 = a2 + probs[pb + s + 2u] * v[vb + (s + 2u) * vs];
        a3 = a3 + probs[pb + s + 3u] * v[vb + (s + 3u) * vs];
    }
    for (; s < visible; s = s + 1u) { a0 = a0 + probs[pb + s] * v[vb + s * vs]; }
    out[idx] = (a0 + a1) + (a2 + a3);
}
