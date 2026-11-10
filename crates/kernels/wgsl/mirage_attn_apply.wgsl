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
// Params: batch, nq, heads, kvh, vd, nk, visible. Dispatch: batch*nq*heads*vd threads.

struct Params { batch: u32, nq: u32, heads: u32, kvh: u32, vd: u32, nk: u32, visible: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

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
    var acc = 0.0;
    for (var s = 0u; s < p.visible; s = s + 1u) {
        acc = acc + probs[row * p.nk + s] * v[((b * p.nk + s) * p.kvh + kv) * p.vd + i];
    }
    out[idx] = acc;
}
