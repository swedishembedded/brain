// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Masked grouped-query attention with an online softmax: self (causal prefix), encoder (cell mask) and cross (memory mask)
// @how   one thread per (image, query, head), serial over keys, running max
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// q: [batch, nq, heads, hd]; k: [batch, nk, kvh, hd]; v: [batch, nk, kvh, vd]; key_mask: [batch, nk]
// (1 live key, 0 padding); out: [batch, nq, heads, vd]. Keys at index >= visible are ignored. Query head h
// reads kv head h / (heads / kvh). A row with no live key yields zeros. vd is at most 128.
// Params: batch, nq, heads, kvh, hd, vd, nk, visible, scale (f32 bits). Dispatch: batch*nq*heads threads.

struct Params { batch: u32, nq: u32, heads: u32, kvh: u32, hd: u32, vd: u32, nk: u32, visible: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q: array<f32>;
@group(0) @binding(2) var<storage, read>       k: array<f32>;
@group(0) @binding(3) var<storage, read>       v: array<f32>;
@group(0) @binding(4) var<storage, read>       key_mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.nq * p.heads) { return; }
    let h = idx % p.heads;
    let b = idx / (p.heads * p.nq);
    let kv = h / (p.heads / p.kvh);
    let qb = idx * p.hd;
    var acc: array<f32, 128>;
    for (var i = 0u; i < p.vd; i = i + 1u) { acc[i] = 0.0; }
    var mx = -1.0e30;
    var den = 0.0;
    for (var s = 0u; s < p.visible; s = s + 1u) {
        if (key_mask[b * p.nk + s] == 0.0) { continue; }
        let kb = ((b * p.nk + s) * p.kvh + kv) * p.hd;
        var dot = 0.0;
        for (var i = 0u; i < p.hd; i = i + 1u) { dot = dot + q[qb + i] * k[kb + i]; }
        let sc = dot * p.scale;
        let nm = max(mx, sc);
        let corr = exp(mx - nm);
        let e = exp(sc - nm);
        den = den * corr + e;
        let vb = ((b * p.nk + s) * p.kvh + kv) * p.vd;
        for (var i = 0u; i < p.vd; i = i + 1u) { acc[i] = acc[i] * corr + e * v[vb + i]; }
        mx = nm;
    }
    let ob = idx * p.vd;
    for (var i = 0u; i < p.vd; i = i + 1u) {
        var r = 0.0;
        if (den > 0.0) { r = acc[i] / den; }
        out[ob + i] = r;
    }
}
