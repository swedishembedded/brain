// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Single-query attention of one (sequence, head) in one workgroup, scores to gated output, no split merge
// @how   512-thread workgroup: key-parallel scores into workgroup memory, tree max, exp and tree sum, then column-quad groups walk the keys with eight values in flight
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// The decoder's attention for a batch of one is 8 query rows over a few hundred keys: launch and dependent-load
// latency, not bandwidth. mirage_attn_decode splits the keys of a row over workgroups so the card fills, and then
// needs mirage_attn_merge and a separate sigmoid gate: three dependent launches. This kernel keeps a whole row in one
// workgroup of 512 threads, which is enough threads to hide the loads (the keys are scored one per thread, the values
// summed by ten groups of columns each holding eight loads in flight) and removes both extra launches:
//   phase 1: thread t scores keys t, t + 512, ... (masked and not-yet-written keys score -1e30), tree max;
//   phase 2: p_s = exp(score_s - max) in place, tree sum;
//   phase 3: group g = t / (vd / 4) sums p_s * v[s, 4 (t mod vd/4) ..+4] over its keys s = g, g + G, ...; the groups fold into the row,
//            divided by the sum (zero when no key is live) and multiplied by sigmoid(gate): att / (1 + exp(-gate)).
// The result equals mirage_attn_decode + mirage_attn_merge + mirage_sigmoid_mul up to the order of the sums.
//
//   q: [batch * heads, hd]   k: [batch, nk, kvh, hd]   v: [batch, nk, kvh, vd]   key_mask: [batch, nk]
//   pos: one word; visible == 0xFFFFFFFF means pos[0] + 1 keys.
//   gate: [batch, gate_stride] read at gate_off + head * vd + column   o: [batch * heads, vd]
// q, k, v and o are read as vec4 rows (one load per four values: the scalar form of this kernel was instruction bound,
// not memory bound). Contract: hd and vd multiples of 4 and at most 64, min(nk, visible) <= 1024. Dispatch: batch * heads workgroups.

struct Params { batch: u32, heads: u32, kvh: u32, hd: u32, vd: u32, nk: u32, visible: u32, scale: f32, gate_stride: u32, gate_off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       v: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       key_mask: array<f32>;
@group(0) @binding(5) var<storage, read>       pos: array<u32>;
@group(0) @binding(6) var<storage, read>       gate: array<f32>;
@group(0) @binding(7) var<storage, read_write> o: array<vec4<f32>>;

var<workgroup> qs: array<vec4<f32>, 16>;
var<workgroup> sc: array<f32, 1024>;
var<workgroup> red: array<f32, 512>;
var<workgroup> part: array<vec4<f32>, 384>;

@compute @workgroup_size(512)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p.batch * p.heads) { return; }
    let t = li.x;
    let h = row % p.heads;
    let b = row / p.heads;
    let kv = h / (p.heads / p.kvh);
    let hd4 = p.hd / 4u;
    let vd4 = p.vd / 4u;
    let visible = select(p.visible, pos[0] + 1u, p.visible == 0xFFFFFFFFu);
    let n = min(visible, p.nk);
    if (t < hd4) { qs[t] = q[row * hd4 + t]; }
    workgroupBarrier();
    var m = -1.0e30;
    for (var s = t; s < n; s = s + 512u) {
        var r = -1.0e30;
        if (key_mask[b * p.nk + s] != 0.0) {
            let kb = ((b * p.nk + s) * p.kvh + kv) * hd4;
            var acc4 = vec4<f32>(0.0);
            var d = 0u;
            for (; d + 4u <= hd4; d = d + 4u) {
                let k0 = k[kb + d];
                let k1 = k[kb + d + 1u];
                let k2 = k[kb + d + 2u];
                let k3 = k[kb + d + 3u];
                acc4 = acc4 + qs[d] * k0 + qs[d + 1u] * k1 + qs[d + 2u] * k2 + qs[d + 3u] * k3;
            }
            for (; d < hd4; d = d + 1u) { acc4 = acc4 + qs[d] * k[kb + d]; }
            r = (acc4.x + acc4.y + acc4.z + acc4.w) * p.scale;
        }
        sc[s] = r;
        m = max(m, r);
    }
    red[t] = m;
    workgroupBarrier();
    for (var st = 256u; st > 8u; st = st >> 1u) {
        if (t < st) { red[t] = max(red[t], red[t + st]); }
        workgroupBarrier();
    }
    var gm = red[0];
    for (var i = 1u; i < 16u; i = i + 1u) { gm = max(gm, red[i]); }
    workgroupBarrier();
    var l = 0.0;
    if (gm > -1.0e29) {
        for (var s = t; s < n; s = s + 512u) {
            let e = exp(sc[s] - gm);
            sc[s] = e;
            l = l + e;
        }
    }
    red[t] = l;
    workgroupBarrier();
    for (var st = 256u; st > 8u; st = st >> 1u) {
        if (t < st) { red[t] = red[t] + red[t + st]; }
        workgroupBarrier();
    }
    var total = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) { total = total + red[i]; }
    let groups = min(512u / vd4, 24u);
    let g = t / vd4;
    let col = t % vd4;
    if (g < groups) {
        var acc = vec4<f32>(0.0);
        if (gm > -1.0e29) {
            let stride = groups;
            var s = g;
            for (; s + 7u * stride < n; s = s + 8u * stride) {
                let vb = ((b * p.nk + s) * p.kvh + kv) * vd4 + col;
                let step = stride * p.kvh * vd4;
                let v0 = v[vb];
                let v1 = v[vb + step];
                let v2 = v[vb + 2u * step];
                let v3 = v[vb + 3u * step];
                let v4 = v[vb + 4u * step];
                let v5 = v[vb + 5u * step];
                let v6 = v[vb + 6u * step];
                let v7 = v[vb + 7u * step];
                acc = acc + sc[s] * v0;
                acc = acc + sc[s + stride] * v1;
                acc = acc + sc[s + 2u * stride] * v2;
                acc = acc + sc[s + 3u * stride] * v3;
                acc = acc + sc[s + 4u * stride] * v4;
                acc = acc + sc[s + 5u * stride] * v5;
                acc = acc + sc[s + 6u * stride] * v6;
                acc = acc + sc[s + 7u * stride] * v7;
            }
            for (; s < n; s = s + stride) {
                acc = acc + sc[s] * v[((b * p.nk + s) * p.kvh + kv) * vd4 + col];
            }
        }
        part[g * 16u + col] = acc;
    }
    workgroupBarrier();
    if (t < vd4) {
        var sum = vec4<f32>(0.0);
        for (var i = 0u; i < groups; i = i + 1u) { sum = sum + part[i * 16u + t]; }
        var r = vec4<f32>(0.0);
        if (total > 0.0) { r = sum / total; }
        let gb = b * p.gate_stride + p.gate_off + h * p.vd + 4u * t;
        let gt = vec4<f32>(gate[gb], gate[gb + 1u], gate[gb + 2u], gate[gb + 3u]);
        o[row * vd4 + t] = r / (vec4<f32>(1.0) + exp(-gt));
    }
}
