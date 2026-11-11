// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Decoder step's query and key head norms with rotary positions, the key and value rows stored into their caches, in one pass
// @how   64-thread workgroup per (sequence, head), 1 barrier: the norm of mirage_rms_rope per head row, the key rows and value copy writing the position cache
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The self-attention front of a decoder step after the causal taps: x holds the tapped q | k | v columns of each sequence
// (row stride x_stride). Workgroup r of sequence b normalises head r (r < heads: a query head, norm weight w[0..hd);
// else key head r - heads, weight w[hd..2 hd)) exactly as mirage_rms_rope does and rotates it by the position in `pos`.
// A query head is stored in q ([b * heads + r, hd]); a key head is stored into its row of the key cache
// ([b, maxt, kvh * hd] at pos[0]) and the same workgroup copies the matching value head (x column k_end + h * vd) into the value
// cache ([b, maxt, kvh * vd]). Three dispatches (two norm-rotations and a row write) become one.
// Contract: hd <= 64 and vd <= 64.
//
//   x: [batch, x_stride]   w: [2 hd]   pos: one word   q: [batch * heads, hd]   kcache, vcache: position caches
//   params: hd, vd, batch, heads, kvh, x_stride, maxt, eps (f32 bits), base (f32 bits). Dispatch: batch * (heads + kvh) workgroups of 64.

struct Params { hd: u32, vd: u32, batch: u32, heads: u32, kvh: u32, x_stride: u32, maxt: u32, eps: f32, base: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read>       pos: array<u32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(6) var<storage, read_write> vcache: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = wg.y * nwg.x + wg.x;
    if (id >= p.batch * (p.heads + p.kvh)) { return; }
    let t = li.x;
    let per = p.heads + p.kvh;
    let b = id / per;
    let r = id % per;
    let is_key = r >= p.heads;
    let h = select(r, r - p.heads, is_key);
    let k_end = p.heads * p.hd + p.kvh * p.hd;
    let base = b * p.x_stride + select(r * p.hd, p.heads * p.hd + h * p.hd, is_key);
    let wo = select(0u, p.hd, is_key);
    let obase = select((b * p.heads + h) * p.hd, ((b * p.maxt + pos[0]) * p.kvh + h) * p.hd, is_key);
    var acc = 0.0;
    if (t < p.hd) {
        let v = x[base + t];
        acc = v * v;
    }
    partial[t] = acc;
    workgroupBarrier();
    var ss = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        ss = ss + partial[i];
    }
    let inv = 1.0 / sqrt(ss / f32(p.hd) + p.eps);
    let half = p.hd / 2u;
    if (t < half) {
        let angle = f32(pos[0]) * exp(-log(p.base) * f32(2u * t) / f32(p.hd));
        let s = sin(angle);
        let co = cos(angle);
        let a = x[base + t] * inv * w[wo + t];
        let bv = x[base + t + half] * inv * w[wo + t + half];
        let lo = a * co - bv * s;
        let hi = bv * co + a * s;
        if (is_key) {
            kcache[obase + t] = lo;
            kcache[obase + t + half] = hi;
        } else {
            q[obase + t] = lo;
            q[obase + t + half] = hi;
        }
    }
    if (is_key && t < p.vd) {
        vcache[((b * p.maxt + pos[0]) * p.kvh + h) * p.vd + t] = x[b * p.x_stride + k_end + h * p.vd + t];
    }
}
