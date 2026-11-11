// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Head norm and rotary position in one pass: out = rope(rmsnorm(x) * w) at the position held in a buffer, one workgroup per head row
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// The decoder step normalises each query and key head and rotates it by the step's position; as two dispatches each
// is latency bound and the second re-reads what the first just wrote. Here the row's sum of squares is reduced
// (strided partials, ONE barrier, every thread folds the 64 partials as in rmsnorm_rows), then the thread that owns
// pair i rotates the normalised (x[i], x[i + hd/2]) and stores both halves.
//
//   x : [rows, hd]   out: [rows, hd]   w: [hd] (1 + scale)   pos: one word
//   angle_i = pos * base^(-2i / hd), the pairing of mirage_rope_pos (i, i + hd/2).
//   params: hd, rows, eps (f32 bits), base (f32 bits), groups, x_stride, x_off, maxt. Dispatch: rows * 64 invocations (one workgroup per row).
//
// Row r of a decoder step is head r % groups of sequence r / groups. With groups == 0 (the plain form) rows are contiguous
// in x and out. With groups > 0 row r is read at (r / groups) * x_stride + x_off + (r % groups) * hd, so the heads of a fused
// projection are normalised where they lie; and with maxt > 0 the rotated row is stored straight into a position cache laid
// out [sequence, maxt, groups * hd] at the position in `pos`, where a separate row-write dispatch would have copied it.

struct Params {
    hd: u32,
    rows: u32,
    eps: f32,
    base: f32,
    groups: u32,
    x_stride: u32,
    x_off: u32,
    maxt: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read>       pos: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    // `let`s only: the CPU interpreter re-initialises a `var` read across the barrier.
    let g = max(p.groups, 1u);
    let strided = p.groups > 0u;
    let base = select(row * p.hd, (row / g) * p.x_stride + p.x_off + (row % g) * p.hd, strided);
    let obase = select(base, select(row * p.hd, ((row / g) * p.maxt + pos[0]) * (g * p.hd) + (row % g) * p.hd, p.maxt > 0u), strided);
    var acc = 0.0;
    for (var c = t; c < p.hd; c = c + 64u) {
        let v = x[base + c];
        acc = acc + v * v;
    }
    partial[t] = acc;
    workgroupBarrier();
    var ss = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        ss = ss + partial[i];
    }
    let inv = 1.0 / sqrt(ss / f32(p.hd) + p.eps);
    let half = p.hd / 2u;
    for (var i = t; i < half; i = i + 64u) {
        let angle = f32(pos[0]) * exp(-log(p.base) * f32(2u * i) / f32(p.hd));
        let s = sin(angle);
        let co = cos(angle);
        let a = x[base + i] * inv * w[i];
        let b = x[base + i + half] * inv * w[i + half];
        out[obase + i] = a * co - b * s;
        out[obase + i + half] = b * co + a * s;
    }
}
