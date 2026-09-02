// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm fused with dynamic per-row int8 activation quantization -
//        the normalized fp32 row is never written to memory at all
// @how   64-thread workgroup tile, 3 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// RMSNorm fused with its own immediately-following int8 activation quant
// (`max_abs_row` -> `quant_pack`) into ONE dispatch: `rmsnorm_rows` -> `max_abs_row`
// -> `quant_pack` writes the normalized row to memory once and reads it back
// TWICE more just to throw it away - on an all-int8-weight engine (`model::
// ops::Weight::I8`'s linear reads only the packed activation, never the fp32
// one) that fp32 write has no OTHER reader, so it is pure waste, not merely
// redundant. This kernel produces exactly the same two outputs the split pair
// would (`sx[row]`, `xq[row, :]`) without ever materialising the fp32
// intermediate:
//
//   x  : [rows, d]   w: [d]   (RMSNorm inputs, unchanged)
//   sx : [rows]      (per-row int8 scale - max_abs_row's own output)
//   xq : [rows, d/4] (packed int8, 4 lanes/word LE - quant_pack's own output)
//   params: d, rows, eps
//
// Three cooperative stages, each ending in the SAME shared-memory reduction
// shape `softmax_rows.wgsl` already uses (write partial -> barrier -> serial
// fold -> barrier before the next stage reuses the array):
//   1. sum of squares over `x`'s row -> `inv` (bit-identical to `rmsnorm_rows`'s
//      own first stage - same operand order, same reduction order).
//   2. the normalized value `v = x[c] * inv * w[c]` (bit-identical to what
//      `rmsnorm_rows` would have WRITTEN, computed with the identical
//      expression and operand order) is used ONLY to fold a row-wide abs-max,
//      producing `sx[row]` - never written to a `v`-sized buffer.
//   3. `v` is recomputed once more (same expression, same bits - IEEE754 is
//      deterministic, so this is not a second, possibly-different value) to
//      quantize+pack it, exactly `quant_pack.wgsl`'s own arithmetic.
//
// No `var<function>` array sized off the runtime `d` - the anti-pattern
// `qknorm_rope_fused.wgsl` already names - recomputing `v` from `x`/`inv`/`w`
// a second time is deliberate: it trades one more pass of cheap (cache-warm)
// reads of `x`/`w` for never allocating a per-thread register array sized by
// a value only known at dispatch time, and for never touching a `d`- or
// `d/4`-wide fp32/int8 buffer more than once each.
//
// GPU-only by construction (3 barriers, like `softmax_rows.wgsl`): select it
// behind `DeviceCaps::workgroup_reductions` on an all-int8-weight engine; a
// device without it, or an engine that keeps any fp32 linear, keeps the
// original `rmsnorm{,_rows}` -> `max_abs_row` -> `quant_pack` sequence, whose
// fp32 output other readers may still need.

struct Params {
    d: u32,
    rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       w:  array<f32>;
@group(0) @binding(3) var<storage, read_write> sx: array<f32>;
@group(0) @binding(4) var<storage, read_write> xq: array<u32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.d;

    // Stage 1: sum of squares -> inv (bit-identical to `rmsnorm_rows.wgsl`).
    var acc = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        let v = x[base + c];
        acc = acc + v * v;
    }
    partial[t] = acc;
    workgroupBarrier();
    var ss = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        ss = ss + partial[i];
    }
    let inv = 1.0 / sqrt(ss / f32(p.d) + p.eps);
    workgroupBarrier(); // all reads of `partial` above must finish before stage 2 overwrites it.

    // Stage 2: fold the row's abs-max over the (unwritten) normalized values.
    var amax = 0.0;
    for (var c = t; c < p.d; c = c + 64u) {
        let v = x[base + c] * inv * w[c];
        amax = max(amax, abs(v));
    }
    partial[t] = amax;
    workgroupBarrier();
    var rowmax = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        rowmax = max(rowmax, partial[i]);
    }
    let scale = max(rowmax, 1e-8) / 127.0;
    if (t == 0u) {
        sx[row] = scale;
    }
    let sinv = 1.0 / scale;

    // Stage 3: recompute `v` once more and pack it - `quant_pack.wgsl`'s own
    // arithmetic, one thread per output u32 word (`d/4` words per row).
    let kg = p.d / 4u;
    for (var g = t; g < kg; g = g + 64u) {
        var word: u32 = 0u;
        for (var b = 0u; b < 4u; b = b + 1u) {
            let c = g * 4u + b;
            let v = x[base + c] * inv * w[c];
            let q = clamp(round(v * sinv), -127.0, 127.0);
            let byte = u32(i32(q) & 0xff);
            word = word | (byte << (8u * b));
        }
        xq[row * kg + g] = word;
    }
}
