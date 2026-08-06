// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  LayerNorm backward w.r.t. x, one WORKGROUP per row — the coalesced variant
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// LayerNorm backward w.r.t. x, one WORKGROUP per row — the coalesced variant.
//
//   x, dy: [rows, d]   gamma: [d]   dx: [rows, d]
//   params: d_model, n_rows, eps (f32 bits)   dispatch: n_rows * 64 invocations
//
// Same math as `layernorm_dx.wgsl`: with xhat = (x-mean)*inv and g = dy*gamma,
//   dx[c] = inv * ( g[c] - mean_k(g) - xhat[c] * mean_k(g*xhat) )
// and mean/inv are recomputed from x (keeps the kernel at 4 storage buffers).
//
// The reference walks the row FOUR times from a single thread, so it is the
// worst offender of the one-thread-per-row family — every one of those passes
// is uncoalesced. Here 64 threads split the row.
//
// The four reductions the row needs look sequentially dependent (mean and inv
// feed sum(g*xhat)), which would want two barriers — and the CPU JIT splits at
// exactly one. They are not: with K = x[row, 0] as the shift,
//
//     S1 = sum(x-K)      S2 = sum((x-K)^2)      S3 = sum(g)      S4 = sum(g*(x-K))
//     moff = S1/d        var  = S2/d - moff^2   inv = rsqrt(var+eps)
//     mean_k(g)      = S3/d
//     mean_k(g*xhat) = inv * (S4 - moff*S3) / d      [ since xhat = ((x-K)-moff)*inv ]
//
// so ALL FOUR partial sums accumulate in one pass behind one barrier, and the
// second pass only applies them. The shifted frame is also what keeps
// `S2/d - moff^2` and `S4 - moff*S3` free of catastrophic cancellation —
// see `layernorm_rows.wgsl`.
//
// @workgroup_size(64), as everywhere in this family: at or below every
// device's queried `max_workgroup_size`, so only `workgroup_reductions` gates
// selection.

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       dy:    array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:    array<f32>;

var<workgroup> p1: array<f32, 64>;
var<workgroup> p2: array<f32, 64>;
var<workgroup> p3: array<f32, 64>;
var<workgroup> p4: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;
    let df = f32(d);

    let k = x[base];
    var s1 = 0.0;
    var s2 = 0.0;
    var s3 = 0.0;
    var s4 = 0.0;
    for (var c = t; c < d; c = c + 64u) {
        let v = x[base + c] - k;
        let g = dy[base + c] * gamma[c];
        s1 = s1 + v;
        s2 = s2 + v * v;
        s3 = s3 + g;
        s4 = s4 + g * v;
    }
    p1[t] = s1;
    p2[t] = s2;
    p3[t] = s3;
    p4[t] = s4;
    workgroupBarrier();
    var t1 = 0.0;
    var t2 = 0.0;
    var t3 = 0.0;
    var t4 = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        t1 = t1 + p1[i];
        t2 = t2 + p2[i];
        t3 = t3 + p3[i];
        t4 = t4 + p4[i];
    }
    let moff = t1 / df;
    let va = max(t2 / df - moff * moff, 0.0);
    let inv = inverseSqrt(va + p.eps);
    let mg = t3 / df;
    let mgx = inv * (t4 - moff * t3) / df;
    for (var c = t; c < d; c = c + 64u) {
        let g = dy[base + c] * gamma[c];
        let xhat = ((x[base + c] - k) - moff) * inv;
        dx[base + c] = inv * (g - mg - xhat * mgx);
    }
}
