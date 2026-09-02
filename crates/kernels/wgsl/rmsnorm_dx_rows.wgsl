// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm backward w.r.t. x, one WORKGROUP per row - the coalesced variant
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// RMSNorm backward w.r.t. x, one WORKGROUP per row - the coalesced twin of
// `rmsnorm_dx.wgsl`. Same math, same `Params` (`d_model`, `n_rows`), same 4
// storage bindings; only the row's own two reductions move from one thread
// walking the whole row twice to 64 threads splitting it once.
//
//   x, dy: [rows, d]   weight: [d]   dx: [rows, d]
//   dispatch: n_rows * 64 invocations (one workgroup per row)
//
// Forward: y_c = w_c*x_c*r, r = 1/sqrt(mean(x^2)+eps). With
// A = sum_c dY_c*w_c*x_c:  dX_i = r*w_i*dY_i - (r^3*x_i/d)*A.
//
// `ss = sum(x^2)` and `A = sum(dy*w*x)` are NOT sequentially dependent on
// each other - unlike LayerNorm's variance, which needs the mean before it
// can form `sum((x-mean)^2)` (see `layernorm_dx_rows.wgsl`'s shifted
// one-pass form), RMSNorm's own statistic is a plain sum of squares with
// nothing to subtract, so there is nothing to shift for cancellation either.
// Both partials accumulate in the SAME pass behind ONE barrier - the CPU
// JIT's single-top-level-barrier limit - same shape as
// `layernorm_dx_rows.wgsl`'s four-partial fold, just with two.
//
// `@workgroup_size(64)`, as everywhere in this family: at or below every
// device's queried `max_workgroup_size`, so only `workgroup_reductions`
// gates selection (see `backend_api::select::Op::RmsNorm`).

struct Params {
    d_model: u32,
    n_rows: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read>       dy:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:     array<f32>;

var<workgroup> p_ss: array<f32, 64>;
var<workgroup> p_a: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let n = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;

    var ss = 0.0;
    var a = 0.0;
    for (var c = t; c < d; c = c + 64u) {
        let xv = x[base + c];
        ss = ss + xv * xv;
        a = a + dy[base + c] * weight[c] * xv;
    }
    p_ss[t] = ss;
    p_a[t] = a;
    workgroupBarrier();
    var tss = 0.0;
    var ta = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        tss = tss + p_ss[i];
        ta = ta + p_a[i];
    }
    let r = inverseSqrt(tss / f32(d) + 1e-6);
    let coef = r * r * r * ta / f32(d);
    for (var c = t; c < d; c = c + 64u) {
        dx[base + c] = r * weight[c] * dy[base + c] - coef * x[base + c];
    }
}
