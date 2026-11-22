// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward of l2clip_scale w.r.t. x, plus the per-element scale partials
// @how   64-thread workgroup per vector, strided partial sums of a^2 and a.dy, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of l2clip_scale.wgsl. With a = x*s, n = |a|, r = thr / (n + eps), and g = dL/da:
//   r >= 1 (not clipped):  g = dy
//   r <  1 (clipped):      g = r * dy - (thr / ((n + eps)^2 * n)) * a * (a . dy)
//   dx = g * s            (g when has_scale == 0)
//   gs = g * x            (the partial of the scale gradient; reduced over vectors by l2clip_scale_ds)
// `gs` is written, and `s` read, only when has_scale != 0 (bind any buffer otherwise). With pinned != 0 the branch is
// `branch[v] != 0` instead of r < 1, as in the forward (l2clip_scale.wgsl); `branch` is read only then.
//
// x, dy, dx, gs: [n_vec, d]; s: [heads, d]; params as l2clip_scale.wgsl.
// Dispatch: n_vec * 64 invocations (one workgroup per vector).

struct Params {
    n_vec: u32,
    heads: u32,
    d: u32,
    has_scale: u32,
    thr: f32,
    eps: f32,
    pinned: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       s:  array<f32>;
@group(0) @binding(3) var<storage, read>       dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> gs: array<f32>;
@group(0) @binding(6) var<storage, read>       branch: array<f32>;

var<workgroup> p_ss: array<f32, 64>;
var<workgroup> p_ad: array<f32, 64>;

fn scale_at(v: u32, c: u32) -> f32 {
    if (p.has_scale != 0u) { return s[(v % p.heads) * p.d + c]; }
    return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let v = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = v < p.n_vec;
    let base = v * p.d;
    var ss = 0.0;
    var ad = 0.0;
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            let a = x[base + c] * scale_at(v, c);
            ss = ss + a * a;
            ad = ad + a * dy[base + c];
        }
    }
    p_ss[t] = ss;
    p_ad[t] = ad;
    workgroupBarrier();
    var tss = 0.0;
    var tad = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        tss = tss + p_ss[i];
        tad = tad + p_ad[i];
    }
    let n = sqrt(tss);
    let r = p.thr / (n + p.eps);
    var clipped = r < 1.0;
    if (p.pinned != 0u && live) { clipped = branch[v] != 0.0; }
    // Clipped means n + eps > thr > 0, so n > 0 unless thr <= eps (or a pinned branch at n == 0); guard that case.
    let proj = select(0.0, p.thr / ((n + p.eps) * (n + p.eps) * n) * tad, clipped && n > 0.0);
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            let sc = scale_at(v, c);
            let a = x[base + c] * sc;
            let g = select(dy[base + c], r * dy[base + c] - proj * a, clipped);
            dx[base + c] = g * sc;
            if (p.has_scale != 0u) { gs[base + c] = g * x[base + c]; }
        }
    }
}
