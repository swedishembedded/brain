// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-vector L2 clip with an optional learned per-(head,dim) scale: a = x*s, y = a * min(1, thr / (|a| + eps))
// @how   64-thread workgroup per vector, strided partial sums of a^2, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Scales each head vector down to L2 norm `thr` when it is longer and leaves shorter vectors unchanged (the
// query/key bound of a cosine-like attention, the clip of a relative-state vector). Unlike l2norm_scale the vector is
// not normalised: only the excess is removed, and the epsilon is added to the norm, not under the root.
//
//   a = x * s                       (s: [heads, d]; absent when has_scale == 0)
//   y = a * min(1, thr / (|a| + eps))
//
// x, y: [n_vec, d] where vector v belongs to head v % heads; params: n_vec, heads, d, has_scale, thr, eps (f32 bits),
// pinned. With has_scale == 0 the `s` binding is never read (bind any buffer). Backward: l2clip_scale_dx.wgsl (dx and
// the per-element scale partials) and l2clip_scale_ds.wgsl (the reduction over vectors).
//
// Pinned branches (pinned != 0): the clip is not differentiable where |a| + eps == thr, so a finite-difference check
// whose step carries a vector across the threshold measures neither branch. `branch[v]` (1.0 clipped, 0.0 not, as
// l2clip_branch.wgsl records them) then chooses each vector's branch instead of the comparison:
//   y = a * (branch[v] != 0 ? thr / (|a| + eps) : 1)
// which equals the unpinned clip wherever the recorded branch is the one the comparison takes, and continues that
// branch smoothly past the threshold. With pinned == 0 the `branch` binding is never read (bind any buffer).
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
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       s: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read>       branch: array<f32>;

var<workgroup> part: array<f32, 64>;

fn scaled(v: u32, c: u32) -> f32 {
    var a = x[v * p.d + c];
    if (p.has_scale != 0u) { a = a * s[(v % p.heads) * p.d + c]; }
    return a;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let v = wg.y * nwg.x + wg.x;
    let t = li.x;
    let live = v < p.n_vec;
    var ss = 0.0;
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            let a = scaled(v, c);
            ss = ss + a * a;
        }
    }
    part[t] = ss;
    workgroupBarrier();
    var tot = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) { tot = tot + part[i]; }
    let r = p.thr / (sqrt(tot) + p.eps);
    var k = min(1.0, r);
    if (p.pinned != 0u && live) { k = select(1.0, r, branch[v] != 0.0); }
    if (live) {
        for (var c = t; c < p.d; c = c + 64u) {
            y[v * p.d + c] = scaled(v, c) * k;
        }
    }
}
