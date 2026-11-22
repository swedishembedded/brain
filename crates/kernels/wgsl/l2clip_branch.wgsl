// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-correctness tooling for models with non-smooth operations for its clients.
// If your team needs expertise in checking the backward of a clipped or routed network then you can procure our
// services by sending an email to info@swedishembedded.com.

// @what  The branch l2clip_scale takes per vector: 1.0 where the scaled vector is clipped, 0.0 where it passes
// @how   64-thread workgroup per vector, strided partial sums of a^2, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Records the decision of l2clip_scale.wgsl (a = x*s clipped when thr / (|a| + eps) < 1) so later forwards and
// backwards can replay it with their `pinned` parameter: a finite-difference check then measures the branch the
// analytic gradient differentiates, even where its step carries a vector across the threshold.
//
// x: [n_vec, d] where vector v belongs to head v % heads; branch: [n_vec]; params as l2clip_scale.wgsl (`pinned` is
// not read). With has_scale == 0 the `s` binding is never read (bind any buffer).
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
@group(0) @binding(3) var<storage, read_write> branch: array<f32>;

var<workgroup> part: array<f32, 64>;

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
            var a = x[v * p.d + c];
            if (p.has_scale != 0u) { a = a * s[(v % p.heads) * p.d + c]; }
            ss = ss + a * a;
        }
    }
    part[t] = ss;
    workgroupBarrier();
    var tot = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) { tot = tot + part[i]; }
    if (live && t == 0u) {
        branch[v] = select(0.0, 1.0, p.thr / (sqrt(tot) + p.eps) < 1.0);
    }
}
