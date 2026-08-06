// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-parameter sum of squares of its gradient, as a COOPERATIVE tree reduction
// @how   64-thread workgroup tile, 2 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
//
// Per-parameter sum of squares of its gradient, as a COOPERATIVE tree reduction:
// `n_wg` workgroups of 64 threads split one gradient buffer, each writing one
// partial into `parts[out_off + wg]`. `clip_coef_wg` folds every tensor's
// partials into the global clip coefficient in one small second pass, so the
// whole grad-norm is two dispatch *stages*, not a serial walk per tensor.
//
// The kernel this replaces (`gradnorm_sq.wgsl`) runs the ENTIRE tensor on ONE
// thread — `if (gidx != 0u) { return; }` and a dispatch of 1 invocation. On a
// GPT-2-small-ish 120 M-param model that is 82-87 % of all GPU training time
// (measured: 30 133 ms of 34 545 ms over 5 `brain gpt train` steps), because a
// 38.6 M-element embedding gradient is 38.6 M dependent scalar loads on one
// lane of a 3840-core card. Same bug class as `gn_stats` and `rmsnorm`, one
// level up in the optimiser — see `docs/performance/overview.md`.
//
// Coalescing: the loop is grid-strided over the WHOLE dispatch
// (`i += n_wg * 64`), so at every step the 64 lanes of a workgroup read 64
// consecutive words and adjacent workgroups read adjacent 256-byte spans —
// every fetched sector is fully used. (A chunk-per-workgroup split would also
// be coalesced within a workgroup; grid-stride additionally keeps the tail
// balanced when `numel` is not a multiple of `n_wg * 64`.)
//
// ONE top-level `workgroupBarrier()`, which is the CPU JIT's limit (it splits a
// kernel body at exactly one barrier). No atomics, no subgroups — the cross-
// workgroup combine is the second dispatch, by construction.
//
// Dispatch: n_wg * 64 invocations. `n_wg` comes from `paramstore::gradnorm_parts`
// (ceil(numel/8192), capped at 512) and is passed in rather than derived from
// `num_workgroups` so a backend that pads the grid cannot write out of range.

struct Params {
    numel: u32,
    out_off: u32,  // this tensor's first slot in `parts`
    n_wg: u32,     // workgroups this dispatch really uses
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       grad:  array<f32>;
@group(0) @binding(2) var<storage, read_write> parts: array<f32>;

var<workgroup> psum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let w = wg.y * nwg.x + wg.x;
    let t = li.x;
    // Uniform across the workgroup, so returning before the barrier is legal.
    if (w >= p.n_wg) { return; }

    let stride = p.n_wg * 64u;
    var acc = 0.0;
    for (var i = w * 64u + t; i < p.numel; i = i + stride) {
        let g = grad[i];
        acc = acc + g * g;
    }
    psum[t] = acc;
    workgroupBarrier();
    // One thread folds the 64 partials (64 adds — cheaper than a second
    // barrier, which the CPU JIT could not compile anyway).
    if (t == 0u) {
        var s = 0.0;
        for (var k = 0u; k < 64u; k = k + 1u) {
            s = s + psum[k];
        }
        parts[p.out_off + w] = s;
    }
}
