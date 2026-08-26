// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Channels-first L2/RMS normalization with a per-channel gain, FUSED - the normalisation runs in NCHW, with no permute either side
// @how   one thread per spatial position, two serial passes over the channel axis
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Channels-first L2 normalization with a learnable per-dim gain, FUSED - the
// NCHW twin of `l2norm_scale`, standing to it exactly as `layernorm2d` stands
// to `layernorm_rows`.
//
//   x, y : [N, C, HW]
//   g    : [C]
//   y[n,c,l] = x[n,c,l] * rsqrt(sum_k x[n,k,l]^2 + eps) * g[c]
// where the sum is over the CHANNEL axis at each spatial position, taken in
// ascending `c` so the fold order is fixed.
//
// Why this exists. The composed form is `nchw_nlc` -> `l2norm_scale` ->
// `nlc_nchw`, and it shipped because the middle stage sees its rows
// contiguous. But the two permutes are pure strided movement: `nchw_nlc`
// gathers `x[(n*C+ch)*HW + l]` with `ch` varying fastest, so a warp's lanes
// land `HW` floats apart and each fetched sector serves one useful float;
// `nlc_nchw` is the mirror image on the store side. The composition therefore
// pays that sector amplification TWICE to let the middle kernel avoid paying
// it once - the same reasoning `layernorm2d`'s header records, and the same
// answer.
//
// It also removes the middle kernel's duplicated arithmetic: `l2norm_scale`
// gives one thread each OUTPUT element and every one of a row's C threads
// redoes that row's whole sum of squares, so its op count scales as C per
// element. Here one thread owns a whole position and the sum is computed once.
//
// Two reads of x (sum, then apply) against the composition's
// read-write-read-write-read-write, all of them coalesced because consecutive
// lanes take consecutive `l`.
//
// One invocation per (n, hw). Barrier-free and array-free, so `backend-cpu`
// JITs it.

struct Params {
    N: u32,
    C: u32,
    HW: u32,
    eps: u32,   // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       g: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.N * p.HW) { return; }

    let n = gidx / p.HW;
    let hw = gidx % p.HW;
    let base = n * p.C * p.HW + hw;      // element (n, 0, hw); channel stride HW

    // Pass 1: sum of squares over the channel axis, ascending c - the same
    // fold order `l2norm_scale` uses over its contiguous row, which is what
    // makes this kernel bit-identical to the composed form.
    var s = 0.0;
    for (var c: u32 = 0u; c < p.C; c = c + 1u) {
        let v = x[base + c * p.HW];
        s = s + v * v;
    }
    let r = inverseSqrt(s + bitcast<f32>(p.eps));

    // Pass 2: scale.
    for (var c: u32 = 0u; c < p.C; c = c + 1u) {
        let i = base + c * p.HW;
        y[i] = x[i] * r * g[c];
    }
}
