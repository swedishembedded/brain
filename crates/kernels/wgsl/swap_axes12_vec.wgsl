// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Swap axes 1 and 2 of a rank-4 [A0,A1,A2,D] tensor (a batched transpose of D-wide vectors) - model::timesfm3's sequence<->variate axis swap
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `[A0,A1,A2,D] -> [A0,A2,A1,D]`. Generalizes `nchw_nlc`/`nlc_nchw` (which
// swap exactly two axes around a fixed batch axis, D implicitly 1) to a
// trailing per-element vector width D - the shape TimesFM-3's variate
// attention needs: sequence attention leaves its input `[B,V,N,D]` (V-major,
// N-minor), variate attention needs `[B,N,V,D]` (N-major, V-minor) so each
// (batch, patch-position) group's V variates are contiguous. One thread per
// OUTPUT scalar:
//   dst[((i0*A2 + i2)*A1 + i1)*D + k] = src[((i0*A1 + i1)*A2 + i2)*D + k]
// Its own inverse: calling it again with A1/A2 swapped undoes it (the same
// bijection run the other direction), same as `nlc_nchw` is to `nchw_nlc`.

struct Params {
    a0: u32,
    a1: u32,
    a2: u32,
    d:  u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.a0 * p.a1 * p.a2 * p.d;
    if (gidx >= total) { return; }

    let k  = gidx % p.d;
    let t1 = gidx / p.d;
    let i1 = t1 % p.a1;
    let t2 = t1 / p.a1;
    let i2 = t2 % p.a2;
    let i0 = t2 / p.a2;

    let src_idx = ((i0 * p.a1 + i1) * p.a2 + i2) * p.d + k;
    dst[gidx] = src[src_idx];
}
