// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Gated DeltaNet's whole intra-chunk UT transform T = (I - attn0)^-1 in ONE dispatch: forward substitution over every row, plus the identity
// @how   one 64-thread workgroup per [c,c] matrix, triangular tiles in workgroup memory, 2 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Replaces `gdn_ut_step.wgsl` dispatched once per row index (`c - 1` launches,
// each a few microseconds of work behind a launch that costs more) followed by
// `gdn_add_identity.wgsl`, with ONE workgroup per `[c, c]` matrix that walks
// the rows itself. At the real chunk of 64 that is 63 launches fewer per
// Gated DeltaNet layer per prefill round - on a 64-layer hybrid model most of
// the host launch work of a round - and the transform itself drops from a
// chain of dependent dispatches to one workgroup-local loop.
//
// For row `i`, column `j < i` (the recurrence `gdn_ut_step.wgsl` derives):
//   T[i,j] = attn0[i,j] + sum_{k=j+1}^{i-1} attn0[i,k] * T[k,j]
// `T[i,i] = 1` (the `+= I` of `gdn_add_identity.wgsl`) and everything above the
// diagonal is 0. Thread `j` owns column `j`: it sums `k` ASCENDING from its own
// initial `attn0[i,j]`, exactly the operation order of the kernel this
// replaces, so the result is BIT-IDENTICAL to the loop on any backend that
// rounds the same way (`crates/model/tests/gdn_ut_fwd.rs` holds that to
// zero ulps).
//
// Only the strictly-lower triangle ever exists in workgroup memory: two
// packed triangles (`attn0` and `T`), row `i` starting at `i*(i-1)/2` and
// holding `i` entries - 2016 floats each at `c = 64`, 16128 bytes together,
// inside the 16 KiB workgroup-storage limit every WebGPU device guarantees.
// Row `i` of `T` depends only on rows `< i` of `T`, finalised before the
// barrier that ends the previous iteration, and `attn0`'s triangle is read
// only, so the two buffers keep the race-freedom argument of
// `gdn_ut_step.wgsl`'s header.
//
// Contract: params `[bhc, c_len]`, `c_len <= 64`, dispatch `bhc * 64` threads
// (one workgroup per matrix). The kernel writes the WHOLE `[bhc, c, c]` output
// - the caller does not clear it.

struct Params { bhc: u32, c_len: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       attn0: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_mat: array<f32>;

var<workgroup> a_tri: array<f32, 2016>;
var<workgroup> t_tri: array<f32, 2016>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    // Uniform across the workgroup, so returning before a barrier is legal.
    if (row >= p.bhc) { return; }
    let c = p.c_len;
    let j = li.x;
    let base = row * c * c;

    // Stage the strictly-lower triangle of attn0: row `i` is `i` contiguous
    // words, read coalesced across `j`.
    for (var i = 1u; i < c; i = i + 1u) {
        if (j < i) { a_tri[i * (i - 1u) / 2u + j] = attn0[base + i * c + j]; }
    }
    workgroupBarrier();

    for (var i = 1u; i < c; i = i + 1u) {
        let r = i * (i - 1u) / 2u;
        if (j < i) {
            var acc = a_tri[r + j];
            for (var k = j + 1u; k < i; k = k + 1u) {
                acc = acc + a_tri[r + k] * t_tri[k * (k - 1u) / 2u + j];
            }
            t_tri[r + j] = acc;
        }
        workgroupBarrier();
    }

    // Whole output: strictly lower from the triangle, unit diagonal, zero above.
    if (j < c) {
        for (var i = 0u; i < c; i = i + 1u) {
            var v = 0.0;
            if (j < i) { v = t_tri[i * (i - 1u) / 2u + j]; }
            if (j == i) { v = 1.0; }
            t_mat[base + i * c + j] = v;
        }
    }
}
