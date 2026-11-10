// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Orthonormal natural-order Walsh transform of every 128-lane group of f32 activation rows, zero padded to whole groups - the activation half of a CQ product
// @how   one thread per (row, group): butterfly in a private 128-lane array, no barriers
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// y = W x for a rotated-codebook weight is sum_g norm_g * (codebook[idx_g] . (H x_g)) with H the symmetric
// orthonormal Walsh matrix in NATURAL (Hadamard) order, so every activation group is rotated once and then shared by
// all output rows of every product that reads it (`matmul_cq_gemv`, `matmul_cq_gemm`). The sequency-ordered
// transform a model may fold into its own matrices is a different permutation of the same rows and is not this one.
//
//   params : m, k, kpad, xr0   rows to rotate, logical width, width padded to whole groups, first row of x
//   x      : [xr0 + m, k] f32
//   out    : [m, kpad] f32     lanes at or past k are rotated zeros

struct Params {
    m: u32,
    k: u32,
    kpad: u32,
    xr0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const GROUP: u32 = 128u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let groups = p.kpad / GROUP;
    if (idx >= p.m * groups) { return; }
    let row = idx / groups;
    let base = (idx % groups) * GROUP;
    var lane: array<f32, 128>;
    for (var j = 0u; j < GROUP; j = j + 1u) {
        var v = 0.0;
        if (base + j < p.k) { v = x[(p.xr0 + row) * p.k + base + j]; }
        lane[j] = v;
    }
    for (var h = 1u; h < GROUP; h = h * 2u) {
        for (var i = 0u; i < GROUP / 2u; i = i + 1u) {
            let a = (i / h) * 2u * h + (i % h);
            let l = lane[a];
            let r = lane[a + h];
            lane[a] = l + r;
            lane[a + h] = l - r;
        }
    }
    for (var j = 0u; j < GROUP; j = j + 1u) {
        out[row * p.kpad + base + j] = lane[j] * 0.08838834764831845;
    }
}
