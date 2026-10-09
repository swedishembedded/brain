// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  BatchNorm backward: accumulate the gamma and beta gradients from bn_dstats' packed sums
// @how   one thread per channel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// BatchNorm backward w.r.t. gamma AND beta, from sums `bn_dstats` already made.
// One invocation per channel c, accumulating into the (pre-zeroed) grads:
//   dgamma[c] += bp[5c+4]   (dxhat_sum = sum dy * xhat)
//   dbeta[c]  += bp[5c+3]   (dsum      = sum dy)
//
// `bn_dgamma` and `bn_dbeta` each walk the whole activation again to compute
// exactly these two sums, which `bn_dstats` has just produced for the input
// gradient - the same terms, in the same order. Reading them out of `bp` is two
// passes over the activations fewer, with the same result.

struct Params {
    C: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       bp:     array<f32>;
@group(0) @binding(2) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> dbeta:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let c = gid.y * (nwg.x * 64u) + gid.x;
    if (c >= p.C) { return; }
    dgamma[c] = dgamma[c] + bp[5u * c + 4u];
    dbeta[c] = dbeta[c] + bp[5u * c + 3u];
}
