// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Mean-squared-error gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Mean-squared-error gradient w.r.t. the predictions (analogue of ce_grad):
//   d_pred[i] = 2 * (pred[i] - target[i]) / n
// the exact derivative of  L = mean_i (pred[i] - target[i])^2  =
// (1/n) * sum_i (pred[i] - target[i])^2  (the value computed by `mse_value`).
// One invocation per element; `n` is the total element count.

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:    array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:     array<f32>;
@group(0) @binding(3) var<storage, read_write> d_pred:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let i = gidx;
    if (i >= p.n) { return; }
    d_pred[i] = 2.0 * (pred[i] - tgt[i]) / f32(p.n);
}
