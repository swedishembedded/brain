// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Mean-squared-error loss value (the Regression-head analogue of ce_value).
//   out[i] = (pred[i] - target[i])^2 / n
// where `n` = total number of elements. The host sums out[] to obtain the
// mean squared error  mean_i (pred[i] - target[i])^2  (matching the gradient in
// `mse_grad`, which divides by the same `n`). One invocation per element; the
// per-element division by `n` keeps the host-side reduction a plain sum, exactly
// like the cross-entropy value kernel.

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:    array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let i = gidx;
    if (i >= p.n) { return; }
    let d = pred[i] - tgt[i];
    out[i] = (d * d) / f32(p.n);
}
