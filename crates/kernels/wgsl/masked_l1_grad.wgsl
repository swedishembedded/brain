// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Masked L1 gradient:  d_pred = sign(pred - tgt) * mask * scale.
//   pred, tgt, mask : [total]
//   d_pred : [total]  read_write
//
// `scale` is a bit-cast f32 in the uniform carrying the host-computed
// 1/(sum(mask) + 1e-8) normalizer — the same host-reduce split masked_l1.wgsl
// uses for the forward.
//
// sign(0) == 0 here (WGSL's sign, NOT Rust's f32::signum which returns +/-1 for
// +/-0.0): |d| has no derivative at d == 0, and 0 is the subgradient the
// reference implementations select.

struct Params {
    total: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<f32>;
@group(0) @binding(3) var<storage, read>       mask:   array<f32>;
@group(0) @binding(4) var<storage, read_write> d_pred: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    d_pred[idx] = sign(pred[idx] - tgt[idx]) * mask[idx] * p.scale;
}
