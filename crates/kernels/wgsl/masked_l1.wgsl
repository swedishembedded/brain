// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Masked L1, per element:  out = /pred - tgt/ * mask
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Masked L1, per element:  out = |pred - tgt| * mask.
//   pred, tgt, mask : [total]
//   out : [total]  read_write
//
// The SSI term of ZipDepth's loss is  sum(|d|*m) / (sum(m) + 1e-8). This emits
// the per-element numerator terms; the HOST sums them, exactly as brain already
// does for every global reduction (mse_value.wgsl: "The host sums out[] to
// obtain the mean"; gradnorm_sq.wgsl). The denominator is likewise a host scalar:
// whoever builds the mask knows its sum, so there is no reason to reduce it on
// device.
//
// The mask is all-ones on the training path — dense proxy/synthetic depth
// supervises every pixel. It exists for SPARSE ground truth (e.g. projected
// LiDAR), where it must be passed explicitly or the loss's median/MAD
// normalization is computed over the zeros and corrupts.
//
// (Four storage bindings: the "<=4 buffers" line in AGENTS.md is already
// contradicted by shipped kernels — router_bwd, mla_scores and layernorm_dgamma
// all bind 5 — and WebGPU's downlevel guarantee is 8.)

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:   array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<f32>;
@group(0) @binding(3) var<storage, read>       mask:   array<f32>;
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    out[idx] = abs(pred[idx] - tgt[idx]) * mask[idx];
}
