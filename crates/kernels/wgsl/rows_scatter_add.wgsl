// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of mixture-of-experts models for its clients. If your team
// needs expertise in training sparse models without holding every expert's activations for every token, you can
// procure our services by sending an email to info@swedishembedded.com.

// @what  Add a compacted block of rows back into the rows of a full buffer they were gathered from
// @how   one thread per (row, column), scatter-add (no atomics: the rows named in one call are unique)
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The scatter-add half of row-compacted MoE training (`model::moe`): for i < n_idx and c < width,
//
//   out[idx[i] * dst_stride + col0 + c] += src[i * src_stride + col0 + c]
//
// A compacted expert's input gradient goes back with width = d_model, col0 = 0; its column of the gate gradient goes
// back with width = 1, col0 = the expert, both strides n_experts. The rows of one call are unique (one expert's
// routed rows), so no two invocations write the same element. An index of 0xFFFFFFFF is padding and is skipped.
//
//   params: n_idx, width, src_stride, dst_stride, col0

struct Params {
    n_idx: u32,
    width: u32,
    src_stride: u32,
    dst_stride: u32,
    col0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       idx: array<u32>;
@group(0) @binding(2) var<storage, read>       src: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let t = gid.y * (nwg.x * 64u) + gid.x;
    if (t >= p.n_idx * p.width) { return; }
    let i = t / p.width;
    let c = t % p.width;
    let row = idx[i];
    if (row == 0xFFFFFFFFu) { return; }
    let o = row * p.dst_stride + p.col0 + c;
    out[o] = out[o] + src[i * p.src_stride + p.col0 + c];
}
