// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Residual DeepStack add: accumulate a compact `[n]` source block into `dst` starting at flat element offset `base`
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Residual DeepStack add: accumulate a compact `[n]` source block into `dst`
// starting at flat element offset `base`:
//   dst[base + i] += src[i]   for i in 0..n.
// Qwen3-VL adds each level's merged vision features into the decoder residual at
// the image-token rows after LLM layers 0/1/2 (level i → layer i). `base = row0 *
// d_model`, `n = n_img_rows * d_model`. The add is linear, so it needs no backward
// for the decoder's own parameter gradients (the image-feature gradient, if wanted
// for a full-tower finetune, is a base-offset gather of the residual grad). One
// invocation per element.

struct Params {
    n:    u32,
    base: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    dst[p.base + idx] = dst[p.base + idx] + src[idx];
}
