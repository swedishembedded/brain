// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Residual of a per-image scale-and-shift fit, and its squared-error term
// @how   one thread per pixel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// With the per-image fit (s, b) solved on the host from ssi_moments.wgsl's sums, the residual on the valid pixels is
//   r = m (s p + b - t)           (zero where the pixel is invalid)
// and the SSI term of the loss is sum_M r^2 / (2 |M|); this kernel emits r and each pixel's r^2 / 2, which the host
// sums per image.
//
//   pred, tgt, mask : [N * hw]    sb : [N, 2] = (s, b) of each image
//   resid, term : [N * hw]  read_write
//
// By the envelope theorem the gradient of that term is s r / |M| and nothing else: (s, b) minimise exactly this
// objective, so the derivative through them vanishes (derived in `zipdepth::loss`).

struct Params {
    total: u32,   // N * hw
    hw: u32,      // pixels per image
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:  array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:   array<f32>;
@group(0) @binding(3) var<storage, read>       mask:  array<f32>;
@group(0) @binding(4) var<storage, read>       sb:    array<f32>;
@group(0) @binding(5) var<storage, read_write> resid: array<f32>;
@group(0) @binding(6) var<storage, read_write> term:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let n = idx / p.hw;
    let r = mask[idx] * (sb[2u * n] * pred[idx] + sb[2u * n + 1u] - tgt[idx]);
    resid[idx] = r;
    term[idx] = 0.5 * r * r;
}
