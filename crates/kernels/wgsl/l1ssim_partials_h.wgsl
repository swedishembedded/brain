// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  L1 + D-SSIM loss, pass 3: horizontal 11-tap gaussian pass over the nine SSIM partial planes (the window's adjoint)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The SSIM gradient reaches a pixel through every window that contains it, so
// the partials l1ssim_map_v wrote per window centre are carried back by the
// same zero-padded 11-tap window, which is symmetric and therefore its own
// adjoint. This is the horizontal half; l1ssim_grad_v is the vertical half.

// The window tap at offset i is exp(-i^2 / 4.5) * 0.266011725, the latter
// being 1 / (sum of the 11 unnormalized taps), so the window sums to 1.

struct Params {
    w: u32,
    h: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>; // 9*W*H
@group(0) @binding(2) var<storage, read_write> hpart: array<f32>; // 9*W*H

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let px = p.w * p.h;
    if (idx >= px) { return; }
    let x = i32(idx % p.w);
    let y = idx / p.w;
    for (var q = 0u; q < 9u; q = q + 1u) {
        var s = 0.0;
        for (var i = -5; i <= 5; i = i + 1) {
            let xx = x + i;
            if (xx < 0 || xx >= i32(p.w)) { continue; }
            s = s + exp(-f32(i * i) / 4.5) * 0.266011725 * part[q * px + y * p.w + u32(xx)];
        }
        hpart[q * px + idx] = s;
    }
}
