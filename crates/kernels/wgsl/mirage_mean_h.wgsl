// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Mean over the height of an NHWC map: y[b, w, c] = mean_h x[b, h, w, c] (Mirage reader frames)
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: batch, h, w, c. Dispatch: batch*w*c threads.

struct Params { batch: u32, h: u32, w: u32, c: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.w * p.c) { return; }
    let c = idx % p.c;
    let wx = (idx / p.c) % p.w;
    let b = idx / (p.c * p.w);
    var acc = 0.0;
    for (var i = 0u; i < p.h; i = i + 1u) { acc = acc + x[((b * p.h + i) * p.w + wx) * p.c + c]; }
    y[idx] = acc / f32(p.h);
}
