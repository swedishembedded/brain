// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Nearest-neighbour x2 upsample of an NHWC map (channels contiguous)
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// y[b, oy, ox, c] = x[b, oy/2, ox/2, c]. Params carry the INPUT dims. Dispatch: batch*2h*2w*c threads.

struct Params { batch: u32, h: u32, w: u32, c: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * 4u * p.h * p.w * p.c) { return; }
    let c = idx % p.c;
    let pix = idx / p.c;
    let ox = pix % (2u * p.w);
    let oy = (pix / (2u * p.w)) % (2u * p.h);
    let b = pix / (4u * p.w * p.h);
    y[idx] = x[((b * p.h + oy / 2u) * p.w + ox / 2u) * p.c + c];
}
