// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training copy of a contiguous span: dst[dst_off + i] (+)= src[src_off + i]
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Concatenation, slicing and their backward (acc 1 accumulates into the gradient). Offsets are in elements.
// Params: len, src_off, dst_off, acc. Dispatch: len threads.

struct Params { len: u32, src_off: u32, dst_off: u32, acc: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.len) { return; }
    if (p.acc != 0u) {
        dst[p.dst_off + idx] = dst[p.dst_off + idx] + src[p.src_off + idx];
    } else {
        dst[p.dst_off + idx] = src[p.src_off + idx];
    }
}
