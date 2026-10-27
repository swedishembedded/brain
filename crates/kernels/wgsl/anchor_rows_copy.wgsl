// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Row copy between pools by row index: dst[dst_row * stride + i] = src[src_row * stride + i]
// @how   one thread per word
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Moves whole rows of a fixed-state pool (u32 or f32 words, copied bit for bit)
// where the row indices come from a buffer, so a recorded decode step names the
// same buffers whichever sequences it carries. Per row (`plan`, 2 words):
// source row, destination row.

struct Params {
    n: u32,
    stride: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       plan: array<u32>;
@group(0) @binding(2) var<storage, read>       src:  array<u32>;
@group(0) @binding(3) var<storage, read_write> dst:  array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.n * p.stride) { return; }
    let i = gidx / p.stride;
    let w = gidx % p.stride;
    dst[plan[i * 2u + 1u] * p.stride + w] = src[plan[i * 2u] * p.stride + w];
}
