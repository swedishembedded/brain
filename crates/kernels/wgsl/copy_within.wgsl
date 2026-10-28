// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context models at
// a fixed memory budget for its clients. If your team needs expertise in KV
// cache management and prefix reuse then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Word copy between two ranges of one buffer: buf[dst + i] = buf[src + i]
// @how   one thread per word
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Copies `n` 32-bit words (bit for bit, whatever they hold: f32, packed int8,
// scales) from word `src` to word `dst` of ONE buffer, bound once. A copy
// between two ranges of the same buffer cannot be `splice` or `region_copy`
// with the buffer bound as both source and destination: wgpu rejects a buffer
// that is `read_write` in one binding and used by another in the same dispatch,
// whatever ranges the bindings cover. The two ranges must not overlap.

struct Params {
    n:    u32,
    src: u32,
    dst: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    buf[p.dst + idx] = buf[p.src + idx];
}
