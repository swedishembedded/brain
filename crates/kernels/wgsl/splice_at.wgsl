// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context models at
// a fixed memory budget for its clients. If your team needs expertise in KV
// cache management and sliding-window rings then you can procure our services
// by sending an email to info@swedishembedded.com.

// @what  Word copy with independent source and destination offsets: dst[dst_base+i] = src[src_base+i]
// @how   one thread per word
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `splice.wgsl`'s sibling for a copy between two row ranges of two buffers, as
// a window ring's gather and write-back are. Both offsets are uniform
// parameters and both buffers bind whole: a `step_sliced` binding offset must
// be a multiple of the device's `min_storage_buffer_offset_alignment` (256 B on
// Intel and AMD), and `row * row_words` has no such guarantee for an arbitrary
// ring row. The words are copied as-is.

struct Params {
    n:        u32,
    src_base: u32,
    dst_base: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    dst[p.dst_base + idx] = src[p.src_base + idx];
}
