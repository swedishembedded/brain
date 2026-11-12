// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context models at
// a fixed memory budget for its clients. If your team needs expertise in KV
// cache management and sliding-window rings then you can procure our services
// by sending an email to info@swedishembedded.com.

// @what  unpack_bf16 with offsets: out[dst + i] = bf16 element (src + i) of the packed buffer, widened
// @how   one thread per output element; a shift into the top 16 bits (exact)
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `unpack_bf16.wgsl` reading `total` elements of the packed buffer starting at
// bf16 ELEMENT `src` (word `src / 2`, half `src % 2`) into `out` starting at
// element `dst`. The offsets are uniform parameters: see `splice_at.wgsl` for
// why a ring row cannot be a binding offset.

struct Params {
    total: u32,
    src:   u32,
    dst:   u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let e = p.src + i;
    let half = (packed[e >> 1u] >> ((e & 1u) * 16u)) & 0xffffu;
    out[p.dst + i] = bitcast<f32>(half << 16u);
}
