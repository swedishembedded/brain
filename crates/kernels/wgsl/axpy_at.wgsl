// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of deep models for its
// clients. If your team needs expertise in gradients that several consumers of
// one activation add into then you can procure our services by sending an email
// to info@swedishembedded.com.

// @what  Scaled accumulate between ranges of two buffers: out[out_off + i] += s * in[in_off + i]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// axpy with an element offset into each operand, for moving one slot of a packed store (a retained-state gradient,
// `[slots, rows, dim]`) into or out of a dense buffer without binding a sub-range, whose byte offset a device may
// refuse unless it is a multiple of its storage-offset alignment. `out` and `in` are distinct buffers.

struct Params {
    n: u32,
    s: f32,
    out_off: u32,
    in_off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read>       inp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    out[p.out_off + idx] = out[p.out_off + idx] + p.s * inp[p.in_off + idx];
}
