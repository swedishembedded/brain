// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory pending block for a bf16 bank: append each sequence's new record, key and value rounded to bf16
// @how   one thread per (sequence, channel pair); round to nearest even, two channels to a word
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `anchor_push` for a pool that stores its keys and values as bf16 (two channels to a `u32`, the even channel in the low
// half): the same rows move, the f32 record is rounded to bf16 on the way in, and the admission score stays f32.
//
//   src_key, src_value : [rows, groups * head_dim] f32      src_score : [rows, groups] f32
//   dst_key, dst_value : [slots * C, groups * head_dim / 2] words      dst_score : [slots * C, groups] f32
//
// `head_dim` is even. Dispatch: n * groups * head_dim / 2 threads.

struct Params {
    n: u32,
    groups: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       plan:      array<u32>;
@group(0) @binding(2) var<storage, read>       src_key:   array<f32>;
@group(0) @binding(3) var<storage, read>       src_value: array<f32>;
@group(0) @binding(4) var<storage, read>       src_score: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst_key:   array<u32>;
@group(0) @binding(6) var<storage, read_write> dst_value: array<u32>;
@group(0) @binding(7) var<storage, read_write> dst_score: array<f32>;

// The bf16 nearest to `x` (ties to even), as its 16 bits.
fn bf16_bits(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return (b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let gd = p.groups * p.head_dim;
    let pairs = gd / 2u;
    if (gidx >= p.n * pairs) { return; }

    let i = gidx / pairs;
    let col = gidx % pairs;
    let dst = plan[i * 2u];
    let src = plan[i * 2u + 1u];
    let s0 = src * gd + 2u * col;
    dst_key[dst * pairs + col] = bf16_bits(src_key[s0]) | (bf16_bits(src_key[s0 + 1u]) << 16u);
    dst_value[dst * pairs + col] = bf16_bits(src_value[s0]) | (bf16_bits(src_value[s0 + 1u]) << 16u);
    if (col < p.groups) {
        dst_score[dst * p.groups + col] = src_score[src * p.groups + col];
    }
}
