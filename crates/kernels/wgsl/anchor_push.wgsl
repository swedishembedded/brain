// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory pending block: append each sequence's new record (key, value, admission score) at its pending row
// @how   one thread per (sequence, channel)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The records of the commit block in progress wait here until the block fills
// and `anchor_commit` merges them into the bank. Per sequence (`plan`, 2 words):
// the destination row in the pending arrays (`slot * C + position % C`) and the
// source row in the step's record tensors.
//
//   src_key, src_value : [rows, groups * head_dim]    src_score : [rows, groups]
//   dst_key, dst_value : [slots * C, groups * head_dim]    dst_score : [slots * C, groups]

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
@group(0) @binding(5) var<storage, read_write> dst_key:   array<f32>;
@group(0) @binding(6) var<storage, read_write> dst_value: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst_score: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let gd = p.groups * p.head_dim;
    if (gidx >= p.n * gd) { return; }

    let i = gidx / gd;
    let col = gidx % gd;
    let dst = plan[i * 2u];
    let src = plan[i * 2u + 1u];
    dst_key[dst * gd + col] = src_key[src * gd + col];
    dst_value[dst * gd + col] = src_value[src * gd + col];
    if (col < p.groups) {
        dst_score[dst * p.groups + col] = src_score[src * p.groups + col];
    }
}
