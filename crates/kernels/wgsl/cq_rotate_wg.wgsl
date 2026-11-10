// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements solutions for running aggressively quantised networks on constrained devices for its
// clients. If your team needs expertise in rotated-codebook weight formats then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Orthonormal natural-order Walsh transform of every 128-lane group of f32 activation rows, zero padded to whole groups - the cooperative form of cq_rotate
// @how   one 64-thread workgroup per (row, group): the group lives in workgroup memory and each of the seven butterfly stages is one lane per pair behind a barrier
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant cq
// @dtype f32
//
// The decode-regime sibling of `cq_rotate`: with one activation row there are only `kpad / 128` groups, and giving each
// to one thread leaves the device idle for the whole serial 7 x 64 butterfly. Here the groups of a row are separate
// workgroups and a group's 64 pair-lanes run each stage together. Same operands, layout and result as `cq_rotate`.
//
//   params : m, k, kpad, xr0
//   x      : [xr0 + m, k] f32
//   out    : [m, kpad] f32

struct Params {
    m: u32,
    k: u32,
    kpad: u32,
    xr0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const GROUP: u32 = 128u;

var<workgroup> lane: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = li.x;
    let groups = p.kpad / GROUP;
    let id = wg.y * nwg.x + wg.x;
    let row = id / groups;
    let base = (id % groups) * GROUP;
    let live = row < p.m;
    for (var j = t; j < GROUP; j = j + 64u) {
        var v = 0.0;
        if (live && base + j < p.k) { v = x[(p.xr0 + row) * p.k + base + j]; }
        lane[j] = v;
    }
    workgroupBarrier();
    for (var h = 1u; h < GROUP; h = h * 2u) {
        let a = (t / h) * 2u * h + (t % h);
        let l = lane[a];
        let r = lane[a + h];
        workgroupBarrier();
        lane[a] = l + r;
        lane[a + h] = l - r;
        workgroupBarrier();
    }
    if (live) {
        for (var j = t; j < GROUP; j = j + 64u) { out[row * p.kpad + base + j] = lane[j] * 0.08838834764831845; }
    }
}
