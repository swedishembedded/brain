// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read scores for all the query heads of a KV group at once: each key row is read once
// @how   one thread per (row, group, slot); the group's heads accumulate in registers, 4-wide loads in the same summation order
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The same logits as `anchor_read_scores` (bit for bit: each head's dot product sums its channels in the same
// order), with the key row of a slot loaded once for the `n_heads / groups` heads that read it instead of once per
// head. Requires `head_dim % 4 == 0` and at most 8 heads per group (the caller falls back to
// `anchor_read_scores` otherwise).

struct Params {
    rows: u32,       // n_seq * t
    t: u32,
    n_heads: u32,
    groups: u32,
    capacity: u32,
    head_dim: u32,
    chunk: u32,
    kv_rows: u32,
    slotted: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       key:    array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       refs:   array<u32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<storage, read>       slots:  array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let cols = p.capacity + 1u;
    if (gidx >= p.rows * p.groups * cols) { return; }

    let c = gidx % cols;
    let rg = gidx / cols;
    let g = rg % p.groups;
    let row = rg / p.groups;
    let hpg = p.n_heads / p.groups;
    let first = row * p.n_heads + g * hpg;   // the group's first (row, head)

    if (c == p.capacity) {
        for (var j = 0u; j < hpg; j = j + 1u) { logits[(first + j) * cols + c] = 0.0; }
        return;
    }
    let s = row / p.t;
    let pos = row % p.t;
    let n_seq = p.rows / p.t;
    let slot = refs[(((pos / p.chunk) * n_seq + s) * p.groups + g) * p.capacity + c];
    if (slot == 0u) {
        for (var j = 0u; j < hpg; j = j + 1u) { logits[(first + j) * cols + c] = -3.4e38; }
        return;
    }

    let hd4 = p.head_dim / 4u;
    var base = s * p.kv_rows;
    if (p.slotted != 0u) { base = slots[s] * p.kv_rows; }
    let k_base = ((base + slot - 1u) * p.groups + g) * hd4;
    var dot: array<f32, 8>;
    for (var j = 0u; j < 8u; j = j + 1u) { dot[j] = 0.0; }
    for (var d = 0u; d < hd4; d = d + 1u) {
        let kv = key[k_base + d];
        for (var j = 0u; j < 8u; j = j + 1u) {
            if (j < hpg) {
                let qv = q[(first + j) * hd4 + d];
                var a = dot[j];
                a = a + qv.x * kv.x;
                a = a + qv.y * kv.y;
                a = a + qv.z * kv.z;
                a = a + qv.w * kv.w;
                dot[j] = a;
            }
        }
    }
    let scale = inverseSqrt(f32(p.head_dim));
    for (var j = 0u; j < hpg; j = j + 1u) { logits[(first + j) * cols + c] = dot[j] * scale; }
}
