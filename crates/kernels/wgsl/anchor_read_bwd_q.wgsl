// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read backward, step 2: the query gradient from the logit gradients and the referenced keys
// @how   one thread per (row, head, channel), serial sum over the bank slots
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
//   dq[row, h, d] = sum_c dlogit[row, h, c] * key[s * t + ref(c), g, d] / sqrt(head_dim)
//
// over the slots that hold a record (an empty slot has a zero logit gradient).
// `dlogit` is `softmax_k_dx`'s output (its sink column is not read). The gradient
// is w.r.t. the ROTATED query.
//
//   dlogit : [n_seq * t * n_heads, capacity + 1]
//   key    : [n_seq * t, groups * head_dim]
//   dq     : [n_seq * t, n_heads * head_dim]

struct Params {
    rows: u32,
    t: u32,
    n_heads: u32,
    groups: u32,
    capacity: u32,
    head_dim: u32,
    chunk: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dlogit: array<f32>;
@group(0) @binding(2) var<storage, read>       key:    array<f32>;
@group(0) @binding(3) var<storage, read>       refs:   array<u32>;
@group(0) @binding(4) var<storage, read_write> dq:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let hd = p.head_dim;
    if (gidx >= p.rows * p.n_heads * hd) { return; }

    let d = gidx % hd;
    let rh = gidx / hd;
    let h = rh % p.n_heads;
    let row = rh / p.n_heads;
    let s = row / p.t;
    let pos = row % p.t;
    let n_seq = p.rows / p.t;
    let g = h / (p.n_heads / p.groups);
    let bank = (((pos / p.chunk) * n_seq + s) * p.groups + g) * p.capacity;
    let cols = p.capacity + 1u;

    var acc = 0.0;
    for (var c = 0u; c < p.capacity; c = c + 1u) {
        let slot = refs[bank + c];
        if (slot != 0u) {
            acc = acc + dlogit[rh * cols + c] * key[(s * p.t + slot - 1u) * p.groups * hd + g * hd + d];
        }
    }
    dq[gidx] = acc * inverseSqrt(f32(hd));
}
