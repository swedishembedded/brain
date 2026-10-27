// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read backward, step 1: the probability gradient dP of every bank slot (zero for empty slots and the sink)
// @how   one thread per (row, head, slot), serial dot over head_dim
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of `anchor_read_apply` w.r.t. the softmax probabilities:
//
//   dP[row, h, c] = sum_d d_read[row, h, d] * value[s * t + ref(c), g, d]    (slot c holds a record)
//                 = 0                                                        (empty slot, and the sink column)
//
// The softmax backward over the `capacity + 1` columns is `softmax_k_dx` with
// `N = rows * heads`, `K = capacity + 1`, `M = 1`; the sink column carries a
// zero value, so its dP is zero.
//
//   d_read : [n_seq * t, n_heads * head_dim]
//   value  : [n_seq * t, groups * head_dim]
//   refs   : [n_blocks, n_seq, groups, capacity]    (`anchor_commit`'s bank, block-major)
//   dp     : [n_seq * t * n_heads, capacity + 1]

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
@group(0) @binding(1) var<storage, read>       d_read: array<f32>;
@group(0) @binding(2) var<storage, read>       value:  array<f32>;
@group(0) @binding(3) var<storage, read>       refs:   array<u32>;
@group(0) @binding(4) var<storage, read_write> dp:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let cols = p.capacity + 1u;
    if (gidx >= p.rows * p.n_heads * cols) { return; }

    let c = gidx % cols;
    let rh = gidx / cols;
    let h = rh % p.n_heads;
    let row = rh / p.n_heads;
    if (c == p.capacity) {
        dp[gidx] = 0.0;
        return;
    }
    let s = row / p.t;
    let pos = row % p.t;
    let n_seq = p.rows / p.t;
    let g = h / (p.n_heads / p.groups);
    let slot = refs[(((pos / p.chunk) * n_seq + s) * p.groups + g) * p.capacity + c];
    if (slot == 0u) {
        dp[gidx] = 0.0;
        return;
    }
    let hd = p.head_dim;
    let r_base = row * p.n_heads * hd + h * hd;
    let v_base = (s * p.t + slot - 1u) * p.groups * hd + g * hd;
    var acc = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
        acc = acc + d_read[r_base + d] * value[v_base + d];
    }
    dp[gidx] = acc;
}
