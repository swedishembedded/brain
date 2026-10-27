// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read logits: each query head against the records its group's bank holds, plus a zero sink
// @how   one thread per (row, head, slot), serial dot over head_dim
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The read side of an anchor memory, step 1 of 3 (this kernel, a row softmax
// over `capacity + 1` columns, `anchor_read_apply`). A token at position `pos`
// of sequence `s` reads the bank committed before its commit block
// `pos / chunk` (see `anchor_commit`), through the head's KV group:
//
//   logits[row, h, c] = q[row, h, :] . key[s * t + ref, g, :] / sqrt(head_dim)
//                       for slot c < capacity holding a record `ref`,
//                     = -3.4e38 for an empty slot,
//   logits[row, h, capacity] = 0       the sink: an always-present column
//                                      whose value is zero, so an empty bank
//                                      reads zero and a weak match is damped.
//
//   q      : [n_seq * t, n_heads * head_dim]   rotated query
//   key    : [n_seq * t, groups * head_dim]    rotated written keys
//   refs   : [n_blocks, n_seq, groups, capacity] (`anchor_commit`'s bank, block-major)
//   logits : [n_seq * t * n_heads, capacity + 1]
//
// A record's key row is `base + ref - 1` where `base = s * kv_rows` for a sequence's own
// tensors (`slotted` 0: the sequence form, `kv_rows` = t) or `slots[s] * kv_rows` for a pool
// of fixed-state banks (`slotted` 1: a decode step or a chunk from a state, `kv_rows` = K).
// There `refs` holds the slot index + 1 of each filled bank slot.

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
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       key:    array<f32>;
@group(0) @binding(3) var<storage, read>       refs:   array<u32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;
@group(0) @binding(5) var<storage, read>       slots:  array<u32>;

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
        logits[gidx] = 0.0;
        return;
    }

    let s = row / p.t;
    let pos = row % p.t;
    let n_seq = p.rows / p.t;
    let g = h / (p.n_heads / p.groups);
    let slot = refs[(((pos / p.chunk) * n_seq + s) * p.groups + g) * p.capacity + c];
    if (slot == 0u) {
        logits[gidx] = -3.4e38;
        return;
    }

    let hd = p.head_dim;
    let q_base = row * p.n_heads * hd + h * hd;
    var base = s * p.kv_rows;
    if (p.slotted != 0u) { base = slots[s] * p.kv_rows; }
    let k_base = (base + slot - 1u) * p.groups * hd + g * hd;
    var dot = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
        dot = dot + q[q_base + d] * key[k_base + d];
    }
    logits[gidx] = dot * inverseSqrt(f32(hd));
}
