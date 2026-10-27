// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read apply: probability-weighted sum of the bank's written values (the sink adds nothing)
// @how   one thread per output element, serial sum over the bank slots
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The read side of an anchor memory, step 3 of 3 (see `anchor_read_scores`):
//
//   out[row, h, :] = sum_c probs[row, h, c] * value[s * t + ref(c), g, :]
//
// over the slots `c < capacity` that hold a record. The sink column
// (`c == capacity`) carries a zero value and is not summed, so the weights of
// a row sum to less than one when the sink took probability mass.
//
//   probs : [n_seq * t * n_heads, capacity + 1]   row softmax of the logits
//   value : [n_seq * t, groups * head_dim]
//   refs  : [n_seq, n_blocks, groups, capacity]   (`anchor_select`'s layout)
//   out   : [n_seq * t, n_heads * head_dim]

struct Params {
    rows: u32,       // n_seq * t
    t: u32,
    n_heads: u32,
    groups: u32,
    capacity: u32,
    head_dim: u32,
    chunk: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       value: array<f32>;
@group(0) @binding(3) var<storage, read>       refs:  array<u32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;

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
    let n_blocks = (p.t + p.chunk - 1u) / p.chunk;
    let g = h / (p.n_heads / p.groups);
    let bank = ((s * n_blocks + pos / p.chunk) * p.groups + g) * p.capacity;
    let cols = p.capacity + 1u;

    var acc = 0.0;
    for (var c = 0u; c < p.capacity; c = c + 1u) {
        let slot = refs[bank + c];
        if (slot == 0u) { continue; }
        acc = acc + probs[rh * cols + c] * value[(s * p.t + slot - 1u) * p.groups * hd + g * hd + d];
    }
    out[gidx] = acc;
}
