// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read backward, step 3: key and value gradients gathered per record through the inverse bank map
// @how   one thread per (record, group, channel), serial over the later blocks and the rows that read it
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Record `j` of group `g` was read, in every commit block `b` after its own
// whose bank held it (slot `c`, from `anchor_slot_inverse`), by every row of
// block `b` and every head of the group:
//
//   dkey[j, g, d]   = sum_(b, row in b, head in g) dlogit[row, head, c] * q[row, head, d] / sqrt(head_dim)
//   dvalue[j, g, d] = sum_(b, row in b, head in g) probs[row, head, c]  * d_read[row, head, d]
//
// The work is the forward read's (every row visits `capacity` slots) and no
// gradient is scattered: each output word has one writer. `dkey` is w.r.t. the
// ROTATED key. Records read by nobody get exactly zero.
//
//   slot_of : [n_blocks, n_seq, groups, t]
//   dlogit, probs : [n_seq * t * n_heads, capacity + 1]
//   q, d_read     : [n_seq * t, n_heads * head_dim]
//   dkey, dvalue  : [n_seq * t, groups * head_dim]

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
@group(0) @binding(1) var<storage, read>       slot_of: array<u32>;
@group(0) @binding(2) var<storage, read>       dlogit:  array<f32>;
@group(0) @binding(3) var<storage, read>       probs:   array<f32>;
@group(0) @binding(4) var<storage, read>       q:       array<f32>;
@group(0) @binding(5) var<storage, read>       d_read:  array<f32>;
@group(0) @binding(6) var<storage, read_write> dkey:    array<f32>;
@group(0) @binding(7) var<storage, read_write> dvalue:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let hd = p.head_dim;
    if (gidx >= p.rows * p.groups * hd) { return; }

    let d = gidx % hd;
    let rg = gidx / hd;
    let g = rg % p.groups;
    let row_j = rg / p.groups;            // s * t + position of the record
    let s = row_j / p.t;
    let pos = row_j % p.t;
    let n_blocks = (p.t + p.chunk - 1u) / p.chunk;
    let n_seq = p.rows / p.t;
    let per_group = p.n_heads / p.groups;
    let cols = p.capacity + 1u;
    let scale = inverseSqrt(f32(hd));

    var acc_k = 0.0;
    var acc_v = 0.0;
    for (var b = pos / p.chunk + 1u; b < n_blocks; b = b + 1u) {
        let sl = slot_of[((b * n_seq + s) * p.groups + g) * p.t + pos];
        if (sl != 0u) {
            let c = sl - 1u;
            let r_end = min(p.t, (b + 1u) * p.chunk);
            for (var r = b * p.chunk; r < r_end; r = r + 1u) {
                let row = s * p.t + r;
                for (var hh = 0u; hh < per_group; hh = hh + 1u) {
                    let h = g * per_group + hh;
                    let rh = row * p.n_heads + h;
                    acc_k = acc_k + dlogit[rh * cols + c] * q[rh * hd + d];
                    acc_v = acc_v + probs[rh * cols + c] * d_read[rh * hd + d];
                }
            }
        }
    }
    dkey[gidx] = acc_k * scale;
    dvalue[gidx] = acc_v;
}
