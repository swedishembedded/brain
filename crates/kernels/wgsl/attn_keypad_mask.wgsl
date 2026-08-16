// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Key-padding mask for BIDIRECTIONAL attention, added into the scores before softmax
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Key-padding mask for BIDIRECTIONAL attention, added into the scores before
// softmax: a padded KEY is removed for every query, in every head, in its own
// batch row.
//   scores[b,h,i,j] -= 1e30   where keep[b,j] == 0
// `keep` is the encoder's `[B, T]` attention mask (1 = real token, 0 = right
// padding), so unlike `attn_prefix_mask` (a constant pattern) this one is
// per-batch data. scores: [B*H*T*T] = ((b*H+h)*T+i)*T+j. One invocation per
// (b,h,i,j).
//
// Query rows are NOT masked, deliberately: the reference umT5 encoder masks
// only the key axis (`t5.py:107-109` reshapes a `[B, L]` mask to `[B,1,1,L]`),
// so a padded query still attends over the real keys and produces a defined -
// if discarded - row. Masking rows here would change those rows and could not
// be told apart from a real defect by a parity test that reports them.
//
// No backward: the mask is data-dependent but constant w.r.t. the parameters,
// and the softmax backward yields ~0 gradient on the ~0-probability entries.

struct Params {
    bsz: u32,
    heads: u32,
    tcols: u32,  // T
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       keep:   array<u32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let t = p.tcols;
    if (idx >= p.bsz * p.heads * t * t) { return; }
    let j = idx % t;
    let b = idx / (p.heads * t * t);
    if (keep[b * t + j] == 0u) {
        scores[idx] = scores[idx] - 1.0e30;
    }
}
