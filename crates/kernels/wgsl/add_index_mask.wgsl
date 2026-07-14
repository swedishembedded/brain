// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Add the DSA per-(query,key) sparse mask into the MLA attention scores before
// softmax: scores[b,h,i,j] += mask[b,i,j]  (the mask is shared across heads).
// `mask` is [B*T, T] = (b*T+i)*T+j ; `scores` is [B*H*T*T]. One invocation per
// (b,h,i,j).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,  // T
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       mask:   array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.bsz * p.n_heads * T * T;
    if (gidx >= total) { return; }

    let j = gidx % T;
    let r1 = gidx / T;
    let i = r1 % T;
    let r2 = r1 / T;
    let b = r2 / p.n_heads;
    // -inf + finite stays -inf; adding the mask keeps non-selected keys masked.
    scores[gidx] = scores[gidx] + mask[(b * T + i) * T + j];
}
