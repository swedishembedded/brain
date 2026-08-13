// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Pack three separate [seq, d_model] projections (q, k, v) into one fused [seq, 3*d_model] buffer laid out per token as [ q(d) / k(d) / v(d) ] - the layout the bidirectional attention trio (attn_scores_bidir / _softmax_ / _apply_) reads via q_off=0, k_off=d, v_off=2d
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Pack three separate [seq, d_model] projections (q, k, v) into one fused
// [seq, 3*d_model] buffer laid out per token as [ q(d) | k(d) | v(d) ] — the
// layout the bidirectional attention trio (attn_scores_bidir / _softmax_ /
// _apply_) reads via q_off=0, k_off=d, v_off=2d. Used by DiT single-stream
// attention, where q/k are RoPE'd separately before packing. One invocation per
// output element (seq*3*d).

struct Params {
    seq_len: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:   array<f32>;
@group(0) @binding(2) var<storage, read>       k:   array<f32>;
@group(0) @binding(3) var<storage, read>       v:   array<f32>;
@group(0) @binding(4) var<storage, read_write> qkv: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let d = p.d_model;
    let total = p.seq_len * 3u * d;
    if (gidx >= total) { return; }

    let stride = 3u * d;
    let t = gidx / stride;
    let r = gidx % stride;
    if (r < d) {
        qkv[gidx] = q[t * d + r];
    } else if (r < 2u * d) {
        qkv[gidx] = k[t * d + (r - d)];
    } else {
        qkv[gidx] = v[t * d + (r - 2u * d)];
    }
}
