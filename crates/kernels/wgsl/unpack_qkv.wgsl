// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Inverse of pack_qkv: split one fused [seq, 3*d_model] gradient buffer (laid out per token as [ q(d) / k(d) / v(d) ]) back into three contiguous [seq, d_model] grad buffers
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Inverse of pack_qkv: split one fused [seq, 3*d_model] gradient buffer (laid out
// per token as [ q(d) | k(d) | v(d) ]) back into three contiguous [seq, d_model]
// grad buffers. The bidirectional attention backward trio writes d_qkv in the
// packed layout; the DiT block backward then routes the q/k regions through the
// interleaved-RoPE and QK-RMSNorm backward (which need contiguous [seq, d]) and
// the v region straight into the wv linear backward. One invocation per input
// element (seq*3*d).

struct Params {
    seq_len: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> q:   array<f32>;
@group(0) @binding(3) var<storage, read_write> k:   array<f32>;
@group(0) @binding(4) var<storage, read_write> v:   array<f32>;

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
        q[t * d + r] = qkv[gidx];
    } else if (r < 2u * d) {
        k[t * d + (r - d)] = qkv[gidx];
    } else {
        v[t * d + (r - 2u * d)] = qkv[gidx];
    }
}
