// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Rotary position embedding, NeoX / half-split style (Chronos-2, GPT-NeoX,
// Llama-HF): within a head the rotated pair is (j, j + head_dim/2), NOT the
// interleaved (2j, 2j+1) of `rope.wgsl`. Applied in place to a q or k region.
//
// q'[j]      = q[j]*cos(a) - q[j+half]*sin(a)
// q'[j+half] = q[j+half]*cos(a) + q[j]*sin(a)
// with a = t * theta^(-2j/head_dim), half = head_dim/2, j in 0..half.
//
// Buffer layout per token row: `row_stride` floats, a head occupies `head_dim`
// contiguous channels starting at `base_off + h*head_dim`. Works on a standalone
// [seq, n_heads*head_dim] q buffer (base_off=0, row_stride=n_heads*head_dim) or a
// fused [q|k|v] layout (base_off = d_model for k). One invocation per
// (token, head, pair); pairs never overlap.

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    row_stride: u32,
    base_off: u32,
    theta: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    let total = p.seq_len * p.n_heads * half;
    if (gidx >= total) { return; }

    let j = gidx % half;
    let tmp = gidx / half;
    let h = tmp % p.n_heads;
    let t = tmp / p.n_heads;

    let base = t * p.row_stride + p.base_off + h * p.head_dim;
    let angle = f32(t) * pow(p.theta, -f32(2u * j) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let e = buf[base + j];
    let o = buf[base + half + j];
    buf[base + j]        = e * c - o * s;
    buf[base + half + j] = o * c + e * s;
}
