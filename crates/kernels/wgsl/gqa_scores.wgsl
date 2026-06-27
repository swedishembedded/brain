// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GQA attention scores (materialised, for training), separate q/k buffers:
//   scores[b,h,i,j] = (q[b,i,h,:] . k[b,j,hkv,:]) / sqrt(head_dim)   for j <= i
//                   = -inf                                            for j >  i
// where hkv = h / group maps each query head to its shared key/value head
// (grouped-query attention). q is [B*T, n_heads*head_dim], k is
// [B*T, n_kv_heads*head_dim], both head-major within a row. scores layout
// matches the dense path so `attn_softmax` is reused unchanged:
//   ((b*n_heads + h)*T + i)*T + j.
// One invocation per (b,h,i,j).

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,        // T
    head_dim: u32,
    group: u32,        // n_heads / n_kv_heads
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.bsz * p.n_heads * T * T;
    let idx = gidx;
    if (idx >= total) { return; }

    let j = idx % T;
    let r1 = idx / T;
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    if (j > i) { scores[idx] = -3.4e38; return; }

    let hd = p.head_dim;
    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let k_row = p.n_kv_heads * hd;
    let q_base = (b * T + i) * q_row + h * hd;
    let k_base = (b * T + j) * k_row + hkv * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[q_base + d] * k[k_base + d];
    }
    scores[idx] = s * inverseSqrt(f32(hd));
}
