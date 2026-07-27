// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Decode-step attention apply: the single query's context vector as the
// probability-weighted sum of the cached values, GQA-aware.
//   ctx[h, d] = sum_j probs[h, j] * vcache[j, kvhead(h), d]   for j in 0..t
//   probs : [n_heads, cap]              cap = max_T stride
//   vcache: [max_T, n_kv*head_dim]      persistent value cache (rows 0..t valid)
//   ctx   : [n_heads*head_dim]          the attention output for the new token
// GQA: kvhead(h) = h / (n_heads/n_kv_heads). One invocation per (h, d).
// Barrier-free → JITs to CPU.

struct Params {
    n_heads: u32,
    group: u32,      // n_heads / n_kv_heads
    head_dim: u32,
    t: u32,          // cached length
    cap: u32,        // probs row stride (max_T)
    kv_stride: u32,  // n_kv_heads * head_dim (cache row width)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs:  array<f32>;
@group(0) @binding(2) var<storage, read>       vcache: array<f32>;
@group(0) @binding(3) var<storage, read_write> ctx:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n_heads * p.head_dim;
    if (idx >= total) { return; }
    let h = idx / p.head_dim;
    let d = idx % p.head_dim;
    let kvh = h / p.group;
    let pbase = h * p.cap;
    var acc = 0.0;
    for (var j: u32 = 0u; j < p.t; j = j + 1u) {
        acc = acc + probs[pbase + j] * vcache[j * p.kv_stride + kvh * p.head_dim + d];
    }
    ctx[idx] = acc;
}
