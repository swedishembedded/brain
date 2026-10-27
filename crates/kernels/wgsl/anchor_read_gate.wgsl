// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory per-head read gate: tanh(scale) * 2 * sigmoid(q . w / sqrt(head_dim) + bias)
// @how   one thread per (row, head), serial dot over head_dim
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// How much of a head's memory read is let into the layer output:
//
//   gate[row, h] = tanh(scale[h]) * 2 * sigmoid(q[row, h, :] . w[h, :] / sqrt(head_dim) + bias[h])
//
// `q` is the head's normalised query BEFORE the rotary embedding, so the gate
// depends on content only, not on position. Each head has its own weight row.
//
//   q    : [rows, n_heads * head_dim]
//   gate : [rows * n_heads]

struct Params {
    rows: u32,
    n_heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       scale:  array<f32>;
@group(0) @binding(3) var<storage, read>       weight: array<f32>;
@group(0) @binding(4) var<storage, read>       bias:   array<f32>;
@group(0) @binding(5) var<storage, read_write> gate:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.rows * p.n_heads) { return; }

    let h = gidx % p.n_heads;
    let row = gidx / p.n_heads;
    let hd = p.head_dim;
    var dot = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
        dot = dot + q[row * p.n_heads * hd + h * hd + d] * weight[h * hd + d];
    }
    let z = dot * inverseSqrt(f32(hd)) + bias[h];
    gate[gidx] = tanh(scale[h]) * 2.0 / (1.0 + exp(-z));
}
