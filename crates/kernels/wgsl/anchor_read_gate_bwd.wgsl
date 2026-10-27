// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-context inference at a
// fixed memory budget for its clients. If your team needs expertise in
// bounded-memory model architectures then you can procure our services by
// sending an email to info@swedishembedded.com.

// @what  Anchor-memory read gate backward, per (row, head): dz, the scale gradient term and the gradient into the normalised query
// @how   one thread per (row, head), serial pass over head_dim
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward of `anchor_read_gate`: with `t = tanh(scale[h])`, `s = sigmoid(z)`,
// `z = q . w[h] / sqrt(head_dim) + bias[h]` and `gate = 2 t s`:
//
//   dz[row, h]  = dgate * 2 t s (1 - s)
//   dsc[row, h] = dgate * 2 s (1 - t^2)               (summed over rows by `anchor_read_gate_wgrad`)
//   dq[row, h, :] += dz * w[h, :] / sqrt(head_dim)    (dq already holds the read's gradient)
//
//   q  : [rows, n_heads * head_dim]   the normalised query BEFORE the rotary embedding
//   dq : [rows, n_heads * head_dim]   read_write

struct Params {
    rows: u32,
    n_heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       dgate:  array<f32>;
@group(0) @binding(3) var<storage, read>       scale:  array<f32>;
@group(0) @binding(4) var<storage, read>       weight: array<f32>;
@group(0) @binding(5) var<storage, read>       bias:   array<f32>;
@group(0) @binding(6) var<storage, read_write> dz:     array<f32>;
@group(0) @binding(7) var<storage, read_write> dsc:    array<f32>;
@group(0) @binding(8) var<storage, read_write> dq:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.rows * p.n_heads) { return; }

    let h = gidx % p.n_heads;
    let row = gidx / p.n_heads;
    let hd = p.head_dim;
    let inv = inverseSqrt(f32(hd));
    var dot = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
        dot = dot + q[row * p.n_heads * hd + h * hd + d] * weight[h * hd + d];
    }
    let s = 1.0 / (1.0 + exp(-(dot * inv + bias[h])));
    let t = tanh(scale[h]);
    let z_grad = dgate[gidx] * 2.0 * t * s * (1.0 - s);
    dz[gidx] = z_grad;
    dsc[gidx] = dgate[gidx] * 2.0 * s * (1.0 - t * t);
    for (var d = 0u; d < hd; d = d + 1u) {
        let i = row * p.n_heads * hd + h * hd + d;
        dq[i] = dq[i] + z_grad * weight[h * hd + d] * inv;
    }
}
