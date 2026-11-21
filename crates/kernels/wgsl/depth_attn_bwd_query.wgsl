// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements depth-attention residual streams for its
// clients. If your team needs expertise in training decoders that attend over
// their own earlier layers then you can procure our services by sending an email
// to info@swedishembedded.com.

// @what  Depth attention backward to the learned queries: dquery += scale * sum over rows and states of dz * normed state
// @how   one thread per query element, serial over rows and the stack
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The queries are shared by every row, so their gradient reduces over the rows:
//   dquery[q, j] += scale * sum_r sum_i dz[r, q, i] * rms[r, i] * s_i[r, j]
// with dz from depth_attn_bwd and rms from the forward's stats. ADDED to dquery: a looped layer's queries are used
// once per visit and the gradient is the sum over the visits.
//
//   store, partial, stats: as the forward   dz: [rows, nq*n]   dquery: [nq, dim] (+=)
//   params: the forward's
// Dispatch: nq * dim invocations.

struct Params {
    rows: u32,
    dim: u32,
    count: u32,
    nq: u32,
    eps: f32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    slots: array<vec4<u32>, 8>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       store: array<f32>;
@group(0) @binding(2) var<storage, read>       partial: array<f32>;
@group(0) @binding(3) var<storage, read>       stats: array<f32>;
@group(0) @binding(4) var<storage, read>       dz: array<f32>;
@group(0) @binding(5) var<storage, read_write> dquery: array<f32>;

fn state(i: u32, row: u32, j: u32) -> f32 {
    if (i < p.count) {
        let slot = p.slots[i / 4u][i % 4u];
        return store[(slot * p.rows + row) * p.dim + j];
    }
    return partial[row * p.dim + j];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.nq * p.dim) { return; }
    let q = idx / p.dim;
    let j = idx % p.dim;
    let n = p.count + 1u;
    let stride = n * (1u + p.nq);
    var acc = 0.0;
    for (var row = 0u; row < p.rows; row = row + 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            acc = acc + dz[(row * p.nq + q) * n + i] * stats[row * stride + i] * state(i, row, j);
        }
    }
    dquery[idx] = dquery[idx] + p.scale * acc;
}
