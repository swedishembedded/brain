// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Embedding gather scaled and broadcast into the lanes of a hyper-connection decoder: out[b, j, :] = scale * emb[token[b], :]
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: batch, n (lanes), d, scale (f32 bits). Dispatch: batch*n*d threads.

struct Params { batch: u32, n: u32, d: u32, scale: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens: array<u32>;
@group(0) @binding(2) var<storage, read>       emb: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.n * p.d) { return; }
    let c = idx % p.d;
    let b = idx / (p.d * p.n);
    out[idx] = emb[tokens[b] * p.d + c] * p.scale;
}
