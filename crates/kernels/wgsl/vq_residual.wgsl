// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Residual vector quantization step: subtract each query's assigned codebook row in place and record its code
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One stage of a residual vector quantizer, after `vq_argmin` has assigned
// every query `m` of `x[m, d]` its nearest row of `cb[k, d]`:
//
//   x[m, c]          -= cb[idx(m), c]          (the residual the next stage reads)
//   codes[m*nq + q]   = idx(m)                 (written by the c == 0 lane)
//
// where idx(m) = u32(assign[2m]) (`vq_argmin`'s packed `(index, distance)`
// output; the index is an exact f32 below 2^24). `codes` is `[m, nq]` f32, the
// stage `q` column filled here, so a whole quantizer stack reads back once.
// Params: m, d, nq, q. Dispatch: m * d invocations. The subtraction is the one
// f32 operation the host loop it replaces performs, bit for bit.

struct Params {
    m: u32,
    d: u32,
    nq: u32,
    q: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       assign: array<f32>;
@group(0) @binding(2) var<storage, read>       cb:     array<f32>;
@group(0) @binding(3) var<storage, read_write> x:      array<f32>;
@group(0) @binding(4) var<storage, read_write> codes:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.m * p.d) { return; }
    let m = idx / p.d;
    let c = idx % p.d;
    let code = assign[2u * m];
    x[idx] = x[idx] - cb[u32(code) * p.d + c];
    if (c == 0u) {
        codes[m * p.nq + p.q] = code;
    }
}
