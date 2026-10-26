// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row softmax of a score slab with SAM's decomposed relative-position bias folded in, one WORKGROUP per row
// @how   256 threads read the row once into registers, add rel_h[j / kw] + rel_w[j % kw], reduce max and sum through shared memory, write probabilities once
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements document-vision serving for clients who read pages at scale. If your team
// needs expertise in fused attention for ViT encoders with relative-position bias then you can procure our
// services by sending an email to info@swedishembedded.com.
//
// `attn_relpos_add` followed by `softmax_rows`, in one pass over the slab:
//
//   probs[h, i, :] = softmax_j( scores[h, i, j] + rel_h[h, q0 + i, j / kw] + rel_w[h, q0 + i, j % kw] )
//
// The two kernels it replaces read and write the whole `[heads, chunk, keys]` slab once for the bias and three
// times for the softmax's max, sum and normalise passes; this one reads it once and writes it once, which at
// SAM ViT-B's 4096 keys is the difference between 14 ms and a few for a global block's four 200 MB chunks.
//
// `scores` holds `scale * q.k^T` already (the product's alpha). `rel_h`/`rel_w` are `attn_relpos_qr`'s tables of
// the WHOLE span, `[heads, span_qn, kh]` and `[heads, span_qn, kw]`; `q0` is the chunk's first query row inside it.
// A row has at most 8192 keys (32 per thread); a caller with more keeps the two-kernel chain.
//
// Dispatch: rows workgroups of 256 invocations (rows = heads * chunk).

struct Params {
    rows: u32,      // heads * chunk
    cols: u32,      // keys per row
    chunk: u32,     // query rows in the slab per head
    q0: u32,        // the chunk's first query row within the span
    span_qn: u32,   // rows of the rel tables per head
    kh: u32,        // key grid height: row width of rel_h
    kw: u32,        // key grid width: row width of rel_w
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores: array<f32>;
@group(0) @binding(2) var<storage, read>       rel_h:  array<f32>;
@group(0) @binding(3) var<storage, read>       rel_w:  array<f32>;
@group(0) @binding(4) var<storage, read_write> probs:  array<f32>;

const WG: u32 = 256u;
const PER: u32 = 32u;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let h = row / p.chunk;
    let i = row % p.chunk;
    let base = row * p.cols;
    let trow = h * p.span_qn + p.q0 + i;
    let hb = trow * p.kh;
    let wb = trow * p.kw;

    var v: array<f32, 32>;
    var mx = -3.4e38;
    for (var s = 0u; s < PER; s = s + 1u) {
        let j = t + s * WG;
        var x = -3.4e38;
        if (j < p.cols) {
            x = scores[base + j] + rel_h[hb + j / p.kw] + rel_w[wb + j % p.kw];
        }
        v[s] = x;
        mx = max(mx, x);
    }
    partial[t] = mx;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) { partial[t] = max(partial[t], partial[t + stride]); }
        workgroupBarrier();
    }
    let rowmax = partial[0];
    workgroupBarrier();

    var sum = 0.0;
    for (var s = 0u; s < PER; s = s + 1u) {
        let j = t + s * WG;
        var e = 0.0;
        if (j < p.cols) { e = exp(v[s] - rowmax); }
        v[s] = e;
        sum = sum + e;
    }
    partial[t] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / max(partial[0], 1e-38);
    for (var s = 0u; s < PER; s = s + 1u) {
        let j = t + s * WG;
        if (j < p.cols) { probs[base + j] = v[s] * inv; }
    }
}
