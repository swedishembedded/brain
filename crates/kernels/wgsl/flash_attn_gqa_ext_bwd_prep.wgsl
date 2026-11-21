// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Extended causal GQA attention backward, row prologue: D = dO.O per (row, head), packed with lse for the gradient kernels, and the sink-logit gradient
// @how   one thread per (row, head), serial reduction over head_dim
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements memory-bounded attention training kernels
// for its clients. If your team needs expertise in long-context fine-tuning
// then you can procure our services by sending an email to
// info@swedishembedded.com.
//
// First of the backward of `flash_attn_gqa_ext.wgsl` (contiguous layout). With
// `D_i = sum_d dO_i[d] * O_i[d]` and the forward's `lse_i` (which includes the
// sink), it writes `stats[2*(row*H+h)] = lse`, `stats[.. + 1] = D` - one
// binding for both, so the gradient kernels stay within 8 storage buffers -
// and, with SINK, `d_sink = -exp(sink - lse) * D`: the sink's logit joins the
// softmax with value 0, so `dL/dsink = p_sink * (0 - D)`.
//
// DOCS: `doc[row]` is the row's position inside its packed document (the
// forward's `index`), and the thread of head 0 also writes the row's document
// start, relative to its sequence, as an exact f32 at
// `stats[2*rows*n_heads + row]`: the gradient kernels read it there (each is
// already at 8 storage buffers). Without DOCS nothing is written past `2*rows*n_heads`.

struct Params {
    n_seqs: u32,       // batch
    max_q: u32,        // T, rows per sequence
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 256
    block_size: u32,   // unused (contiguous only)
    max_bt: u32,       // unused
    ring: u32,         // unused
    window: u32,       // 0: plain causal
    flags: u32,        // 1 SINK, 2 BIAS, 4 TAU, 8 TAU_BIAS, 32 DOCS (PAGED is refused by the host)
    extent: u32,
    d_rel: u32,        // <= 64
    sink_off: u32,
    rel_off: u32,
    bank_off: u32,
    pos0: u32,         // absolute position of row 0 (0 with DOCS)
    scale: f32,
    tau_alpha: f32,
    tau_n_floor: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       o:      array<f32>;
@group(0) @binding(2) var<storage, read>       d_o:    array<f32>;
@group(0) @binding(3) var<storage, read>       lse:    array<f32>;
@group(0) @binding(4) var<storage, read>       aux:    array<f32>;
@group(0) @binding(5) var<storage, read_write> stats:  array<f32>;
@group(0) @binding(6) var<storage, read_write> d_sink: array<f32>;
@group(0) @binding(7) var<storage, read>       doc:    array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n_seqs * p.max_q * p.n_heads) { return; }
    let hd = p.head_dim;
    let base = idx * hd;
    var dsum = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) { dsum = dsum + d_o[base + d] * o[base + d]; }
    let l = lse[idx];
    stats[2u * idx] = l;
    stats[2u * idx + 1u] = dsum;
    if ((p.flags & 1u) != 0u) {
        d_sink[idx] = -exp(aux[p.sink_off + idx] - l) * dsum;
    }
    if ((p.flags & 32u) != 0u && idx % p.n_heads == 0u) {
        let row = idx / p.n_heads;
        let i = row % p.max_q;
        stats[2u * p.n_seqs * p.max_q * p.n_heads + row] = f32(i - min(doc[row], i));
    }
}
