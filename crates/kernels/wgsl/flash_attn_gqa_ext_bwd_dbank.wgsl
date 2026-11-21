// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Extended causal GQA attention backward, d-bank: the distance-bank gradient, one thread per distance summing every (batch, head, row)
// @how   one thread per distance, serial walk over every query row at that distance
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
// The bank side of the backward of `flash_attn_gqa_ext.wgsl` (contiguous layout;
// `stats` from `flash_attn_gqa_ext_bwd_prep`). The bias is
// `bias[i,h,d] = sum_r rel[i,h,r] * bank[r,d]` for `d < extent`, so with `dZ_ij`
// as in `flash_attn_gqa_ext_bwd_dq.wgsl`:
//
//   dBank[r, d] = sum_{b,h,i: i >= d, d inside the window} tb_i * dZ_{i,i-d} * rel[b,i,h,r]
//
// One thread owns one distance and all `d_rel` of its outputs, so every element
// is written exactly once (no atomics). The walk recomputes the score at that
// one diagonal: `O(extent * rows * heads * head_dim)` work, a banded attention's.

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
    flags: u32,        // 1 SINK, 2 BIAS, 4 TAU, 8 TAU_BIAS (PAGED is refused by the host)
    extent: u32,
    d_rel: u32,        // <= 64
    sink_off: u32,
    rel_off: u32,
    bank_off: u32,
    pos0: u32,         // absolute position of row 0
    scale: f32,
    tau_alpha: f32,
    tau_n_floor: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read>       v:      array<f32>;
@group(0) @binding(4) var<storage, read>       d_o:    array<f32>;
@group(0) @binding(5) var<storage, read>       stats:  array<f32>;
@group(0) @binding(6) var<storage, read>       aux:    array<f32>;
@group(0) @binding(7) var<storage, read_write> d_bank: array<f32>;

// The per-row query factor and bias factor at absolute position `pos`.
fn tau_q(pos: u32) -> f32 {
    if ((p.flags & 4u) == 0u) { return 1.0; }
    return 1.0 + p.tau_alpha * log(max(f32(pos + 1u) / p.tau_n_floor, 1.0));
}

fn tau_b(pos: u32) -> f32 {
    if ((p.flags & 12u) != 12u) { return 1.0; }
    return tau_q(pos);
}

// The distance-bank bias of query row `row`, head `h` at distance `dist` (0 beyond the extent or without BIAS).
fn bias_at(row: u32, h: u32, dist: u32) -> f32 {
    if ((p.flags & 2u) == 0u || dist >= p.extent) { return 0.0; }
    let rb = p.rel_off + (row * p.n_heads + h) * p.d_rel;
    var b = 0.0;
    for (var r = 0u; r < p.d_rel; r = r + 1u) {
        b = b + aux[rb + r] * aux[p.bank_off + r * p.extent + dist];
    }
    return b;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let dist = gid.y * (nwg.x * 64u) + gid.x;
    if (dist >= p.extent) { return; }
    let T = p.max_q;
    let nh = p.n_heads;
    let hd = p.head_dim;
    let kvd = p.n_kv_heads * hd;
    let group = nh / p.n_kv_heads;

    var acc: array<f32, 64>;
    for (var r = 0u; r < p.d_rel; r = r + 1u) { acc[r] = 0.0; }
    let live = dist < T && (p.window == 0u || dist < p.window);
    if (live) {
        for (var b = 0u; b < p.n_seqs; b = b + 1u) {
            for (var h = 0u; h < nh; h = h + 1u) {
                let hkv = h / group;
                for (var i = dist; i < T; i = i + 1u) {
                    let j = i - dist;
                    let row = b * T + i;
                    let pos = p.pos0 + i;
                    let qb = (row * nh + h) * hd;
                    let kb = (b * T + j) * kvd + hkv * hd;
                    let st = 2u * (row * nh + h);
                    var dot = 0.0;
                    var dp = 0.0;
                    for (var d = 0u; d < hd; d = d + 1u) {
                        dot = dot + q[qb + d] * k[kb + d];
                        dp = dp + d_o[qb + d] * v[kb + d];
                    }
                    let tb = tau_b(pos);
                    let z = p.scale * tau_q(pos) * dot + tb * bias_at(row, h, dist);
                    let w = tb * exp(z - stats[st]) * (dp - stats[st + 1u]);
                    let rb = p.rel_off + (row * nh + h) * p.d_rel;
                    for (var r = 0u; r < p.d_rel; r = r + 1u) { acc[r] = acc[r] + w * aux[rb + r]; }
                }
            }
        }
    }
    for (var r = 0u; r < p.d_rel; r = r + 1u) { d_bank[r * p.extent + dist] = acc[r]; }
}
