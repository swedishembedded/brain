// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Extended causal GQA attention backward, dQ and d-rel: rebuilds each weight from lse, never the [H,T,T] probabilities
// @how   one thread per (query row, head), serial key walk
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
// The query side of the backward of `flash_attn_gqa_ext.wgsl` (contiguous
// layout; `stats` from `flash_attn_gqa_ext_bwd_prep`). With the forward's
// logit `z_ij = tq_i*scale*<q_i,k_j> + tb_i*bias_ij`, `P_ij = exp(z_ij - lse_i)`,
// `dP_ij = <dO_i, v_j>` and `dZ_ij = P_ij * (dP_ij - D_i)`:
//
//   dQ_i        = tq_i * scale * sum_j dZ_ij * k_j
//   dRel[i,h,r] = tb_i * sum_{j: i-j < extent} dZ_ij * bank[r, i-j]      (BIAS only)
//
// over the keys the forward saw (causal, inside the window).

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
@group(0) @binding(1) var<storage, read>       q:     array<f32>;
@group(0) @binding(2) var<storage, read>       k:     array<f32>;
@group(0) @binding(3) var<storage, read>       v:     array<f32>;
@group(0) @binding(4) var<storage, read>       d_o:   array<f32>;
@group(0) @binding(5) var<storage, read>       stats: array<f32>;
@group(0) @binding(6) var<storage, read>       aux:   array<f32>;
@group(0) @binding(7) var<storage, read_write> dq:    array<f32>;
@group(0) @binding(8) var<storage, read_write> d_rel: array<f32>;

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
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let nh = p.n_heads;
    let T = p.max_q;
    if (idx >= p.n_seqs * T * nh) { return; }
    let h = idx % nh;
    let row = idx / nh;
    let i = row % T;
    let b = row / T;
    let pos = p.pos0 + i;
    let hd = p.head_dim;
    let hkv = h / (nh / p.n_kv_heads);
    let kvd = p.n_kv_heads * hd;
    let qb = idx * hd;
    let qs = p.scale * tau_q(pos);
    let tb = tau_b(pos);
    let lse_i = stats[2u * idx];
    let d_i = stats[2u * idx + 1u];

    var lo = 0u;
    if (p.window > 0u && i + 1u > p.window) { lo = i + 1u - p.window; }

    var acc: array<f32, 256>;
    for (var d = 0u; d < hd; d = d + 1u) { acc[d] = 0.0; }
    var racc: array<f32, 64>;
    for (var r = 0u; r < p.d_rel; r = r + 1u) { racc[r] = 0.0; }
    let bias_on = (p.flags & 2u) != 0u;

    for (var j = lo; j <= i; j = j + 1u) {
        let kb = (b * T + j) * kvd + hkv * hd;
        var dot = 0.0;
        var dp = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            dot = dot + q[qb + d] * k[kb + d];
            dp = dp + d_o[qb + d] * v[kb + d];
        }
        let dist = i - j;
        let z = qs * dot + tb * bias_at(row, h, dist);
        let dz = exp(z - lse_i) * (dp - d_i);
        for (var d = 0u; d < hd; d = d + 1u) { acc[d] = acc[d] + dz * k[kb + d]; }
        if (bias_on && dist < p.extent) {
            for (var r = 0u; r < p.d_rel; r = r + 1u) {
                racc[r] = racc[r] + dz * aux[p.bank_off + r * p.extent + dist];
            }
        }
    }
    for (var d = 0u; d < hd; d = d + 1u) { dq[qb + d] = acc[d] * qs; }
    if (bias_on) {
        let rb = idx * p.d_rel;
        for (var r = 0u; r < p.d_rel; r = r + 1u) { d_rel[rb + r] = racc[r] * tb; }
    }
}
