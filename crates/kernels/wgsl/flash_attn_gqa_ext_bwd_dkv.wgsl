// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Extended causal GQA attention backward, dK and dV: one thread per key row and kv head, summing every query head of its group
// @how   one thread per (key row, kv head), serial query walk
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
// The key side of the backward of `flash_attn_gqa_ext.wgsl` (contiguous layout;
// `stats` from `flash_attn_gqa_ext_bwd_prep`). With `P_ij`, `dZ_ij` as in
// `flash_attn_gqa_ext_bwd_dq.wgsl`:
//
//   dV_j = sum_h sum_i P_ij * dO_i        dK_j = sum_h sum_i dZ_ij * tq_i * scale * q_i
//
// over the queries that saw key `j` (`j <= i`, inside the window, and with DOCS
// the queries of `j`'s own document: a later document starts past `j`) and the
// query heads `h` of kv head `hkv`: every dK/dV element is written once, no atomics.

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
@group(0) @binding(1) var<storage, read>       q:     array<f32>;
@group(0) @binding(2) var<storage, read>       k:     array<f32>;
@group(0) @binding(3) var<storage, read>       v:     array<f32>;
@group(0) @binding(4) var<storage, read>       d_o:   array<f32>;
@group(0) @binding(5) var<storage, read>       stats: array<f32>;
@group(0) @binding(6) var<storage, read>       aux:   array<f32>;
@group(0) @binding(7) var<storage, read_write> dk:    array<f32>;
@group(0) @binding(8) var<storage, read_write> dv:    array<f32>;

// The first row of query row `row`'s document, relative to its sequence (0 without DOCS): what
// `flash_attn_gqa_ext_bwd_prep` left past the per-(row, head) statistics.
fn doc_start(row: u32) -> u32 {
    if ((p.flags & 32u) == 0u) { return 0u; }
    return u32(stats[2u * p.n_seqs * p.max_q * p.n_heads + row]);
}

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
    let nkv = p.n_kv_heads;
    let T = p.max_q;
    if (idx >= p.n_seqs * T * nkv) { return; }
    let hkv = idx % nkv;
    let krow = idx / nkv;
    let j = krow % T;
    let b = krow / T;
    let hd = p.head_dim;
    let nh = p.n_heads;
    let group = nh / nkv;
    let kb = idx * hd;

    var hi = T - 1u;
    if (p.window > 0u) { hi = min(hi, j + p.window - 1u); }

    var dk_acc: array<f32, 256>;
    var dv_acc: array<f32, 256>;
    for (var d = 0u; d < hd; d = d + 1u) { dk_acc[d] = 0.0; dv_acc[d] = 0.0; }

    for (var g = 0u; g < group; g = g + 1u) {
        let h = hkv * group + g;
        for (var i = j; i <= hi; i = i + 1u) {
            let row = b * T + i;
            let ds = doc_start(row);
            if (ds > j) { break; }
            let pos = p.pos0 + i - ds;
            let qb = (row * nh + h) * hd;
            let st = 2u * (row * nh + h);
            var dot = 0.0;
            var dp = 0.0;
            for (var d = 0u; d < hd; d = d + 1u) {
                dot = dot + q[qb + d] * k[kb + d];
                dp = dp + d_o[qb + d] * v[kb + d];
            }
            let qs = p.scale * tau_q(pos);
            let z = qs * dot + tau_b(pos) * bias_at(row, h, i - j);
            let pw = exp(z - stats[st]);
            let dz = pw * (dp - stats[st + 1u]);
            for (var d = 0u; d < hd; d = d + 1u) {
                dv_acc[d] = dv_acc[d] + pw * d_o[qb + d];
                dk_acc[d] = dk_acc[d] + dz * qs * q[qb + d];
            }
        }
    }
    for (var d = 0u; d < hd; d = d + 1u) {
        dk[kb + d] = dk_acc[d];
        dv[kb + d] = dv_acc[d];
    }
}
