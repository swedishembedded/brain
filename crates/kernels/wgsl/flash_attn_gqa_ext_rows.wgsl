// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal GQA online-softmax attention with window, sinks, distance-bank bias, log-position scale and documents, one WORKGROUP per (query row, head): flash_attn_gqa_ext's cooperative twin
// @how   128-thread workgroup; keys in 128-key tiles at absolute positions (one logit per lane), tile max/sum by tree, the value mix coalesced over head_dim
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements memory-bounded attention kernels for
// long-context models for its clients. If your team needs expertise in
// attention numerics on CPUs and GPUs then you can procure our services by
// sending an email to info@swedishembedded.com.
//
// Same Params, bindings, addressing (contiguous, paged, ring, documents) and outputs as flash_attn_gqa_ext - read its
// header for the math. The reference gives one THREAD a (query row, head) and walks its keys serially with a
// head_dim-wide accumulator in private memory: a decode step of a few sequences, or a prompt of tens of rows, runs a
// few hundred threads. Here a workgroup owns the pair: each lane scores one key of a 128-key tile (the q row and the
// bias coefficients staged in workgroup memory), the tile's maximum and exp-sum fold by tree, and each lane mixes the
// values of its own head_dim columns (coalesced across lanes; the dot products, the bias and the value mix accumulate
// by fused multiply-adds). The online softmax rescales once per tile instead of
// once per key, so the sums reassociate: held to the reference within a tolerance (`model` tests `flash_ext`).
//
// The tiles are the 128 positions `[128 c, 128 c + 128)` (absolute positions, a key below the row's window masked), so a
// row's arithmetic is a function of its position alone, and flash_attn_gqa_ext_tiled (the prompt-chunk form) computes
// the same bits per row: change the tiling, a score, the tree or an update order here and there together (`model` tests
// `flash_ext` hold the two bit for bit).
//
// Dispatch: one workgroup per (sequence, query row, head) - `n_seqs * max_q * n_heads` workgroups, the reference's
// thread count - 2D-grid safe. A workgroup past its sequence's `n_q` returns whole.

const WG: u32 = 128u;
const FLOOR: f32 = -3.4e38;

struct Params {
    n_seqs: u32,
    max_q: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 256
    block_size: u32,
    max_bt: u32,
    ring: u32,
    window: u32,
    flags: u32,        // 1 SINK, 2 BIAS, 4 TAU, 8 TAU_BIAS, 16 PAGED, 32 DOCS (contiguous only)
    extent: u32,
    d_rel: u32,        // <= 64
    sink_off: u32,
    rel_off: u32,
    bank_off: u32,
    pos0: u32,
    scale: f32,
    tau_alpha: f32,
    tau_n_floor: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k: array<f32>;
@group(0) @binding(3) var<storage, read>       pool_v: array<f32>;
@group(0) @binding(4) var<storage, read>       index:  array<u32>;
@group(0) @binding(5) var<storage, read>       aux:    array<f32>;
@group(0) @binding(6) var<storage, read_write> out:    array<f32>;
@group(0) @binding(7) var<storage, read_write> lse:    array<f32>;

var<workgroup> q_s: array<f32, 256>;
var<workgroup> rel_s: array<f32, 64>;
var<workgroup> red: array<f32, WG>;
var<workgroup> prob: array<f32, WG>;
var<workgroup> kbase: array<u32, WG>;

// Fold `red` to red[0] by maximum (`is_max`) or sum. Called by every lane; red[0] is valid on return.
fn fold(lane: u32, is_max: bool) {
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            if (is_max) {
                red[lane] = max(red[lane], red[lane + s]);
            } else {
                red[lane] = red[lane] + red[lane + s];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Everything up to the key loop is uniform across the workgroup, so the returns keep every barrier uniform.
    let idx = wg.y * nwg.x + wg.x;
    let lane = li.x;
    let nh = p.n_heads;
    if (idx >= p.n_seqs * p.max_q * nh) { return; }
    let h = idx % nh;
    let i = (idx / nh) % p.max_q;
    let s = idx / (nh * p.max_q);

    let paged = (p.flags & 16u) != 0u;
    var q_row0 = s * p.max_q;
    var n_q = p.max_q;
    var q_pos0 = p.pos0;
    var kv_pos0 = p.pos0;
    if (paged) {
        q_row0 = index[s * 4u];
        n_q = index[s * 4u + 1u];
        q_pos0 = index[s * 4u + 2u];
        kv_pos0 = index[s * 4u + 3u];
    }
    if (i >= n_q) { return; }
    let row = q_row0 + i;
    var pos = q_pos0 + i;
    var key_row0 = s * p.max_q;
    if (!paged && (p.flags & 32u) != 0u) {
        let dp = min(index[row], i);
        key_row0 = key_row0 + i - dp;
        pos = dp;
    }

    var lo = kv_pos0;
    if (p.window > 0u && pos + 1u > p.window) { lo = max(lo, pos + 1u - p.window); }

    let hd = p.head_dim;
    let hkv = h / (nh / p.n_kv_heads);
    let kvd = p.n_kv_heads * hd;
    let qb = (row * nh + h) * hd;
    var tq = 1.0;
    var tb = 1.0;
    if ((p.flags & 4u) != 0u) {
        tq = 1.0 + p.tau_alpha * log(max(f32(pos + 1u) / p.tau_n_floor, 1.0));
        if ((p.flags & 8u) != 0u) { tb = tq; }
    }
    let qs = p.scale * tq;
    let bias_on = (p.flags & 2u) != 0u;
    let rel_b = p.rel_off + (row * nh + h) * p.d_rel;
    let tables = p.n_seqs * 4u + s * p.max_bt;

    for (var d = lane; d < hd; d = d + WG) { q_s[d] = q[qb + d]; }
    if (bias_on && lane < p.d_rel) { rel_s[lane] = aux[rel_b + lane]; }

    // This lane's two head_dim columns of the output.
    var o0 = 0.0;
    var o1 = 0.0;
    var m = FLOOR;
    var l = 0.0;
    for (var t0 = lo - lo % WG; t0 <= pos; t0 = t0 + WG) {
        workgroupBarrier();
        // Score one key of the tile.
        let kp = t0 + lane;
        let reads = kp <= pos && kp >= lo;
        var logit = FLOOR;
        var kb = 0u;
        if (reads) {
            var krow = key_row0 + (kp - kv_pos0);
            if (paged) {
                var slot = kp - kv_pos0;
                if (p.ring > 0u) { slot = slot % p.ring; }
                krow = index[tables + slot / p.block_size] * p.block_size + slot % p.block_size;
            }
            kb = krow * kvd + hkv * hd;
            var dot = 0.0;
            for (var d = 0u; d < hd; d = d + 1u) {
                let ki = kb + d;
                dot = fma(q_s[d], pool_k[ki], dot);
            }
            logit = qs * dot;
            let dist = pos - kp;
            if (bias_on && dist < p.extent) {
                var b = 0.0;
                for (var r = 0u; r < p.d_rel; r = r + 1u) {
                    b = fma(rel_s[r], aux[p.bank_off + r * p.extent + dist], b);
                }
                logit = logit + tb * b;
            }
        }
        kbase[lane] = kb;
        red[lane] = logit;
        fold(lane, true);
        let m_new = max(m, red[0]);
        workgroupBarrier();
        var e = 0.0;
        if (reads) { e = exp(logit - m_new); }
        prob[lane] = e;
        red[lane] = e;
        fold(lane, false);
        let corr = exp(m - m_new);
        l = l * corr + red[0];
        m = m_new;
        // Mix the tile's values (the row's keys, ascending) into this lane's columns.
        let n_keys = min(WG, pos + 1u - t0);
        var j_lo = 0u;
        if (lo > t0) { j_lo = lo - t0; }
        o0 = o0 * corr;
        o1 = o1 * corr;
        for (var j = j_lo; j < n_keys; j = j + 1u) {
            let pj = prob[j];
            let vb = kbase[j];
            if (lane < hd) {
                let vi = vb + lane;
                o0 = fma(pj, pool_v[vi], o0);
            }
            if (lane + WG < hd) {
                let vi = vb + lane + WG;
                o1 = fma(pj, pool_v[vi], o1);
            }
        }
    }
    if ((p.flags & 1u) != 0u) {
        let sk = aux[p.sink_off + row * nh + h];
        let m_new = max(m, sk);
        let corr = exp(m - m_new);
        l = l * corr + exp(sk - m_new);
        o0 = o0 * corr;
        o1 = o1 * corr;
        m = m_new;
    }
    let inv = 1.0 / l;
    if (lane < hd) { out[qb + lane] = o0 * inv; }
    if (lane + WG < hd) { out[qb + lane + WG] = o1 * inv; }
    if (lane == 0u) { lse[row * nh + h] = m + log(l); }
}
