// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal GQA online-softmax attention with optional sliding window, sink logits, in-tile distance-bank bias, log-position query scale and packed documents, over contiguous or paged/ring KV
// @how   one thread per (query row, head), serial key walk with an online softmax
// @opt   2
// @cpu   yes
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
// The reference tier of the extended causal attention: the forward every
// faster form (`flash_attn_gqa_ext_tiled`) is gated against, and the one the
// CPU JIT runs (no barrier). Memory is O(rows * head_dim): no [H,T,T] slab.
//
// For query row `i` of sequence `s` at ABSOLUTE position `P` and head `h`
// (kv head `h / group`), over keys at absolute positions `p`:
//
//   visible(p)  = p <= P  and  (window == 0 or P - p < window)  and  p >= kv_pos0
//   logit(p)    = tq(P) * scale * <q_i, k_p>  +  tb(P) * bias(i, h, P - p)
//   bias(i,h,d) = sum_r rel[i, h, r] * bank[r, d]   for d < extent, else 0 (zero, not masked)
//   out_i       = sum_p softmax([logit; sink(i,h)])_p * v_p     (the sink takes mass, carries value 0)
//   lse(i,h)    = log(sum_p exp(logit(p)) + exp(sink(i,h)))
//
// with `tq(P) = 1 + alpha * ln(max((P + 1) / n_floor, 1))` when the TAU flag is
// set (else 1) and `tb = tq` when TAU_BIAS is also set (else 1). Each option is
// a bit of `flags` and is off at 0: with flags 0 and window 0 this is plain
// causal GQA attention with the scale `scale`.
//
// `aux` packs the optional per-row inputs at word offsets: `sink` `[rows, n_heads]`
// at `sink_off`, `rel` `[rows, n_heads, d_rel]` at `rel_off`, `bank`
// `[d_rel, extent]` at `bank_off` (one binding: the storage-buffer budget is 8).
//
// Addressing. CONTIGUOUS (flag PAGED clear, the training layout): `n_seqs`
// sequences of `max_q` rows; q/out are `[n_seqs*max_q, n_heads*head_dim]` and
// k/v the same rows `[n_seqs*max_q, n_kv_heads*head_dim]`; row `i` of sequence
// `s` sits at absolute position `pos0 + i` and so does key `i`. PAGED: `index`
// holds `n_seqs` records `(q_row0, n_q, q_pos0, kv_pos0)` and then the block
// tables `[n_seqs, max_bt]`; query `i < n_q` of sequence `s` is q row
// `q_row0 + i` at position `q_pos0 + i`; the key at position `p` sits in
// logical slot `L = p - kv_pos0` (`L % ring` for a ring of `ring` rows), i.e.
// pool row `table[s][L / block_size] * block_size + L % block_size`. Distances
// are always absolute-position differences, so a chunked prefill, a ring and a
// decode step compute the same logits as one whole-sequence pass.
//
// DOCS (contiguous only, `pos0 == 0`): the row packs several documents and
// `index[row]` is that row's position inside its document (0 at a document's
// first row, +1 per row; clamped to the row index, so a sequence may open
// mid-document). A query sees only the keys of its own document, and its
// position - for the window, the distances and tau - is the in-document one,
// so every packed document computes exactly what it computes alone (the keys
// are walked in the same order).
//
// lse is `[rows, n_heads]` (row-major per query row), what the backward reads.

struct Params {
    n_seqs: u32,
    max_q: u32,        // query rows per sequence (contiguous), or the longest n_q (paged)
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 256
    block_size: u32,   // paged only
    max_bt: u32,       // paged only: block-table row stride
    ring: u32,         // paged only: 0 linear, else logical slot = (p - kv_pos0) % ring
    window: u32,       // 0: plain causal
    flags: u32,        // 1 SINK, 2 BIAS, 4 TAU, 8 TAU_BIAS, 16 PAGED, 32 DOCS (contiguous only)
    extent: u32,       // bias distances 0..extent
    d_rel: u32,        // <= 64
    sink_off: u32,
    rel_off: u32,
    bank_off: u32,
    pos0: u32,         // contiguous only: absolute position of row 0 (0 with DOCS)
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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
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
    // Contiguous key row of logical slot 0 of this query's keys: the sequence's first row, or with DOCS the first
    // row of the query's document (whose in-document position is then the query's position).
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

    var o: array<f32, 256>;
    for (var d = 0u; d < hd; d = d + 1u) { o[d] = 0.0; }
    var m = -3.4e38;
    var l = 0.0;
    for (var kp = lo; kp <= pos; kp = kp + 1u) {
        var krow = key_row0 + (kp - kv_pos0);
        if (paged) {
            var slot = kp - kv_pos0;
            if (p.ring > 0u) { slot = slot % p.ring; }
            krow = index[tables + slot / p.block_size] * p.block_size + slot % p.block_size;
        }
        let kb = krow * kvd + hkv * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let ki = kb + d;
            dot = dot + q[qb + d] * pool_k[ki];
        }
        var logit = qs * dot;
        let dist = pos - kp;
        if (bias_on && dist < p.extent) {
            var b = 0.0;
            for (var r = 0u; r < p.d_rel; r = r + 1u) {
                b = b + aux[rel_b + r] * aux[p.bank_off + r * p.extent + dist];
            }
            logit = logit + tb * b;
        }
        let m_new = max(m, logit);
        let corr = exp(m - m_new);
        let e = exp(logit - m_new);
        l = l * corr + e;
        for (var d = 0u; d < hd; d = d + 1u) {
            let vi = kb + d;
            o[d] = o[d] * corr + e * pool_v[vi];
        }
        m = m_new;
    }
    if ((p.flags & 1u) != 0u) {
        let sk = aux[p.sink_off + row * nh + h];
        let m_new = max(m, sk);
        let corr = exp(m - m_new);
        l = l * corr + exp(sk - m_new);
        for (var d = 0u; d < hd; d = d + 1u) { o[d] = o[d] * corr; }
        m = m_new;
    }
    let inv = 1.0 / l;
    for (var d = 0u; d < hd; d = d + 1u) { out[qb + d] = o[d] * inv; }
    lse[row * nh + h] = m + log(l);
}
