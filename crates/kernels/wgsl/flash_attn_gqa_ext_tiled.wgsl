// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Causal GQA online-softmax attention with window, sinks, distance-bank bias and log-position scale, a block of 32 query rows of one head per WORKGROUP: flash_attn_gqa_ext_rows's query-tiled twin for prompt chunks, the same bits per row
// @how   128 threads; keys in 128-key tiles at absolute positions, staged once per tile for the 32 rows (K by 16-column slices, V by 16-key slices); 8x4 register tiles for the scores and the value mix, the rows broadcast as vec4; per-row tile max and the rows twin's exp-sum tree
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
// Same Params, bindings, addressing (contiguous or paged/ring) and outputs as flash_attn_gqa_ext - read its header for
// the math - for `head_dim <= 128`, `d_rel <= 16` and without packed documents (the per-row twin serves those). The
// per-row form reads every key once per (query row, head): at a prompt chunk of hundreds of rows against thousands of
// keys that read is the whole cost. Here a workgroup owns 32 consecutive query rows of one sequence and one head, and
// each tile of keys and values is loaded into workgroup memory once for all 32. A thread holds 8 rows by 4 keys of a
// tile's scores (keys `kl + 32 i`) and 8 rows by 4 columns of the output, so each operand it reads from workgroup memory
// feeds 4 or 8 products, and the rows' query words are read four at a time, the same for every lane.
//
// BIT-IDENTICAL to flash_attn_gqa_ext_rows, row by row: a serving engine prefills a prompt in chunks of any size (this
// kernel) and decodes it a row at a time (the rows twin), and every position must compute the bits the whole-sequence
// pass computes. So, for each (row, head), exactly the rows twin's arithmetic in the same order:
// - the key tiles are the 128 positions `[128 c, 128 c + 128)` (absolute positions, never relative to a chunk's start);
//   a key outside the row's causal range or window scores nothing and adds an exact zero to the tile's sum;
// - a score is `scale * tau(pos)` times the dot product summed over head_dim ascending, plus `tau_b` times the bias
//   summed over d_rel ascending (each step one fused multiply-add) (columns past head_dim are staged as zeros: adding the +0 or -0 their product gives
//   leaves a sum that started at +0 exactly as it was);
// - the tile maximum (exact in any order), then the exp-sum folded by the rows twin's tree (lane `j` is the key at
//   `t0 + j`; strides 64, 32, ..., 1: the first two strides join a thread's own four keys `kl + 32 i`, the rest run in
//   workgroup memory), then `l = l * corr + sum`;
// - the output scaled by `corr`, then the values added key by key (fused multiply-adds), ascending, for the row's keys
//   only;
// - the sink, then `out = o * (1 / l)` and `lse = m + log(l)`.
// A tile wholly outside a row's keys leaves its state as it was (corr is exactly 1, the sum exactly 0), so the rows a
// block carries beside a row never change that row's bits. `model` tests `flash_ext` hold this kernel to the rows twin
// bit for bit across chunkings.
//
// Workgroup memory is 49,024 bytes (the most a CUDA block declares statically is 49,152): the caller dispatches it only where the device reports that much
// (`DeviceCaps::workgroup_mem_bytes`).
//
// Dispatch: one workgroup per (sequence, 32-row query block, head) - `n_seqs * ceil(max_q / 32) * n_heads` workgroups,
// head fastest - 2D-grid safe. Rows past a sequence's `n_q` compute nothing and store nothing.

const WG: u32 = 128u;
const QB: u32 = 32u;      // query rows a workgroup
const RT: u32 = 8u;       // rows a thread
const KT: u32 = 128u;     // keys a tile
const HD: u32 = 128u;
const DREL: u32 = 16u;
const SL: u32 = 16u;      // head_dim columns of a K slice / keys of a V slice
const KSTRIDE: u32 = 17u; // a staged K row, padded so the 32 keys a warp reads sit in 32 banks
const BSTRIDE: u32 = 160u; // a staged bank column: the QB + KT - 1 distances a tile's pairs read
const FLOOR: f32 = -3.4e38;

struct Params {
    n_seqs: u32,
    max_q: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 128 here
    block_size: u32,
    max_bt: u32,
    ring: u32,
    window: u32,
    flags: u32,        // 1 SINK, 2 BIAS, 4 TAU, 8 TAU_BIAS, 16 PAGED (32 DOCS is not served here)
    extent: u32,
    d_rel: u32,        // <= 16 here
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

var<workgroup> q_s: array<vec4<f32>, 1024>;  // [QB][HD / 4] the block's query rows
var<workgroup> p_s: array<f32, 4096>;        // [DREL][BSTRIDE] the tile's bank columns, then [QB][KT] its probabilities (-1: not one of the row's keys)
var<workgroup> kv_s: array<f32, 2176>;       // [KT][KSTRIDE] a K slice | [SL][HD] a V slice
var<workgroup> part: array<f32, 1024>;       // [QB][32] partial maxima, then the exp-sum tree from stride 16 down
var<workgroup> rel_s: array<f32, 512>;       // [QB][DREL] the rows' bias coefficients
var<workgroup> kbase: array<u32, 128>;       // [KT] element offset of each staged key's head
var<workgroup> row_m: array<f32, 32>;
var<workgroup> row_l: array<f32, 32>;
var<workgroup> row_corr: array<f32, 32>;
var<workgroup> row_pos: array<u32, 32>;
var<workgroup> row_lo: array<u32, 32>;
var<workgroup> row_qs: array<f32, 32>;
var<workgroup> row_tb: array<f32, 32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Everything before the key loop is uniform across the workgroup, so the returns keep every barrier uniform.
    let idx = wg.y * nwg.x + wg.x;
    let t = li.x;
    let nh = p.n_heads;
    let blocks = (p.max_q + QB - 1u) / QB;
    if (idx >= p.n_seqs * blocks * nh) { return; }
    let h = idx % nh;
    let qb = (idx / nh) % blocks;
    let s = idx / (nh * blocks);

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
    let i0 = qb * QB;
    if (i0 >= n_q) { return; }
    let rows = min(QB, n_q - i0);
    let hd = p.head_dim;
    let hkv = h / (nh / p.n_kv_heads);
    let kvd = p.n_kv_heads * hd;
    let tables = p.n_seqs * 4u + s * p.max_bt;
    let bias_on = (p.flags & 2u) != 0u;
    let key_row0 = s * p.max_q;

    // The block's keys: from the first row's lowest key to the last row's own position (lo is nondecreasing in pos).
    let pos_first = q_pos0 + i0;
    let pos_last = q_pos0 + i0 + rows - 1u;
    var blk_lo = kv_pos0;
    if (p.window > 0u && pos_first + 1u > p.window) { blk_lo = max(blk_lo, pos_first + 1u - p.window); }

    // Stage the block's query rows and bias coefficients (zero past n_q and past head_dim / d_rel), and each row's
    // position, lowest key, query scale and bias scale.
    for (var e = t; e < QB * HD / 4u; e = e + WG) {
        let r = e / (HD / 4u);
        let d = (e % (HD / 4u)) * 4u;
        var v = vec4<f32>(0.0);
        if (r < rows) {
            let qr = ((q_row0 + i0 + r) * nh + h) * hd;
            if (d < hd) { v.x = q[qr + d]; }
            if (d + 1u < hd) { v.y = q[qr + d + 1u]; }
            if (d + 2u < hd) { v.z = q[qr + d + 2u]; }
            if (d + 3u < hd) { v.w = q[qr + d + 3u]; }
        }
        q_s[e] = v;
    }
    if (bias_on) {
        for (var e = t; e < QB * DREL; e = e + WG) {
            let r = e / DREL;
            let j = e % DREL;
            var v = 0.0;
            if (r < rows && j < p.d_rel) { v = aux[p.rel_off + ((q_row0 + i0 + r) * nh + h) * p.d_rel + j]; }
            rel_s[e] = v;
        }
    }
    if (t < QB) {
        let pos = q_pos0 + i0 + t;
        var lo = kv_pos0;
        if (p.window > 0u && pos + 1u > p.window) { lo = max(lo, pos + 1u - p.window); }
        var tq = 1.0;
        var tb = 1.0;
        if ((p.flags & 4u) != 0u) {
            tq = 1.0 + p.tau_alpha * log(max(f32(pos + 1u) / p.tau_n_floor, 1.0));
            if ((p.flags & 8u) != 0u) { tb = tq; }
        }
        row_pos[t] = pos;
        row_lo[t] = lo;
        row_qs[t] = p.scale * tq;
        row_tb[t] = tb;
        row_m[t] = FLOOR;
        row_l[t] = 0.0;
    }

    // This thread's eight rows (scores and outputs alike), its four keys `kl + 32 i` of a tile and its four output
    // columns `kl + 32 i`.
    let rg = (t / 32u) * RT;
    let kl = t % 32u;
    var o: array<f32, 32>;   // [row][column]
    for (var c = 0u; c < 32u; c = c + 1u) { o[c] = 0.0; }

    for (var t0 = blk_lo - blk_lo % KT; t0 <= pos_last; t0 = t0 + KT) {
        workgroupBarrier();
        // The pool offset of each key of the tile the block reads; the others are never read.
        let kp_t = t0 + t;
        var kb = 0u;
        if (kp_t >= blk_lo && kp_t <= pos_last) {
            var krow = key_row0 + (kp_t - kv_pos0);
            if (paged) {
                var slot = kp_t - kv_pos0;
                if (p.ring > 0u) { slot = slot % p.ring; }
                krow = index[tables + slot / p.block_size] * p.block_size + slot % p.block_size;
            }
            kb = krow * kvd + hkv * hd;
        }
        kbase[t] = kb;
        // The bank columns the tile's (row, key) pairs read, staged in p_s until the probabilities replace them: pair
        // (row, j) reads distance `pos_first + row - (t0 + j)` at slot `row - j + KT - 1`.
        if (bias_on) {
            for (var e = t; e < DREL * BSTRIDE; e = e + WG) {
                let x = e / BSTRIDE;
                let slot = e % BSTRIDE;
                var bv = 0.0;
                let at = pos_first + slot;          // the distance plus t0 + KT - 1
                if (x < p.d_rel && slot < QB + KT - 1u && at >= t0 + KT - 1u && at - (t0 + KT - 1u) < p.extent) {
                    bv = aux[p.bank_off + x * p.extent + at - (t0 + KT - 1u)];
                }
                p_s[e] = bv;
            }
        }

        // Scores: dot products over head_dim ascending, a 16-column K slice at a time.
        var acc: array<f32, 32>;   // [row][key]
        for (var c = 0u; c < 32u; c = c + 1u) { acc[c] = 0.0; }
        for (var d0 = 0u; d0 < hd; d0 = d0 + SL) {
            workgroupBarrier();
            for (var e = t; e < KT * SL; e = e + WG) {
                let j = e / SL;
                let dd = e % SL;
                let kp = t0 + j;
                var kv = 0.0;
                if (kp >= blk_lo && kp <= pos_last && d0 + dd < hd) {
                    let ki = kbase[j] + d0 + dd;
                    kv = pool_k[ki];
                }
                kv_s[j * KSTRIDE + dd] = kv;
            }
            workgroupBarrier();
            for (var dq = 0u; dq < SL; dq = dq + 4u) {
                var qv: array<vec4<f32>, 8>;
                for (var r = 0u; r < RT; r = r + 1u) { qv[r] = q_s[(rg + r) * (HD / 4u) + (d0 + dq) / 4u]; }
                for (var c = 0u; c < 4u; c = c + 1u) {
                    var kr: array<f32, 4>;
                    for (var i = 0u; i < 4u; i = i + 1u) { kr[i] = kv_s[(kl + 32u * i) * KSTRIDE + dq + c]; }
                    for (var r = 0u; r < RT; r = r + 1u) {
                        let qc = qv[r][c];
                        for (var i = 0u; i < 4u; i = i + 1u) {
                            acc[r * 4u + i] = fma(qc, kr[i], acc[r * 4u + i]);
                        }
                    }
                }
            }
        }
        // Logits (the bias added per key) for the row's keys only, and this thread's partial maximum of each row.
        for (var r = 0u; r < RT; r = r + 1u) {
            let row = rg + r;
            let pos = row_pos[row];
            let lo = row_lo[row];
            var pm = FLOOR;
            for (var i = 0u; i < 4u; i = i + 1u) {
                let kp = t0 + kl + 32u * i;
                var logit = FLOOR;
                if (row < rows && kp <= pos && kp >= lo) {
                    logit = row_qs[row] * acc[r * 4u + i];
                    let dist = pos - kp;
                    if (bias_on && dist < p.extent) {
                        var b = 0.0;
                        let slot = row + KT - 1u - (kl + 32u * i);
                        for (var x = 0u; x < p.d_rel; x = x + 1u) {
                            b = fma(rel_s[row * DREL + x], p_s[x * BSTRIDE + slot], b);
                        }
                        logit = logit + row_tb[row] * b;
                    }
                }
                acc[r * 4u + i] = logit;
                pm = max(pm, logit);
            }
            part[row * 32u + kl] = pm;
        }
        workgroupBarrier();
        // Each row's new running maximum (the tile maximum is exact in any order) and its rescale.
        if (t < QB) {
            var mt = FLOOR;
            for (var x = 0u; x < 32u; x = x + 1u) { mt = max(mt, part[t * 32u + x]); }
            let m_new = max(row_m[t], mt);
            row_corr[t] = exp(row_m[t] - m_new);
            row_m[t] = m_new;
        }
        workgroupBarrier();
        // Probabilities into p_s (-1 marks a key the row does not read); the tree's strides 64 and 32 join this
        // thread's own keys (lane kl + 64 into kl, kl + 96 into kl + 32, then kl + 32 into kl).
        for (var r = 0u; r < RT; r = r + 1u) {
            let row = rg + r;
            let pos = row_pos[row];
            let lo = row_lo[row];
            let m_new = row_m[row];
            var e: array<f32, 4>;
            for (var i = 0u; i < 4u; i = i + 1u) {
                let kp = t0 + kl + 32u * i;
                var pr = -1.0;
                e[i] = 0.0;
                if (row < rows && kp <= pos && kp >= lo) {
                    e[i] = exp(acc[r * 4u + i] - m_new);
                    pr = e[i];
                }
                acc[r * 4u + i] = pr;
            }
            part[row * 32u + kl] = (e[0] + e[2]) + (e[1] + e[3]);
        }
        for (var r = 0u; r < RT; r = r + 1u) {
            for (var i = 0u; i < 4u; i = i + 1u) { p_s[(rg + r) * KT + kl + 32u * i] = acc[r * 4u + i]; }
        }
        // The rest of the rows twin's tree, every row at once: lane l += lane l + stride, strides 16 down to 1.
        for (var st = 16u; st > 0u; st = st >> 1u) {
            workgroupBarrier();
            if (kl < st) {
                for (var r = 0u; r < RT; r = r + 1u) {
                    let row = rg + r;
                    part[row * 32u + kl] = part[row * 32u + kl] + part[row * 32u + kl + st];
                }
            }
        }
        workgroupBarrier();
        if (t < QB) {
            row_l[t] = row_l[t] * row_corr[t] + part[t * 32u];
        }
        // The value mix: rescale, then the row's keys ascending, a 16-key V slice at a time.
        for (var r = 0u; r < RT; r = r + 1u) {
            let corr = row_corr[rg + r];
            for (var i = 0u; i < 4u; i = i + 1u) { o[r * 4u + i] = o[r * 4u + i] * corr; }
        }
        for (var j0 = 0u; j0 < KT; j0 = j0 + SL) {
            workgroupBarrier();
            for (var e = t; e < SL * HD; e = e + WG) {
                let j = e / HD;
                let d = e % HD;
                let kp = t0 + j0 + j;
                var vv = 0.0;
                if (kp >= blk_lo && kp <= pos_last && d < hd) {
                    let vi = kbase[j0 + j] + d;
                    vv = pool_v[vi];
                }
                kv_s[e] = vv;
            }
            workgroupBarrier();
            for (var j = 0u; j < SL; j = j + 1u) {
                var vr: array<f32, 4>;
                for (var i = 0u; i < 4u; i = i + 1u) { vr[i] = kv_s[j * HD + kl + 32u * i]; }
                for (var r = 0u; r < RT; r = r + 1u) {
                    let pj = p_s[(rg + r) * KT + j0 + j];
                    if (pj >= 0.0) {
                        for (var i = 0u; i < 4u; i = i + 1u) {
                            o[r * 4u + i] = fma(pj, vr[i], o[r * 4u + i]);
                        }
                    }
                }
            }
        }
    }
    // The sink, then the rows' outputs.
    workgroupBarrier();
    if (t < QB) {
        var m = row_m[t];
        var l = row_l[t];
        var corr = 1.0;
        if ((p.flags & 1u) != 0u && t < rows) {
            let sk = aux[p.sink_off + (q_row0 + i0 + t) * nh + h];
            let m_new = max(m, sk);
            corr = exp(m - m_new);
            l = l * corr + exp(sk - m_new);
            m = m_new;
        }
        row_corr[t] = corr;
        row_l[t] = l;
        row_m[t] = m;
    }
    workgroupBarrier();
    for (var r = 0u; r < RT; r = r + 1u) {
        let row = rg + r;
        if (row < rows) {
            let qrow = q_row0 + i0 + row;
            let corr = row_corr[row];
            let l = row_l[row];
            let inv = 1.0 / l;
            let ob = (qrow * nh + h) * hd;
            for (var i = 0u; i < 4u; i = i + 1u) {
                let col = kl + 32u * i;
                if (col < hd) {
                    var oc = o[r * 4u + i];
                    if ((p.flags & 1u) != 0u) { oc = oc * corr; }
                    out[ob + col] = oc * inv;
                }
            }
            if (kl == 0u) { lse[qrow * nh + h] = row_m[row] + log(l); }
        }
    }
}
