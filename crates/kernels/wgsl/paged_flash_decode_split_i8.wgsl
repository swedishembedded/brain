// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched GQA split-key FlashDecode over an INT8 paged KV pool: one workgroup per (sequence, kv head, 4-head slab, key split) attends every query head of the slab, so each K/V byte is read once per slab, and writes a partial online-softmax state
// @how   128-thread workgroup over 128-key chunks, a lane-per-key score pass (16-byte row loads) and a lane-per-word value pass (coalesced 4-byte loads), 8 barriers (5 per chunk)
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Swedish Embedded AB implements long-context inference serving for clients
// whose GPU decides how many users one card carries. If your team needs
// expertise in decode attention that runs at the memory roofline over a
// quantized KV cache, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// WHAT IT COMPUTES. For every live sequence `b`, query head `h` and its kv
// head `h / group`: `softmax(q . K^T / sqrt(head_dim)) V` over the sequence's
// `seq_lens[b]` cached tokens, where K and V are int8 with one f32 scale per
// (token slot, kv head). The keys are cut into splits; this kernel writes one
// PARTIAL state per (sequence, head, split) and `paged_flash_decode_combine_i8`
// merges them.
//
// WORK SCALES WITH THE LIVE LENGTH, NOT WITH CAPACITY. Nothing here reads the
// block table's capacity: a sequence of `len` tokens has `ceil(len / 128)`
// chunks, split `s` of `n_splits` walks the contiguous chunks
// `[s * cps, (s + 1) * cps)` with `cps = ceil(chunks / n_splits)`, and a
// workgroup whose first chunk lies past the end returns before touching
// memory. `n_splits` is therefore only an UPPER bound on the parallelism a
// short sequence gets, which lets the host choose it from the batch size
// alone (a recorded dispatch list cannot see sequence lengths): few splits
// when many sequences already fill the device, many when one sequence must.
//
// GQA AND THE SLAB. A workgroup scores 4 query heads against each key it
// loads (`SLAB`), so a kv head serving `group` query heads is read
// `ceil(group / 4)` times instead of `group` times (once for Qwen3's groups
// of 2 and 4). Slab `k` covers heads `4k .. 4k + 3` of the group; heads past
// `group` are computed on a zero query and never stored.
//
// THE THREAD MAP (128 threads, 128-key chunks).
//   scores  - lane `t` owns key `t` of the chunk and dots its whole row with
//             all four heads: `head_dim / 16` 16-byte loads of the key row,
//             unpacked to f32 and fused-multiply-added against queries held in
//             workgroup memory (every lane reads the same address, a
//             broadcast). Rows of different keys lie `n_kv_heads * head_dim`
//             bytes apart, but each is a whole 128-byte line used by one lane
//             across the loop, so DRAM sees full lines. The key's block-table
//             entry and both dequant scales are read once per key here; the
//             K scale folds into the score, the V scale into the probability.
//   softmax - the chunk max per head is a two-level reduction (lane `t` of
//             head `t / 32` scans four keys, then one lane per head folds the
//             32 partials), the running max and the accumulator rescale
//             follow, and lane `t` turns its key's four scores into
//             probabilities.
//   values  - lane `t` owns the packed word `t % 32` of the value row
//             (channels `4w .. 4w + 3`, coalesced across the warp) for the
//             keys `t / 32 + 4i`: one 4-byte load, one unpack and sixteen FMAs
//             into registers per key. The four key groups are summed once at
//             the end.
// The softmax denominator is accumulated alongside the values from the
// UNSCALED probabilities (the lane's own partial, summed at the end), so no
// extra reduction tree is needed.
//
// MEMORY. qs 2 KiB + sc 2 KiB + pe/pv 4 KiB + fin 2 KiB + red/rowb 1 KiB +
// small ~= 11.2 KiB, under WebGPU's 16 KiB floor. 8 storage buffers (q, pool_k,
// pool_v, scales_k, scales_v, block_tables, seq_lens, part) - exactly the
// guaranteed floor. `pool_k` is bound as 16-byte vectors, so `head_dim` must
// be a multiple of 16 and at most 128 (the host checks).
//
//   q             : [batch, n_heads * head_dim]                      (f32)
//   pool_k/pool_v : [num_blocks * block_size * n_kv_heads * head_dim / 4] (u32, 4 int8 per word)
//   scales_k/v    : [num_blocks * block_size, n_kv_heads]            (f32)
//   block_tables  : [batch, max_bt]
//   seq_lens      : [batch]   live tokens per sequence, the new token included
//   part          : [batch, n_heads, n_splits, 2 + 128]  (m, l, then the UNNORMALISED accumulator)
//
// A split with no live chunk writes nothing; `paged_flash_decode_combine_i8`
// derives the live split count from `seq_lens` with the same arithmetic.
// NOT bit-identical to the three-kernel triad (a different reduction order);
// gated against it at 1e-3 absolute on unit-variance inputs.

const HD_MAX: u32 = 128u;
const CHUNK: u32 = 128u;          // keys per chunk, and threads per workgroup
const SLAB: u32 = 4u;             // query heads attended per workgroup
const PART_STRIDE: u32 = 2u + HD_MAX; // one partial-state entry: [m, l, o_0..o_127]

struct Params {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // multiple of 16, <= 128
    group: u32,        // n_heads / n_kv_heads
    block_size: u32,
    max_bt: u32,       // block_tables row stride (blocks per sequence)
    n_splits: u32,     // dispatched splits per (sequence, kv head, slab)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<vec4<u32>>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<u32>;
@group(0) @binding(4) var<storage, read>       scales_k:     array<f32>;
@group(0) @binding(5) var<storage, read>       scales_v:     array<f32>;
@group(0) @binding(6) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(7) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(8) var<storage, read_write> part_out:     array<f32>;

var<workgroup> qs:   array<vec4<f32>, 128>;  // SLAB heads x 32 four-channel pieces
var<workgroup> sc:   array<f32, 512>;        // [head][key] scores of the chunk
var<workgroup> red:  array<f32, 128>;        // per-(head, lane) partial maxima
var<workgroup> rowb: array<u32, 128>;        // word index of each chunk key's value row
var<workgroup> pe:   array<vec4<f32>, 128>;  // [key] probabilities of the 4 heads
var<workgroup> pv:   array<vec4<f32>, 128>;  // [key] probabilities x the key's V scale
var<workgroup> mrun: array<f32, 4>;          // running max per head
var<workgroup> corr: array<f32, 4>;          // this chunk's accumulator rescale per head
var<workgroup> fin:  array<f32, 512>;        // [head][channel] cross-key-group reduction
var<workgroup> lfin: array<f32, 16>;         // [key group][head] denominator partials

// Four signed bytes of a packed word as f32, lowest byte first.
fn unpack_i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(vec4<i32>(
        bitcast<i32>(w << 24u) >> 24u,
        bitcast<i32>(w << 16u) >> 24u,
        bitcast<i32>(w << 8u) >> 24u,
        bitcast<i32>(w) >> 24u));
}

// `acc + a . k`, as explicit fused multiply-adds (the CUDA build compiles
// with contraction off, so `a * b + c` would be two instructions).
fn fma_dot4(a: vec4<f32>, k: vec4<f32>, acc: f32) -> f32 {
    return fma(a.w, k.w, fma(a.z, k.z, fma(a.y, k.y, fma(a.x, k.x, acc))));
}

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Flat workgroup id -> (b, kv head, slab, split), split fastest.
    let wg = wgid.y * nwg.x + wgid.x;
    let s = wg % p.n_splits;
    let wg1 = wg / p.n_splits;
    let slabs = (p.group + SLAB - 1u) / SLAB;
    let slab = wg1 % slabs;
    let wg2 = wg1 / slabs;
    let kvh = wg2 % p.n_kv_heads;
    let b = wg2 / p.n_kv_heads;
    if (b >= p.batch) { return; }

    // Both exits are uniform across the workgroup, so they precede every barrier.
    let len = seq_lens[b];
    let nchunks = (len + CHUNK - 1u) / CHUNK;
    let cps = (nchunks + p.n_splits - 1u) / p.n_splits;
    let c0 = s * cps;
    if (c0 >= nchunks) { return; }
    let c1 = min(c0 + cps, nchunks);

    let hd = p.head_dim;
    let nq = hd / 16u;                       // 16-byte pieces per key row
    let t = lid.x;
    let w = t % 32u;                         // value pass: packed word
    let kg = t / 32u;                        // value pass: key group
    let nact = min(SLAB, p.group - slab * SLAB);
    let h0 = kvh * p.group + slab * SLAB;
    let scale = inverseSqrt(f32(hd));

    {
        let g = t / 32u;
        let i = t % 32u;
        var qv = vec4<f32>(0.0);
        if (g < nact && i * 4u < hd) {
            let base = (b * p.n_heads + h0 + g) * hd + i * 4u;
            qv = vec4<f32>(q[base], q[base + 1u], q[base + 2u], q[base + 3u]);
        }
        qs[t] = qv;
    }
    if (t < SLAB) { mrun[t] = -3.4e38; }

    var acc: array<vec4<f32>, 4>;
    for (var g = 0u; g < SLAB; g = g + 1u) { acc[g] = vec4<f32>(0.0); }
    var lp = vec4<f32>(0.0);
    workgroupBarrier();

    for (var c = c0; c < c1; c = c + 1u) {
        // ---- scores: key `t` of the chunk against the four heads.
        let j = c * CHUNK + t;
        let live = j < len;
        var d0 = 0.0;
        var d1 = 0.0;
        var d2 = 0.0;
        var d3 = 0.0;
        var vrow = 0u;
        var kscale = 0.0;
        var vscale = 0.0;
        if (live) {
            let physical = block_tables[b * p.max_bt + j / p.block_size];
            let slot = physical * p.block_size + (j % p.block_size);
            let row = slot * p.n_kv_heads + kvh;   // (token slot, kv head) pair
            kscale = scales_k[row] * scale;
            vscale = scales_v[row];
            vrow = row * (hd / 4u);
            for (var i = 0u; i < 8u; i = i + 1u) {
                if (i < nq) {
                    let kw = pool_k[row * nq + i];
                    let k0 = unpack_i8x4(kw.x);
                    let k1 = unpack_i8x4(kw.y);
                    let k2 = unpack_i8x4(kw.z);
                    let k3 = unpack_i8x4(kw.w);
                    let qi = i * 4u;
                    d0 = fma_dot4(qs[qi + 3u], k3, fma_dot4(qs[qi + 2u], k2, fma_dot4(qs[qi + 1u], k1, fma_dot4(qs[qi], k0, d0))));
                    d1 = fma_dot4(qs[32u + qi + 3u], k3, fma_dot4(qs[32u + qi + 2u], k2, fma_dot4(qs[32u + qi + 1u], k1, fma_dot4(qs[32u + qi], k0, d1))));
                    d2 = fma_dot4(qs[64u + qi + 3u], k3, fma_dot4(qs[64u + qi + 2u], k2, fma_dot4(qs[64u + qi + 1u], k1, fma_dot4(qs[64u + qi], k0, d2))));
                    d3 = fma_dot4(qs[96u + qi + 3u], k3, fma_dot4(qs[96u + qi + 2u], k2, fma_dot4(qs[96u + qi + 1u], k1, fma_dot4(qs[96u + qi], k0, d3))));
                }
            }
        }
        rowb[t] = vrow;
        sc[t] = select(-3.4e38, d0 * kscale, live);
        sc[128u + t] = select(-3.4e38, d1 * kscale, live);
        sc[256u + t] = select(-3.4e38, d2 * kscale, live);
        sc[384u + t] = select(-3.4e38, d3 * kscale, live);
        workgroupBarrier();

        // ---- softmax: head `t / 32` reduces keys `4 * (t % 32) .. + 3`, then
        // one lane per head folds the 32 partials.
        {
            let base = (t / 32u) * CHUNK + (t % 32u) * 4u;
            red[t] = max(max(sc[base], sc[base + 1u]), max(sc[base + 2u], sc[base + 3u]));
        }
        workgroupBarrier();
        if (t < SLAB) {
            var cm = -3.4e38;
            for (var i = 0u; i < 32u; i = i + 1u) { cm = max(cm, red[t * 32u + i]); }
            let m_old = mrun[t];
            let m_new = max(m_old, cm);
            corr[t] = exp(m_old - m_new);
            mrun[t] = m_new;
        }
        workgroupBarrier();
        {
            // Keys past the sequence score -3.4e38, so they weigh exactly 0.
            let e = vec4<f32>(
                exp(sc[t] - mrun[0]),
                exp(sc[128u + t] - mrun[1]),
                exp(sc[256u + t] - mrun[2]),
                exp(sc[384u + t] - mrun[3]));
            pe[t] = e;
            pv[t] = e * vscale;
        }
        workgroupBarrier();

        // ---- values: word `w` of the value row for keys `kg + 4i`.
        let cr = vec4<f32>(corr[0], corr[1], corr[2], corr[3]);
        for (var g = 0u; g < SLAB; g = g + 1u) { acc[g] = acc[g] * cr[g]; }
        lp = lp * cr;
        if (w * 4u < hd) {
            let nlive = min(CHUNK, len - c * CHUNK);
            // Four independent loads in flight per step. Entries past
            // `nlive` hold a zero weight and a valid row index.
            for (var r = kg; r < nlive; r = r + 16u) {
                let v0 = unpack_i8x4(pool_v[rowb[r] + w]);
                let v1 = unpack_i8x4(pool_v[rowb[r + 4u] + w]);
                let v2 = unpack_i8x4(pool_v[rowb[r + 8u] + w]);
                let v3 = unpack_i8x4(pool_v[rowb[r + 12u] + w]);
                let a0 = pv[r];
                let a1 = pv[r + 4u];
                let a2 = pv[r + 8u];
                let a3 = pv[r + 12u];
                lp = lp + pe[r] + pe[r + 4u] + pe[r + 8u] + pe[r + 12u];
                acc[0] = fma(vec4<f32>(a3.x), v3, fma(vec4<f32>(a2.x), v2, fma(vec4<f32>(a1.x), v1, fma(vec4<f32>(a0.x), v0, acc[0]))));
                acc[1] = fma(vec4<f32>(a3.y), v3, fma(vec4<f32>(a2.y), v2, fma(vec4<f32>(a1.y), v1, fma(vec4<f32>(a0.y), v0, acc[1]))));
                acc[2] = fma(vec4<f32>(a3.z), v3, fma(vec4<f32>(a2.z), v2, fma(vec4<f32>(a1.z), v1, fma(vec4<f32>(a0.z), v0, acc[2]))));
                acc[3] = fma(vec4<f32>(a3.w), v3, fma(vec4<f32>(a2.w), v2, fma(vec4<f32>(a1.w), v1, fma(vec4<f32>(a0.w), v0, acc[3]))));
            }
        }
        // `sc`, `red`, `rowb`, `pe` and `pv` are rewritten by the next chunk.
        workgroupBarrier();
    }

    // ---- fold the four key groups into group 0, one group per step.
    if (w == 0u) {
        lfin[kg * 4u] = lp.x;
        lfin[kg * 4u + 1u] = lp.y;
        lfin[kg * 4u + 2u] = lp.z;
        lfin[kg * 4u + 3u] = lp.w;
    }
    for (var rd = 1u; rd < 4u; rd = rd + 1u) {
        if (kg == rd) {
            for (var g = 0u; g < SLAB; g = g + 1u) {
                let o = g * HD_MAX + w * 4u;
                fin[o] = acc[g].x;
                fin[o + 1u] = acc[g].y;
                fin[o + 2u] = acc[g].z;
                fin[o + 3u] = acc[g].w;
            }
        }
        workgroupBarrier();
        if (kg == 0u) {
            for (var g = 0u; g < SLAB; g = g + 1u) {
                let o = g * HD_MAX + w * 4u;
                acc[g] = acc[g] + vec4<f32>(fin[o], fin[o + 1u], fin[o + 2u], fin[o + 3u]);
            }
        }
        workgroupBarrier();
    }

    if (kg == 0u && w * 4u < hd) {
        for (var g = 0u; g < nact; g = g + 1u) {
            let base = ((b * p.n_heads + h0 + g) * p.n_splits + s) * PART_STRIDE + 2u + w * 4u;
            part_out[base] = acc[g].x;
            part_out[base + 1u] = acc[g].y;
            part_out[base + 2u] = acc[g].z;
            part_out[base + 3u] = acc[g].w;
        }
    }
    if (t < nact) {
        let base = ((b * p.n_heads + h0 + t) * p.n_splits + s) * PART_STRIDE;
        part_out[base] = mrun[t];
        part_out[base + 1u] = lfin[t] + lfin[4u + t] + lfin[8u + t] + lfin[12u + t];
    }
}
