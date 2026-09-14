// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused causal chunked paged-attention prefill over an INT8 KV pool: BR query rows/workgroup, online softmax, no [nh,N,N] slab
// @how   256-thread workgroup per (head, query-tile), LANE-SPLIT across head_dim, tiled K/V staging with per-token int8 dequant, causal early-exit over K tiles, 3 barriers/tile
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant int8
// @dtype f32
//
// The INT8-KV twin of `paged_flash_prefill.wgsl`: identical online-softmax
// algorithm, identical `BR=64` query-row tiling, identical `LANES=4`/`CH=32`
// head_dim lane split, identical causal early-exit and per-row masking (see
// that kernel's header for the full derivation and for the TWO CALL-SITE
// CONTRACTS - one sequence per dispatch, non-decreasing `seq_lens` across a
// tile - which this kernel inherits unchanged). The ONLY difference is how a
// K/V tile is staged: `pool_k`/`pool_v` here are packed 4-int8-per-`u32`
// pools with a per-`(token slot, kv head)` dequant scale, exactly the scheme
// `paged_decode_scores_i8_batched`/`paged_decode_apply_i8_batched` and
// `paged_flash_decode_i8` already establish (`dequant = signed_byte *
// scales[slot * n_kv_heads + kv_head]`).
//
// Dequantizing happens ONCE, while a tile is staged into `ksh`/`vsh`.
// Downstream of the staging barrier the tiles are plain f32 and every
// remaining stage - partial dot product, cross-lane fold, online-softmax
// rescale, V accumulation, epilogue - is byte-for-byte the fp32 kernel's own
// code. So the online softmax sees exactly the numerics it would have seen on
// a dequantized fp32 pool: int8 is a STORAGE tier here, never a change to the
// reduction order, which is why this kernel's error against the fp32 fused
// kernel is plain quantization noise and nothing else.
//
//   q            : [bsz, n_heads*head_dim]                          (f32, bsz = chunk length, ONE sequence)
//   pool_k/pool_v: [num_blocks*block_size*n_kv_heads*head_dim / 4]   (u32, 4 int8/word)
//   scales_k/v   : [num_blocks*block_size, n_kv_heads]               (f32, per token x kv-head)
//   block_tables : [bsz, max_bt]   (physical block index per logical block)
//   seq_lens     : [bsz]           (live key count per query row)
//   ctx          : [bsz, n_heads*head_dim]                           (f32)
//
// A packed `u32` must stay within ONE head, or its four lanes would span two
// heads' scales - i.e. `head_dim % 4 == 0`, the same contract the append
// kernels impose and `qwen3::serve::kv_int8_supported` checks before an engine
// ever selects int8 KV.
//
// 8 storage buffers (q, pool_k, pool_v, scales_k, scales_v, block_tables,
// seq_lens, ctx) sit exactly at the WebGPU `maxStorageBuffersPerShaderStage`
// guaranteed floor - the same ceiling `paged_flash_decode_i8` already uses.
//
// Same GPU-ONLY-BY-CONSTRUCTION reasoning as the fp32 sibling: three top-level
// `workgroupBarrier()`s per K tile exceed the CPU JIT's one-barrier-per-body
// limit, so `@cpu no`; the three-stage int8 triad (`paged_decode_scores_i8_
// batched` -> `decode_softmax_batched` -> `paged_decode_apply_i8_batched`)
// stays registered as the CPU/reference path, this is an additional GPU
// sibling, never a replacement.
//
// Shared memory: ksh/vsh `BC*HD` = 4 KiB each, part `BC*BR*LANES` = 8 KiB ->
// 16 KiB total, IDENTICAL to `paged_flash_prefill`'s own budget (which sits
// exactly at WebGPU's guaranteed `maxComputeWorkgroupStorageSize` floor).

const BR: u32 = 64u;     // query rows per workgroup
const BC: u32 = 8u;      // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const CH: u32 = 32u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

struct Params {
    bsz: u32,          // chunk length (query rows), ONE sequence
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 128, and a multiple of 4 (packed-word contract)
    group: u32,        // n_heads / n_kv_heads
    block_size: u32,
    max_bt: u32,       // block_tables row stride (blocks per sequence)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<u32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<u32>;
@group(0) @binding(4) var<storage, read>       scales_k:     array<f32>;
@group(0) @binding(5) var<storage, read>       scales_v:     array<f32>;
@group(0) @binding(6) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(7) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(8) var<storage, read_write> ctx:          array<f32>;

var<workgroup> ksh:  array<f32, 1024>;  // BC*HD  -> 4 KiB
var<workgroup> vsh:  array<f32, 1024>;  // BC*HD  -> 4 KiB
var<workgroup> part: array<f32, 2048>;  // BC*BR*LANES -> 8 KiB

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (h, query-tile) - no separate sequence axis: a
    // prefill dispatch's `bsz` rows all belong to the ONE sequence being
    // chunked (see the header's contract 1).
    let wg = wgid.y * nwg.x + wgid.x;
    let ntiles_q = (p.bsz + BR - 1u) / BR;
    let qt = wg % ntiles_q;
    let h = wg / ntiles_q;
    if (h >= p.n_heads) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;                 // 0..255
    let row = lt / LANES;           // 0..63  -> query row within the tile
    let lane = lt % LANES;          // 0..3   -> channel phase
    let i = qt * BR + row;          // this thread's absolute query row
    let live = i < p.bsz;

    // This thread's slice of q, and its slice of the output accumulator.
    // Channels are interleaved: slot c holds channel c*LANES + lane.
    var qv: array<f32, 32>;
    var o: array<f32, 32>;
    let q_base = i * q_row + h * hd;
    for (var c = 0u; c < CH; c = c + 1u) {
        let d = c * LANES + lane;
        if (live && d < hd) { qv[c] = q[q_base + d]; } else { qv[c] = 0.0; }
        o[c] = 0.0;
    }

    var m = -3.4e38;   // running max
    var l = 0.0;       // running sum of exp

    var pj: array<f32, 8>;   // BC softmax weights for the current tile

    // This row's own causal boundary (live key count).
    let t_i = select(0u, seq_lens[i], live);

    // Causal early-exit (header contract 2): the workgroup's largest live
    // boundary is its LAST row's own `seq_lens`, since a chunk's boundaries
    // only grow with row index - so K tiles beyond it need never be staged.
    let max_i_in_wg = min(qt * BR + BR - 1u, p.bsz - 1u);
    let max_t = seq_lens[max_i_in_wg];
    let ntiles_k = (max_t + BC - 1u) / BC;

    // Every row in this tile shares one physical block table (header
    // contract 1) - read through the tile's first row.
    let bt_row = qt * BR;

    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        // Stage the K,V tile [BC x HD] into shared, dequantizing each int8
        // byte against its own token's per-kv-head scale as it is staged -
        // downstream of this loop the tile is plain f32, unchanged from the
        // fp32 kernel.
        for (var e = lt; e < BC * HD; e = e + 256u) {
            let jr = e / HD;
            let d = e % HD;
            let j = kt * BC + jr;
            if (j < max_t && d < hd) {
                let physical = block_tables[bt_row * p.max_bt + j / p.block_size];
                let tok_slot = physical * p.block_size + (j % p.block_size);
                let elem = tok_slot * kv_row + hkv * hd + d;

                let kbyte = (pool_k[elem / 4u] >> (8u * (elem % 4u))) & 0xffu;
                let kiv = i32(kbyte);
                let ksv = select(kiv, kiv - 256, kiv > 127);
                ksh[e] = f32(ksv) * scales_k[tok_slot * p.n_kv_heads + hkv];

                let vbyte = (pool_v[elem / 4u] >> (8u * (elem % 4u))) & 0xffu;
                let viv = i32(vbyte);
                let vsv = select(viv, viv - 256, viv > 127);
                vsh[e] = f32(vsv) * scales_v[tok_slot * p.n_kv_heads + hkv];
            } else {
                ksh[e] = 0.0;
                vsh[e] = 0.0;
            }
        }
        workgroupBarrier();

        // Partial dot products: this lane's 32 channels of every key in the tile.
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = 0.0;
            let ko = j * HD + lane;
            for (var c = 0u; c < CH; c = c + 1u) {
                s = s + qv[c] * ksh[ko + c * LANES];
            }
            part[j * (BR * LANES) + row * LANES + lane] = s;
        }
        workgroupBarrier();

        // Every lane re-sums the LANES partials (redundant but register-cheap,
        // and it leaves each lane with the p_j it needs for its own
        // channels). The causal mask lands here, against THIS ROW's own
        // `t_i`: a key at or past `t_i` is excluded before the tile-wide
        // max/exp.
        var krows = BC;
        let rem = max_t - kt * BC;
        if (rem < BC) { krows = rem; }
        var tmax = -3.4e38;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = -3.4e38;
            if (j < krows && (kt * BC + j) < t_i) {
                let po = j * (BR * LANES) + row * LANES;
                s = (part[po] + part[po + 1u] + part[po + 2u] + part[po + 3u]) * scale;
            }
            pj[j] = s;
            tmax = max(tmax, s);
        }
        let m_new = max(m, tmax);
        let corr = exp(m - m_new);
        var lsum = 0.0;
        for (var j = 0u; j < BC; j = j + 1u) {
            let e = exp(pj[j] - m_new);
            pj[j] = e;
            lsum = lsum + e;
        }
        l = l * corr + lsum;
        m = m_new;

        // One rescale of o per TILE, not per key.
        for (var c = 0u; c < CH; c = c + 1u) {
            let vo = c * LANES + lane;
            var acc = 0.0;
            for (var j = 0u; j < BC; j = j + 1u) {
                acc = acc + pj[j] * vsh[j * HD + vo];
            }
            o[c] = o[c] * corr + acc;
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    if (live) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = i * q_row + h * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { ctx[o_base + d] = o[c] * inv; }
        }
    }
}
