// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused paged-attention decode over an INT8 KV pool: block-table traversal, online softmax, context - one dispatch
// @how   64-thread workgroup per (sequence, head), tiled key staging with per-token int8 dequant, 5 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The INT8-KV twin of `paged_flash_decode.wgsl` (M2.2): identical online-
// softmax algorithm and tiling (`BC = 8` keys/tile, `LANES = 8`-way head_dim
// split - see that kernel's own header for the full derivation), but
// `pool_k`/`pool_v` are packed 4-int8-per-`u32` pools with a per-`(token,
// kv_head)` dequant scale, exactly the scheme `paged_decode_scores_i8_batched`
// / `paged_decode_apply_i8_batched` already establish (`dequant = signed_byte
// * scales[slot * n_kv_heads + kv_head]`). Dequantizing happens once, while
// staging a key/value tile into shared memory - every stage downstream (dot
// product, online-softmax fold, V accumulation) is byte-for-byte the same
// code as the fp32 kernel's, reading the same `ksh`/`vsh` tiles.
//
// This is strictly the worst case to fuse against: the int8 paged-attention
// path has no cooperative `_wg` sibling at all today (unlike fp32's
// `paged_decode_scores_wg`), so the reference this kernel replaces is three
// dispatches (`paged_decode_scores_i8_batched` -> `decode_softmax_batched` ->
// `paged_decode_apply_i8_batched`), each of which re-reads or re-writes the
// `[batch, n_heads, cap]` scores/probs slab in addition to the packed pool.
//
//   q             : [batch, n_heads*head_dim]                        (f32)
//   pool_k/pool_v : [num_blocks*block_size*n_kv_heads*head_dim / 4]  (u32, 4 int8/word)
//   scales_k/v    : [num_blocks*block_size, n_kv_heads]              (f32, per token x kv-head)
//   block_tables  : [batch, max_bt]
//   seq_lens      : [batch]
//   ctx           : [batch, n_heads*head_dim]                        (f32)
//
// Same GPU-ONLY-BY-CONSTRUCTION reasoning as the fp32 sibling: FIVE top-level
// `workgroupBarrier()`s exceed the CPU JIT's one-barrier-per-body limit, so
// `@cpu no`; the three-stage int8 triad stays registered as the CPU/reference
// path, this is an additional GPU sibling, never a replacement.
//
// 8 storage buffers (q, pool_k, pool_v, scales_k, scales_v, block_tables,
// seq_lens, ctx) sit exactly at the WebGPU `maxStorageBuffersPerShaderStage`
// guaranteed floor - the same ceiling the splat backward kernels already use.

const BC: u32 = 8u;      // keys staged and scored per tile
const LANES: u32 = 8u;   // threads cooperating on one key's head_dim reduction
const CH: u32 = 16u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide

struct Params {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 128
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

var<workgroup> qsh:  array<f32, 128>;  // HD
var<workgroup> ksh:  array<f32, 1024>; // BC*HD -> 4 KiB
var<workgroup> vsh:  array<f32, 1024>; // BC*HD -> 4 KiB
var<workgroup> part: array<f32, 64>;   // BC*LANES
var<workgroup> sc:   array<f32, 8>;    // BC

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    let wg = wgid.y * nwg.x + wgid.x;
    let h = wg % p.n_heads;
    let b = wg / p.n_heads;
    if (b >= p.batch) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;
    let row = lt / LANES;
    let lane = lt % LANES;

    let q_base = (b * q_row) + h * hd;
    for (var e = lt; e < HD; e = e + 64u) {
        qsh[e] = select(0.0, q[q_base + e], e < hd);
    }
    workgroupBarrier();

    var o: array<f32, 16>;
    for (var c = 0u; c < CH; c = c + 1u) { o[c] = 0.0; }
    var m = -3.4e38;
    var l = 0.0;

    let t = seq_lens[b];
    let ntiles = (t + BC - 1u) / BC;

    for (var kt = 0u; kt < ntiles; kt = kt + 1u) {
        // Stage the K,V tile [BC x HD] into shared, dequantizing each int8
        // byte against its own token's per-kv-head scale as it is staged -
        // downstream of this loop the tile is plain f32, unchanged from the
        // fp32 kernel.
        for (var e = lt; e < BC * HD; e = e + 64u) {
            let jr = e / HD;
            let d = e % HD;
            let j = kt * BC + jr;
            if (j < t && d < hd) {
                let physical = block_tables[b * p.max_bt + j / p.block_size];
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

        // Partial dot product: this lane's CH channels of this tile's key `row`.
        var sdot = 0.0;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            sdot = sdot + qsh[d] * ksh[row * HD + d];
        }
        part[row * LANES + lane] = sdot;
        workgroupBarrier();

        // One thread per row folds its LANES partials into the scaled score,
        // masking a key past the live sequence length to -inf.
        if (lane == 0u) {
            let j = kt * BC + row;
            if (j < t) {
                let po = row * LANES;
                var ssum = 0.0;
                for (var li = 0u; li < LANES; li = li + 1u) { ssum = ssum + part[po + li]; }
                sc[row] = ssum * scale;
            } else {
                sc[row] = -3.4e38;
            }
        }
        workgroupBarrier();

        var tile_max = -3.4e38;
        for (var r = 0u; r < BC; r = r + 1u) { tile_max = max(tile_max, sc[r]); }
        let m_new = max(m, tile_max);
        let corr = exp(m - m_new);

        var pj: array<f32, 8>;
        var tile_l = 0.0;
        for (var r = 0u; r < BC; r = r + 1u) {
            let e = exp(sc[r] - m_new);
            pj[r] = e;
            tile_l = tile_l + e;
        }
        l = l * corr + tile_l;
        m = m_new;

        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            var acc = 0.0;
            for (var r = 0u; r < BC; r = r + 1u) {
                acc = acc + pj[r] * vsh[r * HD + d];
            }
            o[c] = o[c] * corr + acc;
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    if (row == 0u) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = (b * q_row) + h * hd;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { ctx[o_base + d] = o[c] * inv; }
        }
    }
}
