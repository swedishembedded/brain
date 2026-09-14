// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Split-key phase of a two-pass FlashDecode over an INT8 KV pool: each workgroup owns one (sequence, head, key-split) triple and writes a PARTIAL online-softmax state, not a final context
// @how   64-thread workgroup per (sequence, head, split), tiled key staging with per-token int8 dequant, 5 barriers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// The INT8-KV twin of `paged_flash_decode_split.wgsl`, exactly the way
// `paged_flash_decode_i8.wgsl` is the INT8-KV twin of `paged_flash_decode.wgsl`
// - same tiling and partial-state algorithm as the fp32 split kernel (see
// that file's header for the full occupancy-fix derivation: this is M2.7's
// split-key design, not a different attention formulation), but `pool_k`/
// `pool_v` are packed 4-int8-per-`u32` pools with a per-`(token, kv_head)`
// dequant scale - the identical `dequant = signed_byte * scales[slot *
// n_kv_heads + kv_head]` scheme `paged_flash_decode_i8.wgsl`'s tile-load
// already establishes. Dequantizing happens once, while staging a key/value
// tile into shared memory; every stage downstream (dot product, online-
// softmax fold, V accumulation) is byte-for-byte the same code as the fp32
// split kernel's.
//
// Exists because `Op::PagedAttentionFused`'s own doc names the gap this
// closes: `paged_flash_decode_i8` (the single-fused int8 kernel, M2.2) was
// gated correctness-only and never independently measured for occupancy -
// measured now (`qwen_bench flash-decode-i8`), it loses to the int8 triad by
// over 2x, the same serialised-tile-walk regression M2.1 found and M2.7 fixed
// for fp32. This is that fix, ported.
//
// ONE merged partial-state output, NOT the fp32 split kernel's three
// (`part_m`/`part_l`/`part_o`) - the two extra dequant-scale inputs this
// kernel needs over the fp32 one push a naive port to 10 storage buffers,
// over WebGPU's guaranteed 8-per-stage floor (`paged_flash_decode_i8`'s own
// header already notes landing exactly AT that floor with one fewer output
// than this kernel family has). Folding `m`/`l`/`o` into one `part` buffer,
// stride `2 + HD` per `(b, h, s)` (`[m, l, o_0..o_{hd-1}]`), buys back two
// bindings at no cost this kernel's own body pays - the three quantities were
// already computed and written together, per split, by the same thread.
// `paged_flash_decode_combine_i8` is this layout's own combine phase; plain
// `paged_flash_decode_combine` (three separate buffers) is UNCHANGED and
// still serves the fp32 split kernel.
//
// A workgroup whose split lies entirely past the sequence's own live length
// needs no special case - see `paged_flash_decode_split.wgsl`'s own note;
// identical here, the degenerate `(m=-3.4e38, l=0.0, o=[0.0; CH])` triple is
// exactly what `paged_flash_decode_combine_i8` already treats as a no-op
// merge.
//
//   q             : [batch, n_heads*head_dim]                       (f32)
//   pool_k/pool_v : [num_blocks*block_size*n_kv_heads*head_dim / 4] (u32, 4 int8/word)
//   scales_k/v    : [num_blocks*block_size, n_kv_heads]             (f32, per token x kv-head)
//   block_tables  : [batch, max_bt]
//   seq_lens      : [batch]
//   part          : [batch, n_heads, n_splits, 2 + HD]  (m, l, then the UNNORMALISED accumulator)
//
// Shared memory: identical to `paged_flash_decode_split.wgsl` - qsh 512 B +
// ksh/vsh 4 KiB each + part 256 B + sc 32 B ~= 8.8 KiB, well under WebGPU's
// guaranteed 16 KiB floor. (The workgroup-local `part` array below is the
// per-tile dot-product scratch inherited from the fp32 kernel, unrelated to
// the merged storage buffer of the same English name.)
//
// 8 storage buffers (q, pool_k, pool_v, scales_k, scales_v, block_tables,
// seq_lens, part) - exactly WebGPU's guaranteed floor.

const BC: u32 = 8u;      // keys staged and scored per tile
const LANES: u32 = 8u;   // threads cooperating on one key's head_dim reduction
const CH: u32 = 16u;     // channels per lane (LANES*CH == HD)
const HD: u32 = 128u;    // max head_dim; tiles are always this wide
const PART_STRIDE: u32 = 2u + HD; // one partial-state entry: [m, l, o_0..o_127]

struct Params {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 128
    group: u32,        // n_heads / n_kv_heads
    block_size: u32,
    max_bt: u32,        // block_tables row stride (blocks per sequence)
    n_splits: u32,      // dispatched split count per (sequence, head)
    tiles_per_split: u32, // contiguous BC-tiles each split's workgroup walks
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<u32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<u32>;
@group(0) @binding(4) var<storage, read>       scales_k:     array<f32>;
@group(0) @binding(5) var<storage, read>       scales_v:     array<f32>;
@group(0) @binding(6) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(7) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(8) var<storage, read_write> part_out:     array<f32>;

var<workgroup> qsh:  array<f32, 128>;  // HD
var<workgroup> ksh:  array<f32, 1024>; // BC*HD -> 4 KiB
var<workgroup> vsh:  array<f32, 1024>; // BC*HD -> 4 KiB
var<workgroup> part: array<f32, 64>;   // BC*LANES (per-tile dot-product scratch)
var<workgroup> sc:   array<f32, 8>;    // BC

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    // Flat workgroup id -> (b, h, s) - split is the fastest-varying index,
    // matching the host's own flat dispatch order (identical convention to
    // `paged_flash_decode_split.wgsl`).
    let wg = wgid.y * nwg.x + wgid.x;
    let s = wg % p.n_splits;
    let wg2 = wg / p.n_splits;
    let h = wg2 % p.n_heads;
    let b = wg2 / p.n_heads;
    if (b >= p.batch) { return; }

    let hkv = h / p.group;
    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;

    let lt = lid.x;                  // 0..63
    let row = lt / LANES;            // 0..7 -> key-in-tile
    let lane = lt % LANES;           // 0..7 -> channel phase

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
    let kt0 = s * p.tiles_per_split;
    let kt_end = min(kt0 + p.tiles_per_split, ntiles);

    for (var kt = kt0; kt < kt_end; kt = kt + 1u) {
        // Stage the K,V tile [BC x HD] into shared, dequantizing each int8
        // byte against its own token's per-kv-head scale as it is staged -
        // downstream of this loop the tile is plain f32, byte-for-byte the
        // same code as the fp32 split kernel's.
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

        var sdot = 0.0;
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            sdot = sdot + qsh[d] * ksh[row * HD + d];
        }
        part[row * LANES + lane] = sdot;
        workgroupBarrier();

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
        workgroupBarrier();
    }

    if (row == 0u) {
        let part_idx = (b * p.n_heads + h) * p.n_splits + s;
        let base = part_idx * PART_STRIDE;
        if (lane == 0u) {
            part_out[base] = m;
            part_out[base + 1u] = l;
        }
        for (var c = 0u; c < CH; c = c + 1u) {
            let d = c * LANES + lane;
            if (d < hd) { part_out[base + 2u + d] = o[c]; }
        }
    }
}
