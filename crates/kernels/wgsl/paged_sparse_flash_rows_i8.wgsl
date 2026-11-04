// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse paged attention of a whole query row (every head) over its selected keys in an int8 KV pool
// @how   64-thread workgroup per query row: each key's full packed K/V row staged once (coalesced) and read by every head, a thread owns a head's channel slice in registers
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
// @tpl   HEADS, KVH, HD -> the model's query heads, KV heads and head width
//        (`HEADS` must divide 64); the int8 byte decode between the
//        kv8-decode markers -> FP8 E4M3 via
//        `kernels::template::fp8_decode_variant`.
//
// Swedish Embedded AB implements long-context prefill whose attention cost
// is set by the keys it selects, not the context it holds, for its clients.
// If your team needs expertise in bandwidth-bound GPU attention kernels, you
// can procure our services by sending an email to info@swedishembedded.com.
//
// The prefill-shaped sparse read over the served int8 KV pool. A sparse
// selection is shared by every head of a query (latent sparse attention
// selects once per query), so one workgroup owns one query ROW and all its
// heads:
//
//   * a selected key is resolved to its pool slot once per row (not once
//     per KV head), and its whole packed K row - every KV head, `KVH * HD`
//     bytes, contiguous in the pool - is staged in one coalesced sweep, then
//     its V row into the same buffer;
//   * thread `t` owns head `t / TPH`, channels `[(t % TPH) * CPT, +CPT)`: its
//     query slice and output accumulator live in registers, its partial dot
//     products go to workgroup memory, and the head's `TPH` partials are
//     folded by one thread per head into the online softmax (weights and
//     rescale in workgroup memory);
//   * each key carries `HEADS * HD * 2` multiply-adds of work per 64 threads,
//     against one resolve and two row loads.
//
// Selection rows as `paged_sparse_flash_decode` documents them: a header
// `[count, table_row, live]`, then logical positions. A position at or past
// `live` is masked. Bindings are `paged_sparse_flash_gqa_i8`'s; dispatch
// `rows` workgroups. Requires `n_heads == HEADS`, `n_kv_heads == KVH`,
// `head_dim == HD`, `HD % 4 == 0`.
//
// Workgroup memory at 16 heads, 8 KV heads of 128: K/V tile 8 KiB + partials
// 2 KiB + weights 512 B + scales 512 B ~= 11 KiB.

const BC: u32 = 8u;               // keys per tile
const WG: u32 = 64u;
const HEADS: u32 = 16u;           // query heads
const KVH: u32 = 8u;              // KV heads
const HD: u32 = 128u;             // head width
const GROUP: u32 = HEADS / KVH;
const TPH: u32 = WG / HEADS;      // threads per head
const CPT: u32 = HD / TPH;        // channels per thread
const KVW: u32 = KVH * HD / 4u;   // packed words per key row
const NO_SLOT: u32 = 0xFFFFFFFFu;

struct Params {
    rows: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    group: u32,
    block_size: u32,
    max_bt: u32,
    sel_width: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<u32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<u32>;
@group(0) @binding(4) var<storage, read>       scales_k:     array<f32>;
@group(0) @binding(5) var<storage, read>       scales_v:     array<f32>;
@group(0) @binding(6) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(7) var<storage, read>       sel:          array<u32>;
@group(0) @binding(8) var<storage, read_write> ctx:          array<f32>;

var<workgroup> tile:  array<u32, BC * KVW>;   // packed K, then V, rows
var<workgroup> tsc:   array<f32, BC * KVH>;   // their per-KV-head scales
var<workgroup> part:  array<f32, BC * WG>;    // partial dot products
var<workgroup> pw:    array<f32, HEADS * BC>; // softmax weights per head
var<workgroup> corr:  array<f32, HEADS>;
var<workgroup> m_sh:  array<f32, HEADS>;
var<workgroup> l_sh:  array<f32, HEADS>;
var<workgroup> kslot: array<u32, BC>;

// @kv8-decode-begin
// Four signed int8 bytes of a packed word, lowest first, widened to f32.
// (`kernels::template::fp8_decode_variant` swaps this block for E4M3.)
fn unpack_i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(bitcast<i32>(w << 24u) >> 24u),
        f32(bitcast<i32>(w << 16u) >> 24u),
        f32(bitcast<i32>(w << 8u) >> 24u),
        f32(bitcast<i32>(w) >> 24u));
}
// @kv8-decode-end

// Stage the tile's packed rows of `pool` and their scales.
fn stage(lt: u32, from_v: bool) {
    for (var w = lt; w < BC * KVW; w = w + WG) {
        let tok = kslot[w / KVW];
        var word = 0u;
        if (tok != NO_SLOT) {
            let at = tok * KVW + w % KVW;
            if (from_v) { word = pool_v[at]; } else { word = pool_k[at]; }
        }
        tile[w] = word;
    }
    for (var e = lt; e < BC * KVH; e = e + WG) {
        let tok = kslot[e / KVH];
        var s = 0.0;
        if (tok != NO_SLOT) {
            let at = tok * KVH + e % KVH;
            if (from_v) { s = scales_v[at]; } else { s = scales_k[at]; }
        }
        tsc[e] = s;
    }
}

// Resolve tile `kt`'s selected positions to pool slots (threads below BC).
fn resolve(lt: u32, kt: u32, count: u32, table: u32, live: u32, sel_base: u32) {
    if (lt < BC) {
        let i = kt * BC + lt;
        var slot = NO_SLOT;
        if (i < count) {
            let j = sel[sel_base + i];
            if (j < live) {
                slot = block_tables[table + j / p.block_size] * p.block_size + j % p.block_size;
            }
        }
        kslot[lt] = slot;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wgid.y * nwg.x + wgid.x;
    if (r >= p.rows) { return; }
    let lt = lid.x;
    let h = lt / TPH;
    let c0 = (lt % TPH) * CPT;
    let hkv = h / GROUP;
    let scale = inverseSqrt(f32(HD));
    let q_row = HEADS * HD;
    let count = sel[r * p.sel_width];
    let table = sel[r * p.sel_width + 1u] * p.max_bt;
    let live = sel[r * p.sel_width + 2u];
    let sel_base = r * p.sel_width + 3u;

    var qv: array<f32, CPT>;
    var o: array<f32, CPT>;
    for (var i = 0u; i < CPT; i = i + 1u) {
        qv[i] = q[r * q_row + h * HD + c0 + i];
        o[i] = 0.0;
    }
    if (lt < HEADS) {
        m_sh[lt] = -3.4e38;
        l_sh[lt] = 0.0;
    }

    let ntiles = (count + BC - 1u) / BC;
    // Four barriers a tile: the next tile's slots are resolved during this
    // tile's P.V sum, and V is staged while one thread per head runs the
    // softmax.
    resolve(lt, 0u, count, table, live, sel_base);
    workgroupBarrier();
    let kbase = hkv * HD / 4u + c0 / 4u;
    for (var kt = 0u; kt < ntiles; kt = kt + 1u) {
        stage(lt, false);
        workgroupBarrier();

        // This thread's slice of every key's score for its head.
        for (var j = 0u; j < BC; j = j + 1u) {
            var acc = 0.0;
            for (var wq = 0u; wq < CPT / 4u; wq = wq + 1u) {
                let kv = unpack_i8x4(tile[j * KVW + kbase + wq]);
                acc = acc + qv[4u * wq] * kv.x + qv[4u * wq + 1u] * kv.y + qv[4u * wq + 2u] * kv.z + qv[4u * wq + 3u] * kv.w;
            }
            part[j * WG + lt] = acc * tsc[j * KVH + hkv];
        }
        workgroupBarrier();

        // One thread per head folds the slices into the online softmax; then
        // every thread stages the V rows over K.
        if (lt < HEADS) {
            let hh = lt;
            var tile_max = -3.4e38;
            for (var j = 0u; j < BC; j = j + 1u) {
                var v = -3.4e38;
                if (kslot[j] != NO_SLOT) {
                    var acc = 0.0;
                    for (var t = 0u; t < TPH; t = t + 1u) { acc = acc + part[j * WG + hh * TPH + t]; }
                    v = acc * scale;
                }
                pw[hh * BC + j] = v;
                tile_max = max(tile_max, v);
            }
            let m_old = m_sh[hh];
            let m_new = max(m_old, tile_max);
            var tile_l = 0.0;
            for (var j = 0u; j < BC; j = j + 1u) {
                let e = select(0.0, exp(pw[hh * BC + j] - m_new), kslot[j] != NO_SLOT);
                pw[hh * BC + j] = e;
                tile_l = tile_l + e;
            }
            let cr = exp(m_old - m_new);
            corr[hh] = cr;
            l_sh[hh] = l_sh[hh] * cr + tile_l;
            m_sh[hh] = m_new;
        }
        stage(lt, true);
        workgroupBarrier();

        // The slots are free again: resolve the next tile's while summing.
        resolve(lt, kt + 1u, count, table, live, sel_base);
        let cr = corr[h];
        for (var i = 0u; i < CPT; i = i + 1u) { o[i] = o[i] * cr; }
        for (var j = 0u; j < BC; j = j + 1u) {
            let w = pw[h * BC + j] * tsc[j * KVH + hkv];
            for (var wq = 0u; wq < CPT / 4u; wq = wq + 1u) {
                let vv = unpack_i8x4(tile[j * KVW + kbase + wq]);
                o[4u * wq] = o[4u * wq] + w * vv.x;
                o[4u * wq + 1u] = o[4u * wq + 1u] + w * vv.y;
                o[4u * wq + 2u] = o[4u * wq + 2u] + w * vv.z;
                o[4u * wq + 3u] = o[4u * wq + 3u] + w * vv.w;
            }
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    let l = l_sh[h];
    for (var i = 0u; i < CPT; i = i + 1u) {
        ctx[r * q_row + h * HD + c0 + i] = select(0.0, o[i] / l, l > 0.0);
    }
}
