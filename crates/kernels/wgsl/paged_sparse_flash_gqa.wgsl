// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse paged attention over per-query selected key positions, one KV head's whole query-head group per workgroup
// @how   64-thread workgroup per (query row, kv head): 16-key tiles, K then V through one tile buffer, split dot products, per-head softmax in workgroup memory
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32|bf16
// @tpl   GROUP -> query heads per KV head (any; default 2);
//        pool_k,pool_v -> bf16 storage variant via
//        `kernels::template::dtype_variant`.
//
// Swedish Embedded AB implements long-context prefill whose attention reads
// each selected key once for every head that needs it, for its clients. If
// your team needs expertise in bandwidth-bound GPU attention kernels, you can
// procure our services by sending an email to info@swedishembedded.com.
//
// The prefill-shaped sibling of `paged_sparse_flash_decode`. A sparse
// selection is shared by every head of a query (latent sparse attention
// selects once per query), and in prefill no query tile shares a key tile, so
// what bounds the kernel is how many times a selected K/V row is fetched and
// how much work each fetch carries. Here:
//
//   * a workgroup owns one (row, kv head) pair and serves all GROUP query
//     heads of that KV head from each staged K/V row (traffic / GROUP);
//   * a tile is BC = 16 selected keys; their pool slots are resolved once
//     per tile into shared memory, not once per staged element;
//   * the GROUP x BC score dot products are split over the 64 threads with
//     no thread repeating another's work; one thread per head folds its
//     scores into the online softmax (weights and rescale in workgroup
//     memory) while the others stage V, and each thread then owns every
//     64th output channel of the group for the softmax-weighted V sum.
//
// K and V take turns in one 8 KiB tile buffer (V is staged after the scores
// no longer need K), keeping the kernel inside WebGPU's guaranteed 16 KiB of
// workgroup storage: at GROUP = 2, q 1 KiB + tile 8 KiB + partials 256 B.
// Five barriers per 16 keys. Agrees with `paged_sparse_flash_decode` to float
// reassociation (its tiles are 8 keys); gated against the host oracle.
//
// Same bindings and parameters as `paged_sparse_flash_decode`; dispatch
// `rows * n_kv_heads` workgroups, and `group` must equal GROUP.

const BC: u32 = 16u;     // keys per tile
const HD: u32 = 128u;    // max head_dim; tiles are always this wide
const GROUP: u32 = 2u;   // query heads per KV head
const WG: u32 = 64u;

const NDOT: u32 = GROUP * BC;              // score dot products per tile
const TPD: u32 = max(1u, WG / NDOT);       // threads per dot product
const DPT: u32 = (NDOT + WG - 1u) / WG;    // dot products per thread
const CPT: u32 = HD / TPD;                 // channels per thread per dot
const OPT: u32 = (GROUP * HD + WG - 1u) / WG; // output channels per thread
const NO_SLOT: u32 = 0xFFFFFFFFu;

struct Params {
    rows: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 128
    group: u32,        // n_heads / n_kv_heads, == GROUP
    block_size: u32,
    max_bt: u32,
    sel_width: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<f32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<f32>;
@group(0) @binding(4) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(5) var<storage, read>       sel:          array<u32>;
@group(0) @binding(6) var<storage, read>       row_meta:     array<u32>;
@group(0) @binding(7) var<storage, read_write> ctx:          array<f32>;

var<workgroup> qsh:   array<f32, GROUP * HD>;
var<workgroup> tile:  array<f32, BC * HD>;
var<workgroup> part:  array<f32, NDOT * TPD>;
var<workgroup> kslot: array<u32, BC>;
var<workgroup> pw:    array<f32, GROUP * BC>;  // scores, then softmax weights
var<workgroup> corr_sh: array<f32, GROUP>;
var<workgroup> m_sh:  array<f32, GROUP>;
var<workgroup> l_sh:  array<f32, GROUP>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

    let wg = wgid.y * nwg.x + wgid.x;
    let hkv = wg % p.n_kv_heads;
    let r = wg / p.n_kv_heads;
    if (r >= p.rows) { return; }

    let q_row = p.n_heads * hd;
    let kv_row = p.n_kv_heads * hd;
    let count = row_meta[r * 3u];
    let table = row_meta[r * 3u + 1u] * p.max_bt;
    let live = row_meta[r * 3u + 2u];
    let sel_base = r * p.sel_width;
    let lt = lid.x;

    // The group's query heads are contiguous: heads hkv*GROUP .. +GROUP.
    let q_base = r * q_row + hkv * GROUP * hd;
    for (var e = lt; e < GROUP * HD; e = e + WG) {
        let d = e % HD;
        qsh[e] = select(0.0, q[q_base + (e / HD) * hd + d], d < hd);
    }

    // Running softmax state per head lives in workgroup memory, owned by
    // thread `gh` during the softmax phase.
    if (lt < GROUP) {
        m_sh[lt] = -3.4e38;
        l_sh[lt] = 0.0;
    }
    // This thread's outputs are `lt + u * WG` over the group's GROUP * HD
    // channels: any head, so OPT need not divide a head.
    var o: array<f32, OPT>;
    for (var u = 0u; u < OPT; u = u + 1u) { o[u] = 0.0; }

    let ntiles = (count + BC - 1u) / BC;
    for (var kt = 0u; kt < ntiles; kt = kt + 1u) {
        // Resolve the tile's selected positions to pool slots once.
        if (lt < BC) {
            let i = kt * BC + lt;
            var slot = NO_SLOT;
            if (i < count) {
                let j = sel[sel_base + i];
                if (j < live) {
                    let physical = block_tables[table + j / p.block_size];
                    slot = (physical * p.block_size + (j % p.block_size)) * kv_row + hkv * hd;
                }
            }
            kslot[lt] = slot;
        }
        workgroupBarrier();

        // Stage K: for a fixed key, consecutive threads read consecutive
        // channels of its contiguous row.
        for (var e = lt; e < BC * HD; e = e + WG) {
            let base = kslot[e / HD];
            let d = e % HD;
            var kval = 0.0;
            if (base != NO_SLOT && d < hd) {
                let slot = base + d;
                kval = pool_k[slot];
            }
            tile[e] = kval;
        }
        workgroupBarrier();

        // Score dot products, each split over TPD threads.
        for (var s = 0u; s < DPT; s = s + 1u) {
            let dot = lt / TPD + s * (WG / TPD);
            if (dot >= NDOT) { break; }
            let piece = lt % TPD;
            let g = dot / BC;
            let j = dot % BC;
            var acc = 0.0;
            for (var c = piece * CPT; c < (piece + 1u) * CPT; c = c + 1u) {
                acc = acc + qsh[g * HD + c] * tile[j * HD + c];
            }
            part[dot * TPD + piece] = acc;
        }
        workgroupBarrier();

        // Thread gh < GROUP folds its head's scores into the online softmax:
        // the tile's weights, the rescale of what came before, the new
        // running max and sum. The rest stage V over K meanwhile.
        if (lt < GROUP) {
            let gh = lt;
            var tile_max = -3.4e38;
            for (var j = 0u; j < BC; j = j + 1u) {
                var v = -3.4e38;
                if (kslot[j] != NO_SLOT) {
                    var acc = 0.0;
                    for (var piece = 0u; piece < TPD; piece = piece + 1u) {
                        acc = acc + part[(gh * BC + j) * TPD + piece];
                    }
                    v = acc * scale;
                }
                pw[gh * BC + j] = v;
                tile_max = max(tile_max, v);
            }
            let m_old = m_sh[gh];
            let m_new = max(m_old, tile_max);
            var tile_l = 0.0;
            for (var j = 0u; j < BC; j = j + 1u) {
                // A masked key weighs nothing, even in a tile where every key
                // is masked and the running max is still the -inf stand-in.
                let e = select(0.0, exp(pw[gh * BC + j] - m_new), kslot[j] != NO_SLOT);
                pw[gh * BC + j] = e;
                tile_l = tile_l + e;
            }
            let corr = exp(m_old - m_new);
            corr_sh[gh] = corr;
            l_sh[gh] = l_sh[gh] * corr + tile_l;
            m_sh[gh] = m_new;
        }
        for (var e = lt; e < BC * HD; e = e + WG) {
            let base = kslot[e / HD];
            let d = e % HD;
            var vval = 0.0;
            if (base != NO_SLOT && d < hd) {
                let slot = base + d;
                vval = pool_v[slot];
            }
            tile[e] = vval;
        }
        workgroupBarrier();

        for (var u = 0u; u < OPT; u = u + 1u) {
            let e = lt + u * WG;
            if (e < GROUP * HD) {
                let gh = e / HD;
                let d = e % HD;
                var acc = 0.0;
                for (var j = 0u; j < BC; j = j + 1u) {
                    acc = acc + pw[gh * BC + j] * tile[j * HD + d];
                }
                o[u] = o[u] * corr_sh[gh] + acc;
            }
        }
        workgroupBarrier(); // done reading the tile before it is overwritten
    }

    for (var u = 0u; u < OPT; u = u + 1u) {
        let e = lt + u * WG;
        if (e < GROUP * HD) {
            let gh = e / HD;
            let d = e % HD;
            let l = l_sh[gh];
            if (d < hd) { ctx[r * q_row + (hkv * GROUP + gh) * hd + d] = select(0.0, o[u] / l, l > 0.0); }
        }
    }
}
