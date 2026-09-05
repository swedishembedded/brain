// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused causal chunked paged-attention prefill at head_dim up to 256: BR query rows/workgroup, online softmax, no [nh,N,N] slab
// @how   256-thread workgroup per (head, query-tile), head_dim split into two 128-wide fragments STREAMED through one shared K/V tile, causal early-exit over K tiles
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// M2.5: `paged_flash_prefill`'s own `HD: u32 = 128u` caps it at `head_dim <=
// 128` (its own header: "max head_dim; tiles are always this wide"). Qwen3.8
// -27B (`qwen35::config::Qwen35Config::qwen38_27b()`) has `head_dim = 256`,
// so every GQA layer of that model stays on the three-stage materialized-
// score-slab triad today (`qwen35::serve`'s own `MAX_PREFILL_TOKENS` doc
// says exactly this: "the fused flash-prefill kernel ... does not fit this
// model's head_dim = 256 yet"). This is a SEPARATE kernel file, not an
// in-place edit of `paged_flash_prefill.wgsl` - the fallback this milestone's
// own instructions name when a lane-split rewrite is not confident enough to
// risk regressing every OTHER model already on the HD=128 fused path. Same
// tape, same `Params`, same tiling/masking CONTRACTS as that kernel (see its
// own header for the full derivation) - only the head_dim handling differs,
// documented below.
//
// THE ACTUAL CHANGE: `flash_attn_causal_gqa`'s lane-split (`LANES *
// CH == head_dim`, one Q/K/V slice held per lane in registers) assumes the
// WHOLE head_dim tile is staged into shared memory at once. Doing that
// naively at `head_dim = 256` doubles `ksh`/`vsh` from 4 KiB each to 8 KiB
// each - 16 KiB total for K+V alone, plus `part`'s own 8 KiB, is 24 KiB,
// over the WebGPU-guaranteed 16 KiB `maxComputeWorkgroupStorageSize` floor
// `paged_flash_prefill`/`flash_attn_causal_gqa` both sit exactly at
// (`backend_api::DeviceCaps::portable_baseline`'s own comment: "kernels
// declare at most @workgroup_size(256) and stage at most 16 KiB ... unless
// the adapter reports more"). Instead of doubling the tile, this kernel
// STREAMS two `HD0 = 128`-wide fragments (`[0,128)` then `[128,256)` of
// head_dim) through the SAME `ksh`/`vsh` buffers, one at a time - `ksh`/
// `vsh` stay sized `BC*HD0` exactly as `paged_flash_prefill` already has
// them, so total shared memory (`ksh` 4 KiB + `vsh` 4 KiB + `part` 8 KiB =
// 16 KiB) is IDENTICAL to that kernel's own budget, not doubled.
//
// WHY THIS IS SOUND, NOT JUST SMALLER: the two fragments cannot be run as
// two independent 128-wide attention passes and concatenated - the softmax
// statistics (row max, sum of exp) are a function of the FULL head_dim dot
// product `Q.K`, computed by summing across BOTH fragments, so a per-
// fragment max/sum would be wrong (a different distribution). This kernel
// therefore computes the score in two passes PER KV TILE - stage K
// fragment 0, accumulate its partial dot product into a per-thread register
// (`sfull`, not shared memory), then stage K fragment 1 over the SAME `ksh`
// buffer and add its partial - before touching softmax at all. Only once
// `sfull` holds the complete head_dim dot product does the online-softmax
// update (running max `m`, running sum `l`, per-tile weights `pj`) run,
// ONCE per tile, exactly as `paged_flash_prefill` already does. `pj` (the
// tile's softmax weights) is then shared, UNCHANGED, by both value
// fragments: P@V splits cleanly per fragment because V's head_dim is what
// is being PRODUCED (`o`'s two 128-wide halves are independent once the
// weights themselves are known), so fragment 0's V tile is staged, applied,
// and evicted from `vsh` before fragment 1's V tile reuses the same buffer -
// the "streaming" restructuring this milestone's own instructions asked for
// in preference to doubling `ksh`/`vsh`.
//
// MORE BARRIERS, NOT MORE SHARED MEMORY: each KV tile now does 2 K-stage +
// 2 V-stage round trips through the shared buffers instead of 1 combined
// K+V stage, so barrier count roughly doubles versus `paged_flash_prefill`
// (about 10 `workgroupBarrier()`s per tile here vs that kernel's 3) - an
// occupancy/instruction-count cost paid to keep the memory footprint flat,
// not a memory cost. `@cpu no` regardless (GPU-only by construction, same
// as its HD=128 sibling), so the CPU JIT's barrier-per-body limit does not
// apply.
//
//   q            : [bsz, n_heads*head_dim]           (bsz = chunk length, ONE sequence)
//   pool_k/pool_v: [num_blocks*block_size, n_kv_heads*head_dim]  (paged pool)
//   block_tables : [bsz, max_bt]  (physical block index per logical block)
//   seq_lens     : [bsz]          (live key count per query row)
//   ctx          : [bsz, n_heads*head_dim]
//
// Same two contracts `paged_flash_prefill`'s own header states in full
// (every row in a workgroup's `BR`-tile shares one physical block table,
// read through the tile's first row; `seq_lens` is non-decreasing across a
// tile, so the workgroup's largest live-key count is its last row's own
// value) - both still derive from "one prefill dispatch is one sequence's
// chunk", unaffected by the head_dim split.
//
// NOT bit-identical to the materialized triad it replaces, same precedent
// `paged_flash_prefill`/`paged_flash_decode` already cite: gated at `1e-3`
// absolute error, not `assert_eq`.

const BR: u32 = 64u;     // query rows per workgroup
const BC: u32 = 8u;      // key/value rows per shared tile
const LANES: u32 = 4u;   // threads cooperating on one query row
const HD0: u32 = 128u;   // width of ONE head_dim fragment - `ksh`/`vsh`'s own tile width, unchanged from `paged_flash_prefill`
const CH0: u32 = 32u;    // channels per lane WITHIN one fragment (LANES*CH0 == HD0)
const NFRAG: u32 = 2u;   // number of head_dim fragments streamed per KV tile (NFRAG*HD0 == max head_dim)
const HD: u32 = 256u;    // max head_dim this kernel supports (NFRAG*HD0)

struct Params {
    bsz: u32,          // chunk length (query rows), ONE sequence
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,     // <= 256
    group: u32,        // n_heads / n_kv_heads
    block_size: u32,
    max_bt: u32,       // block_tables row stride (blocks per sequence)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<f32>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<f32>;
@group(0) @binding(4) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(5) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(6) var<storage, read_write> ctx:          array<f32>;

var<workgroup> ksh:  array<f32, 1024>;  // BC*HD0 -> 4 KiB, reused once per fragment
var<workgroup> vsh:  array<f32, 1024>;  // BC*HD0 -> 4 KiB, reused once per fragment
var<workgroup> part: array<f32, 2048>;  // BC*BR*LANES -> 8 KiB, reused once per fragment

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let hd = p.head_dim;
    let scale = inverseSqrt(f32(hd));

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

    // This thread's slice of q/o, both fragments held at once (registers,
    // not shared memory - only ksh/vsh are streamed).
    var qv: array<f32, 64>;
    var o: array<f32, 64>;
    let q_base = i * q_row + h * hd;
    for (var frag = 0u; frag < NFRAG; frag = frag + 1u) {
        let foff = frag * HD0;
        for (var c = 0u; c < CH0; c = c + 1u) {
            let d = foff + c * LANES + lane;
            let idx = frag * CH0 + c;
            if (live && d < hd) { qv[idx] = q[q_base + d]; } else { qv[idx] = 0.0; }
            o[idx] = 0.0;
        }
    }

    var m = -3.4e38;   // running max
    var l = 0.0;       // running sum of exp

    let t_i = select(0u, seq_lens[i], live);

    let max_i_in_wg = min(qt * BR + BR - 1u, p.bsz - 1u);
    let max_t = seq_lens[max_i_in_wg];
    let ntiles_k = (max_t + BC - 1u) / BC;

    let bt_row = qt * BR;

    for (var kt = 0u; kt < ntiles_k; kt = kt + 1u) {
        // ---- Score: stream both head_dim fragments' K through `ksh`,
        // accumulating the FULL dot product in a per-thread register before
        // any softmax math runs (the statistics need the whole head_dim). ----
        var sfull: array<f32, 8>;
        for (var j = 0u; j < BC; j = j + 1u) { sfull[j] = 0.0; }

        for (var frag = 0u; frag < NFRAG; frag = frag + 1u) {
            let foff = frag * HD0;

            for (var e = lt; e < BC * HD0; e = e + 256u) {
                let jr = e / HD0;
                let dl = e % HD0;
                let j = kt * BC + jr;
                let d = foff + dl;
                if (j < max_t && d < hd) {
                    let physical = block_tables[bt_row * p.max_bt + j / p.block_size];
                    let slot = (physical * p.block_size + (j % p.block_size)) * kv_row + hkv * hd + d;
                    ksh[e] = pool_k[slot];
                } else {
                    ksh[e] = 0.0;
                }
            }
            workgroupBarrier();

            for (var j = 0u; j < BC; j = j + 1u) {
                var s = 0.0;
                let ko = j * HD0 + lane;
                for (var c = 0u; c < CH0; c = c + 1u) {
                    s = s + qv[frag * CH0 + c] * ksh[ko + c * LANES];
                }
                part[j * (BR * LANES) + row * LANES + lane] = s;
            }
            workgroupBarrier();

            for (var j = 0u; j < BC; j = j + 1u) {
                let po = j * (BR * LANES) + row * LANES;
                sfull[j] = sfull[j] + part[po] + part[po + 1u] + part[po + 2u] + part[po + 3u];
            }
            // Done reading `part` (this fragment) and `ksh` (this fragment)
            // before the next fragment (or the next tile's fragment 0)
            // restages either.
            workgroupBarrier();
        }

        // ---- Softmax: identical to `paged_flash_prefill`, fed by `sfull`
        // (the full head_dim dot product, both fragments already summed)
        // instead of a single-fragment partial. ----
        var krows = BC;
        let rem = max_t - kt * BC;
        if (rem < BC) { krows = rem; }
        var tmax = -3.4e38;
        var pj: array<f32, 8>;
        for (var j = 0u; j < BC; j = j + 1u) {
            var s = -3.4e38;
            if (j < krows && (kt * BC + j) < t_i) {
                s = sfull[j] * scale;
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

        // ---- P@V: stream both head_dim fragments' V through `vsh`. The
        // softmax weights `pj` are the SAME for both fragments (computed
        // once, above, from the full head_dim score) - only the output
        // accumulation splits per fragment, since V's head_dim is what is
        // being produced. ----
        for (var frag = 0u; frag < NFRAG; frag = frag + 1u) {
            let foff = frag * HD0;

            for (var e = lt; e < BC * HD0; e = e + 256u) {
                let jr = e / HD0;
                let dl = e % HD0;
                let j = kt * BC + jr;
                let d = foff + dl;
                if (j < max_t && d < hd) {
                    let physical = block_tables[bt_row * p.max_bt + j / p.block_size];
                    let slot = (physical * p.block_size + (j % p.block_size)) * kv_row + hkv * hd + d;
                    vsh[e] = pool_v[slot];
                } else {
                    vsh[e] = 0.0;
                }
            }
            workgroupBarrier();

            for (var c = 0u; c < CH0; c = c + 1u) {
                let vo = c * LANES + lane;
                var acc = 0.0;
                for (var j = 0u; j < BC; j = j + 1u) {
                    acc = acc + pj[j] * vsh[j * HD0 + vo];
                }
                let idx = frag * CH0 + c;
                o[idx] = o[idx] * corr + acc;
            }
            // Done reading `vsh` (this fragment) before the next fragment
            // (or the next tile's fragment 0 K stage) restages it.
            workgroupBarrier();
        }
    }

    if (live) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = i * q_row + h * hd;
        for (var frag = 0u; frag < NFRAG; frag = frag + 1u) {
            let foff = frag * HD0;
            for (var c = 0u; c < CH0; c = c + 1u) {
                let d = foff + c * LANES + lane;
                if (d < hd) { ctx[o_base + d] = o[frag * CH0 + c] * inv; }
            }
        }
    }
}
