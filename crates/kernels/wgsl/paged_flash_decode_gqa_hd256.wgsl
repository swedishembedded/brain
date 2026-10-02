// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Fused GQA decode attention at head_dim 256: one workgroup per (sequence, kv head, key split) attends every query head of the group and writes a partial online-softmax state
// @how   128-thread workgroup, 128-key tiles, a thread-per-key score pass over 16-byte row loads and a thread-per-channel-pair weighted sum
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @tpl   the `kv-tier` blocks below are the whole of what a compact KV cache
//        changes: `kernels::template::kv_tier_variant` swaps them for the bf16
//        (packed 16-bit halves) and int8 (packed bytes, one scale per (token,
//        kv-head) row) bindings and row loaders. Everything else is shared.
//
// Swedish Embedded AB implements long-context inference serving for clients
// whose GPU decides how many users one card carries. If your team needs
// expertise in decode attention that runs at the memory roofline over a
// compact KV cache, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// WHY THIS EXISTS. The batched decode triad (`paged_decode_scores_batched`,
// `decode_softmax_batched`, `paged_decode_apply_batched`) is one thread per
// output element with a serial reduction over the context: `decode_softmax_
// batched` is ONE thread per (sequence, head) walking every key, and the apply
// kernel is one thread per (sequence, head, channel) walking every key. At 128k
// tokens that is 24 resp. 6144 threads each looping 131072 times - measured on
// a GH200 at 0.7-1.4 s and 0.6-1.0 s PER DECODE STEP for ONE sequence (the 16
// GQA layers together), against ~6 ms for the bytes the step reads. It also
// reads every key and value once per QUERY head, six times over for Qwen3.8's
// group of six, and materialises a `[batch, n_heads, context]` score slab twice.
// A compact KV cache saves memory either way; it only saves TIME once the
// attention reads those bytes at bandwidth.
//
// THE SHAPE. FlashDecoding's split-key design (Dao et al.; the same two-pass
// shape `paged_flash_decode_split.wgsl` documents - no code from either is
// transcribed): the sequence's keys are cut into splits of `tiles_per_split`
// 128-key tiles, one workgroup per (sequence, kv head, split), so even ONE
// sequence fills the machine; each workgroup writes its partial `(o, m, l)`
// online-softmax state and `paged_flash_decode_gqa_combine` merges them. What
// is new against that kernel is that a workgroup owns a KV head and attends
// ALL `group` (<= 8) query heads that share it, so each key and value is read
// from memory once per group rather than once per head.
//
// THE THREAD MAP (128 threads, 128-key tiles).
//   scores  - thread `t` owns key `t` of the tile and accumulates ALL heads'
//             dot products over the key's 256 channels, 4 at a time: one
//             16-byte row load (`k4`) feeds `GRP` FMA quartets against `q` held in
//             workgroup memory (a broadcast read). Loading a row as 64
//             16-byte pieces instead of 256 scalars is what the thread-per-key
//             map needs: its 32 lanes read 32 different rows, so every load
//             instruction costs 32 L1 wavefronts whatever its width, and the
//             first version of this kernel (scalar loads, each key scored by
//             four threads) was bound at 95% of the L1/TEX pipe with DRAM at
//             8% - 3.3 ms per layer for a 128k context, f32 and compact tiers
//             alike, because every tier issued one load per ELEMENT.
//   softmax - thread `t` owns head `t / 16` and keys `t % 16 + 16 m`: a 4-step
//             tree reduces the tile's max and then its sum, per head.
//   apply   - thread `t` owns the channel PAIR `2t`, `2t + 1`: it walks the
//             tile's keys four at a time (four independent row loads in flight
//             before any is used), each a coalesced row across the workgroup,
//             and accumulates all `group` heads' outputs in registers.
//
// A tile's keys are looked up in the block table ONCE (`rowb`), by the threads
// that own them in the score phase, and read back by every thread in the apply
// phase. Keys past the sequence's live length are masked out of the scores
// (-3.4e38) and never loaded in the apply phase.
//
// MEMORY. qs 8 KiB + sp 4.5 KiB (rows padded to 144 floats, so the two heads
// a warp's softmax lanes cover hit disjoint banks) + red 0.5 KiB + rowb 0.5 KiB
// + 3 x 32 B ~= 13.7 KiB, under the 16 KiB WebGPU floor. 6 storage buffers
// (the int8 variant adds two scale buffers: 8, the WebGPU limit) - the partial
// states share one buffer with a `(head_dim + 2)`-float record per (sequence,
// head, split): the output channels, then `m`, then `l`.
//
//   q            : [batch, n_heads*256]
//   pool_k/pool_v: [num_blocks*block_size, n_kv_heads*256]  (paged pool)
//   block_tables : [batch, max_bt]
//   seq_lens     : [batch]   live keys per sequence, the new token included
//   part         : [batch, n_heads, n_splits, 258]
//
// A split past a sequence's live length runs zero tiles and writes the
// degenerate `(o = 0, m = -3.4e38, l = 0)` record, which the combine pass
// folds in as a no-op. `p.head_dim` must be 256 and `p.group` must equal the
// compile-time `GRP` (at most 8): the loops over a group's heads are fully
// unrolled into registers, which a runtime bound would not allow, so the model
// registers one specialisation per group it has (`kernels::template::
// specialize` on `GRP`; Qwen3.8's is 6) and the host checks both. NOT
// bit-identical to the triad (a different reduction order); gated against exact
// f64 attention at 2e-4 relative.

const HD: u32 = 256u;
const HD4: u32 = 64u;      // 4-channel pieces per head row
const TILE: u32 = 128u;    // keys per tile, and threads per workgroup
const GMAX: u32 = 8u;      // most query heads one kv head may serve
const GRP: u32 = 6u;       // query heads per kv head - specialised per model, see above
const WG: u32 = 128u;
const SP: u32 = 144u;      // row stride of `sp`: TILE plus 16 pad floats

struct Params {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,        // == 256
    group: u32,           // n_heads / n_kv_heads, == GRP (the host checks)
    block_size: u32,
    max_bt: u32,          // block_tables row stride
    n_splits: u32,        // splits per (sequence, head)
    tiles_per_split: u32, // 128-key tiles one split's workgroup walks
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
// @kv-tier-begin decls
@group(0) @binding(2) var<storage, read>       pool_k:       array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       pool_v:       array<vec2<f32>>;
// @kv-tier-end decls
@group(0) @binding(4) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(5) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(6) var<storage, read_write> part:         array<f32>;

// @kv-tier-begin loads
// `row` is the element index of a 256-element key (or value) row in the pool,
// a multiple of 256. `k4` is channels `4i .. 4i + 3` of a key row, `v2`
// channels `2i, 2i + 1` of a value row, both widened to f32; the row scale is
// 1 for a plain cache.
fn k4(row: u32, i: u32) -> vec4<f32> { return pool_k[(row >> 2u) + i]; }
fn v2(row: u32, i: u32) -> vec2<f32> { return pool_v[(row >> 1u) + i]; }
fn k_scale(row: u32) -> f32 { return 1.0; }
fn v_scale(row: u32) -> f32 { return 1.0; }
// @kv-tier-end loads

var<workgroup> qs:   array<vec4<f32>, 512>;  // GMAX*HD4, this group's queries
var<workgroup> sp:   array<f32, 1152>;       // GMAX*SP, a tile's scores then probabilities
var<workgroup> red:  array<f32, 128>;        // reduction scratch
var<workgroup> rowb: array<u32, 128>;        // element index of each tile key's row in the pool
var<workgroup> mrun: array<f32, 8>;          // running max per head
var<workgroup> lrun: array<f32, 8>;          // running sum per head
var<workgroup> corr: array<f32, 8>;          // this tile's rescale of the accumulators per head

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let wg = wgid.y * nwg.x + wgid.x;
    let s = wg % p.n_splits;
    let wg2 = wg / p.n_splits;
    let kvh = wg2 % p.n_kv_heads;
    let b = wg2 / p.n_kv_heads;
    // Uniform across the workgroup, so returning before the first barrier is
    // safe.
    if (b >= p.batch) { return; }

    let kv_row = p.n_kv_heads * HD;
    let scale = inverseSqrt(f32(HD));
    let t = lid.x;

    // Stage this group's queries, 4 channels at a time.
    for (var e = t; e < GRP * HD4; e = e + WG) {
        let g = e / HD4;
        let i = e % HD4;
        let base = (b * p.n_heads + kvh * GRP + g) * HD + i * 4u;
        qs[e] = vec4<f32>(q[base], q[base + 1u], q[base + 2u], q[base + 3u]);
    }
    if (t < GMAX) {
        mrun[t] = -3.4e38;
        lrun[t] = 0.0;
    }
    workgroupBarrier();

    let tlen = seq_lens[b];
    let ntiles = (tlen + TILE - 1u) / TILE;
    let kt0 = s * p.tiles_per_split;
    let kt_end = min(kt0 + p.tiles_per_split, ntiles);

    // This thread's channel pair of every head's output.
    var o: array<vec2<f32>, 8>;
    for (var g = 0u; g < GRP; g = g + 1u) { o[g] = vec2<f32>(0.0, 0.0); }

    let sg = t / 16u;       // softmax: this thread's head
    let si = t % 16u;       // and its lane within that head's 16 threads

    for (var kt = kt0; kt < kt_end; kt = kt + 1u) {
        // ---- scores: key `t` of the tile, every head of the group.
        let j = kt * TILE + t;
        let live = j < tlen;
        var rb = 0u;
        var sc: array<f32, 8>;
        for (var g = 0u; g < GRP; g = g + 1u) { sc[g] = 0.0; }
        if (live) {
            let physical = block_tables[b * p.max_bt + j / p.block_size];
            rb = (physical * p.block_size + (j % p.block_size)) * kv_row + kvh * HD;
            // Four 4-channel pieces per pass, so a tier that loads wider than
            // one piece (int8: 16 channels to a 16-byte load) issues ONE load
            // for the four `k4` calls below, whose index it shares.
            for (var i0 = 0u; i0 < HD4; i0 = i0 + 4u) {
                let k0 = k4(rb, i0);
                let k1 = k4(rb, i0 + 1u);
                let k2 = k4(rb, i0 + 2u);
                let k3 = k4(rb, i0 + 3u);
                // Explicit `fma`: the CUDA build compiles with contraction off
                // (`--fmad=false`), so `a * b + c` is two instructions.
                for (var g = 0u; g < GRP; g = g + 1u) {
                    let q0 = qs[g * HD4 + i0];
                    let q1 = qs[g * HD4 + i0 + 1u];
                    let q2 = qs[g * HD4 + i0 + 2u];
                    let q3 = qs[g * HD4 + i0 + 3u];
                    var a = sc[g];
                    a = fma(q0.x, k0.x, a); a = fma(q0.y, k0.y, a); a = fma(q0.z, k0.z, a); a = fma(q0.w, k0.w, a);
                    a = fma(q1.x, k1.x, a); a = fma(q1.y, k1.y, a); a = fma(q1.z, k1.z, a); a = fma(q1.w, k1.w, a);
                    a = fma(q2.x, k2.x, a); a = fma(q2.y, k2.y, a); a = fma(q2.z, k2.z, a); a = fma(q2.w, k2.w, a);
                    a = fma(q3.x, k3.x, a); a = fma(q3.y, k3.y, a); a = fma(q3.z, k3.z, a); a = fma(q3.w, k3.w, a);
                    sc[g] = a;
                }
            }
        }
        rowb[t] = rb;
        let kmul = k_scale(rb) * scale;
        for (var g = 0u; g < GRP; g = g + 1u) {
            sp[g * SP + t] = select(-3.4e38, sc[g] * kmul, live);
        }
        workgroupBarrier();

        // ---- softmax: head `sg`, keys `si + 16 m`. Max first.
        var lmax = -3.4e38;
        if (sg < GRP) {
            for (var m = 0u; m < 8u; m = m + 1u) { lmax = max(lmax, sp[sg * SP + si + 16u * m]); }
        }
        red[t] = lmax;
        workgroupBarrier();
        for (var st = 8u; st > 0u; st = st >> 1u) {
            if (si < st) { red[t] = max(red[t], red[t + st]); }
            workgroupBarrier();
        }
        let tmax = red[sg * 16u];
        // Every thread must have read its head's max before the sum below
        // overwrites `red`.
        workgroupBarrier();

        let m_old = mrun[sg];
        let m_new = max(m_old, tmax);
        var lsum = 0.0;
        if (sg < GRP) {
            for (var m = 0u; m < 8u; m = m + 1u) {
                let at = sg * SP + si + 16u * m;
                let e = exp(sp[at] - m_new);
                sp[at] = e;
                lsum = lsum + e;
            }
        }
        red[t] = lsum;
        workgroupBarrier();
        for (var st = 8u; st > 0u; st = st >> 1u) {
            if (si < st) { red[t] = red[t] + red[t + st]; }
            workgroupBarrier();
        }
        if (si == 0u && sg < GRP) {
            let cr = exp(m_old - m_new);
            corr[sg] = cr;
            lrun[sg] = lrun[sg] * cr + red[sg * 16u];
            mrun[sg] = m_new;
        }
        workgroupBarrier();

        // ---- apply: channel pair `t`, every head, four keys at a time.
        for (var g = 0u; g < GRP; g = g + 1u) { o[g] = o[g] * corr[g]; }
        let nlive = min(TILE, tlen - kt * TILE);
        var rr = 0u;
        for (; rr + 4u <= nlive; rr = rr + 4u) {
            let r0 = rowb[rr];
            let r1 = rowb[rr + 1u];
            let r2 = rowb[rr + 2u];
            let r3 = rowb[rr + 3u];
            let v0 = v2(r0, t) * v_scale(r0);
            let v1 = v2(r1, t) * v_scale(r1);
            let v2_ = v2(r2, t) * v_scale(r2);
            let v3 = v2(r3, t) * v_scale(r3);
            for (var g = 0u; g < GRP; g = g + 1u) {
                let a0 = sp[g * SP + rr];
                let a1 = sp[g * SP + rr + 1u];
                let a2 = sp[g * SP + rr + 2u];
                let a3 = sp[g * SP + rr + 3u];
                let ox = fma(v3.x, a3, fma(v2_.x, a2, fma(v1.x, a1, fma(v0.x, a0, o[g].x))));
                let oy = fma(v3.y, a3, fma(v2_.y, a2, fma(v1.y, a1, fma(v0.y, a0, o[g].y))));
                o[g] = vec2<f32>(ox, oy);
            }
        }
        for (; rr < nlive; rr = rr + 1u) {
            let r0 = rowb[rr];
            let v0 = v2(r0, t) * v_scale(r0);
            for (var g = 0u; g < GRP; g = g + 1u) {
                let a0 = sp[g * SP + rr];
                o[g] = vec2<f32>(fma(v0.x, a0, o[g].x), fma(v0.y, a0, o[g].y));
            }
        }
        // `sp`, `red` and `rowb` are rewritten by the next tile.
        workgroupBarrier();
    }

    for (var g = 0u; g < GRP; g = g + 1u) {
        let base = ((b * p.n_heads + kvh * GRP + g) * p.n_splits + s) * (HD + 2u);
        part[base + 2u * t] = o[g].x;
        part[base + 2u * t + 1u] = o[g].y;
    }
    if (t < GRP) {
        let base = ((b * p.n_heads + kvh * GRP + t) * p.n_splits + s) * (HD + 2u);
        part[base + HD] = mrun[t];
        part[base + HD + 1u] = lrun[t];
    }
}
