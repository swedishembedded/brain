// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched paged decode scores, one WORKGROUP per score - the coalesced variant
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Batched paged decode scores, one WORKGROUP per (sequence, head, key) — the
// coalesced variant of `paged_decode_scores_batched`.
//
// Identical contract, identical output: for each sequence `b`, its single query
// attends all `seq_lens[b]` cached keys through that sequence's block table.
//   q      : [batch, n_heads*head_dim]
//   scores : [batch, n_heads, cap]        (only j < seq_lens[b] written)
//
// WHY. The per-element kernel gives thread `t` one score and walks `head_dim`
// serially. The fast thread axis is `j`, and a key's address is
// `(physical*block_size + j%block_size)*kv_stride + kvh*head_dim`, so two
// adjacent lanes are `kv_stride` floats apart — **4 KB** at Qwen3-0.6B's
// `kv_stride = 1024`. At every step of the dot product a warp's 32 lanes touch
// 32 different 32-byte sectors and use 4 bytes of each: 8x read amplification,
// which is why it measured **35.1 GB/s, 12.2% of a P40's bandwidth roof, while
// taking 52.2% of a served step**. This is the same defect a measured pass
// records as "one thread per row is a COALESCING bug", and the same fix: let
// a workgroup own one output and split the
// REDUCTION across its lanes, so consecutive lanes read consecutive addresses.
//
// A workgroup owns `64 / LPS` scores, and `LPS` lanes split each score's
// `head_dim` reduction. Lane `l` of a score accumulates `d = l, l+LPS, …`, so
// consecutive lanes read consecutive addresses and a warp still touches only
// fully-used sectors (`LPS * 4` bytes each).
//
// WHY NOT 64 LANES PER SCORE, which is what this kernel did first. With
// `head_dim = 128` that gives each lane **2 MACs** and then a **64-add serial
// fold on lane 0 while 63 lanes idle** — a serial tail far longer than the
// useful work, and it left the kernel at 23.7% of the bandwidth roof even after
// the coalescing fix. At `LPS = 8` each lane does 16 MACs and the fold is 8
// adds done by 8 lanes in parallel. Coalescing is unchanged: a 32-lane warp
// covers 4 scores, each reading a contiguous 32-byte run.
//
// Exactly ONE top-level `workgroupBarrier()` — the CPU JIT splits a body at one
// and no more.
//
// Dispatch: `ceil(scores / (64/LPS)) * 64` threads. Params are byte-identical to `paged_decode_scores_batched`, so the
// two are interchangeable at the call site and the selector picks between them
// on the queried `DeviceCaps::workgroup_reductions`.

struct Params {
    batch: u32,
    n_heads: u32,
    group: u32,
    head_dim: u32,
    block_size: u32,
    kv_stride: u32,
    cap: u32,       // scores row stride (>= max seq len)
    max_bt: u32,    // block_tables row stride (blocks per sequence)
    scale: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<f32>;
@group(0) @binding(3) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(4) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(5) var<storage, read_write> scores:       array<f32>;

// Lanes cooperating on one score. SWEPT on a P40 at Qwen3-0.6B (head_dim 128),
// never guessed:
//
//     LPS      64      16       8       4       2
//     ms    56.37   19.31   17.73   17.24   28.65
//     %roof  23.7%   69.2%   75.4%   77.5%   46.7%
//
// Unimodal, and both ends are explained. High LPS starves each lane (at 64 it is
// 2 MACs then a 64-add serial fold on lane 0 while 63 idle). Low LPS breaks
// coalescing: at LPS=2 a lane group covers 8 bytes, so a 32-byte sector is only
// a quarter used and the kernel falls back to 46.7%. 4 is the crossover where a
// lane group still covers a whole 16-byte run and each lane does 32 MACs.
const LPS: u32 = 4u;
/// Scores one workgroup owns.
const SPW: u32 = 64u / LPS;

var<workgroup> part: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe: `t` is the flat invocation id, so `t / 64` is this
    // workgroup's output and `t % 64` is the lane within it. Reconstructed the
    // same way as every other kernel in the tree rather than from a workgroup
    // builtin, so a 2D-tiled dispatch (this one exceeds 65535 groups easily)
    // needs no special case.
    let t = gid.y * (nwg.x * 64u) + gid.x;
    let lane = t % 64u;
    let slot_in_wg = lane / LPS;   // which of this workgroup's scores
    let l = lane % LPS;            // this lane's offset within that score
    let idx = (t / 64u) * SPW + slot_in_wg;

    let total = p.batch * p.n_heads * p.cap;
    // Every lane must reach the barrier, so out-of-range and past-the-end
    // workgroups fall through to it rather than returning early.
    let live = idx < total;
    let b = select(0u, idx / (p.n_heads * p.cap), live);
    let rem = select(0u, idx % (p.n_heads * p.cap), live);
    let h = rem / p.cap;
    let j = rem % p.cap;
    let attend = live && j < seq_lens[b];

    var s = 0.0;
    if (attend) {
        let hd = p.head_dim;
        let kvh = h / p.group;
        let physical = block_tables[b * p.max_bt + j / p.block_size];
        let slot = (physical * p.block_size + (j % p.block_size)) * p.kv_stride + kvh * hd;
        let qb = (b * p.n_heads + h) * hd;
        // Consecutive lanes read consecutive `d` — the whole point.
        for (var d: u32 = l; d < hd; d = d + LPS) {
            s = s + q[qb + d] * pool_k[slot + d];
        }
    }
    part[lane] = s;
    workgroupBarrier();

    if (attend && l == 0u) {
        var acc = 0.0;
        let base = slot_in_wg * LPS;
        for (var i: u32 = 0u; i < LPS; i = i + 1u) {
            acc = acc + part[base + i];
        }
        scores[(b * p.n_heads + h) * p.cap + j] = acc * p.scale;
    }
}
