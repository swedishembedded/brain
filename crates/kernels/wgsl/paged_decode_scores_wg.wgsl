// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched paged decode scores, one WORKGROUP per score — the coalesced variant
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
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
// taking 52.2% of a served step**. This is the same defect
// `docs/performance/overview.md` records as "one thread per row is a COALESCING
// bug", and the same fix: let a workgroup own one output and split the
// REDUCTION across its lanes, so consecutive lanes read consecutive addresses.
//
// Lane `t` accumulates `d = t, t+64, t+128, …`, so both `q` and `pool_k` are
// read fully coalesced. Exactly ONE top-level `workgroupBarrier()` — the CPU
// JIT splits a body at one and no more (`docs/lessons.md` #26) — after which
// every lane redundantly folds the 64 partials (64 adds, cheaper than a second
// barrier) and lane 0 writes.
//
// Dispatch: `batch * n_heads * cap * 64` threads, i.e. one 64-thread workgroup
// per score. Params are byte-identical to `paged_decode_scores_batched`, so the
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
    let idx = t / 64u;

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
        for (var d: u32 = lane; d < hd; d = d + 64u) {
            s = s + q[qb + d] * pool_k[slot + d];
        }
    }
    part[lane] = s;
    workgroupBarrier();

    if (attend && lane == 0u) {
        var acc = 0.0;
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            acc = acc + part[i];
        }
        scores[(b * p.n_heads + h) * p.cap + j] = acc * p.scale;
    }
}
