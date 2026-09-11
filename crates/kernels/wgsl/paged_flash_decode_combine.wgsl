// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Combine phase of a two-pass FlashDecode: merges the N key-split partial online-softmax states the split phase wrote into one normalised context per (sequence, head)
// @how   one thread per (sequence, head, channel), serial reduction over splits
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Second pass of M2.7's split-key FlashDecode (see `paged_flash_decode_
// split.wgsl`'s header for why the split exists and what it targets). Each
// thread owns one `(b, h, d)` output channel and walks that sequence-head's
// `p.n_splits` partial `(m, l, o)` triples with the textbook two-term
// online-softmax MERGE (not the tile-local rescale `paged_flash_decode`'s
// own per-tile update uses, though it is the identical algebraic identity
// applied one level up - both are instances of the same
// log-sum-exp-with-a-running-max reassociation `flash_attn_causal_gqa.wgsl`
// already documents in this tree):
//
//   m_new  = max(m, m_i)
//   corr   = exp(m   - m_new)      // rescales the accumulator so far
//   corr_i = exp(m_i - m_new)      // rescales split i's own contribution
//   l      = l * corr + l_i * corr_i
//   o      = o * corr + o_i * corr_i
//   m      = m_new
//
// A split whose workgroup found no live keys in its range wrote the
// degenerate `(m_i = -3.4e38, l_i = 0.0, o_i = 0.0)` triple (`paged_flash_
// decode_split`'s own header names this) - `corr_i = exp(m_i - m_new)`
// evaluates to (at worst) `exp(-3.4e38 - m_new)`, i.e. 0 for any finite
// `m_new`, so it folds in as a true no-op with no special case needed here
// either.
//
// This same two-pass split/merge shape is `flash-decoding`'s (Dao et al.)
// and vLLM's `paged_attention_v2` partition-then-reduce design; no code from
// either is transcribed, only the same well-known merge identity re-derived
// from this repo's own `paged_flash_decode.wgsl` (its per-tile update is
// this exact formula one level down).
//
// No shared memory, no barrier: every thread's `m`/`l` walk is fully
// independent of every other thread (each reads only `part_m`/`part_l` at
// its own `(b, h)`, redundantly across every `d` in the workgroup - cheap
// global reads, not a reduction worth cooperating over at this size), and
// `o` is a per-channel register that never needs another thread's value. A
// plain per-invocation kernel, unlike its `@cpu no` split-phase sibling.
//
//   part_m/part_l : [batch, n_heads, n_splits]            (from the split phase)
//   part_o        : [batch, n_heads, n_splits, head_dim]  (from the split phase)
//   ctx           : [batch, n_heads*head_dim]              (normalised output)

struct Params {
    batch: u32,
    n_heads: u32,
    head_dim: u32,
    n_splits: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part_m: array<f32>;
@group(0) @binding(2) var<storage, read>       part_l: array<f32>;
@group(0) @binding(3) var<storage, read>       part_o: array<f32>;
@group(0) @binding(4) var<storage, read_write> ctx:    array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // Flat thread id -> (b, h, d), 128 threads per (b, h) - a plain
    // per-invocation kernel (no barrier, no shared memory), so this follows
    // `decode_softmax_batched.wgsl`'s own `global_invocation_id` convention
    // rather than `paged_flash_decode_split`'s workgroup_id/local_invocation_id
    // one (that kernel is workgroup-cooperative; this one is not).
    let idx = gid.y * (nwg.x * 128u) + gid.x;
    let d = idx % 128u;
    let wg2 = idx / 128u;
    let h = wg2 % p.n_heads;
    let b = wg2 / p.n_heads;
    if (b >= p.batch) { return; }

    let hd = p.head_dim;
    let base = (b * p.n_heads + h) * p.n_splits;

    var m = -3.4e38;
    var l = 0.0;
    var o = 0.0;
    for (var s = 0u; s < p.n_splits; s = s + 1u) {
        let pm = part_m[base + s];
        let pl = part_l[base + s];
        let m_new = max(m, pm);
        let corr = exp(m - m_new);
        let corr_i = exp(pm - m_new);
        l = l * corr + pl * corr_i;
        if (d < hd) {
            let po = part_o[(base + s) * hd + d];
            o = o * corr + po * corr_i;
        }
        m = m_new;
    }

    if (d < hd) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        let o_base = (b * p.n_heads * hd) + h * hd;
        ctx[o_base + d] = o * inv;
    }
}
