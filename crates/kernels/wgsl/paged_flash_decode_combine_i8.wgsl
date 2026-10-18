// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Combine phase of the INT8-KV split-key FlashDecode: merges the live key-split partial online-softmax states `paged_flash_decode_split_i8` wrote into one normalised context per (sequence, head)
// @how   one thread per (sequence, head, channel), serial merge over the sequence's live splits
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements long-context inference serving for clients
// whose GPU decides how many users one card carries. If your team needs
// expertise in decode attention that runs at the memory roofline over a
// quantized KV cache, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The second pass of `paged_flash_decode_split_i8`. Each thread owns one
// `(b, h, d)` output channel and folds the partial states of its
// sequence-head with the two-term online-softmax merge
// `paged_flash_decode_combine.wgsl` documents:
//
//   m_new = max(m, m_i);  corr = exp(m - m_new);  corr_i = exp(m_i - m_new)
//   l = l * corr + l_i * corr_i;  o = o * corr + o_i * corr_i;  m = m_new
//
// Only the splits that did work are read. The split kernel divides a
// sequence's `ceil(len / 128)` chunks over `n_splits` workgroups as
// `cps = ceil(chunks / n_splits)` contiguous chunks each, so split `s` is live
// exactly when `s * cps < chunks`, i.e. there are `ceil(chunks / cps)` live
// splits (none for an empty sequence, whose context is zero). The records of
// the other splits were never written and must not be read. `CHUNK` and this
// arithmetic are the split kernel's contract.
//
//   part     : [batch, n_heads, n_splits, 2 + head_dim]  (m, l, then o)
//   seq_lens : [batch]
//   ctx      : [batch, n_heads * head_dim]                (normalised output)

const HD: u32 = 128u;
const PART_STRIDE: u32 = 2u + HD;
const CHUNK: u32 = 128u;   // keys per chunk of paged_flash_decode_split_i8

struct Params {
    batch: u32,
    n_heads: u32,
    head_dim: u32,
    n_splits: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part:     array<f32>;
@group(0) @binding(2) var<storage, read>       seq_lens: array<u32>;
@group(0) @binding(3) var<storage, read_write> ctx:      array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 128 threads per (b, h), one per output channel `d`.
    let idx = gid.y * (nwg.x * 128u) + gid.x;
    let d = idx % 128u;
    let wg2 = idx / 128u;
    let h = wg2 % p.n_heads;
    let b = wg2 / p.n_heads;
    if (b >= p.batch) { return; }

    let hd = p.head_dim;
    let nchunks = (seq_lens[b] + CHUNK - 1u) / CHUNK;
    var live = 0u;
    if (nchunks > 0u) {
        let cps = (nchunks + p.n_splits - 1u) / p.n_splits;
        live = (nchunks + cps - 1u) / cps;
    }
    let base = (b * p.n_heads + h) * p.n_splits;

    var m = -3.4e38;
    var l = 0.0;
    var o = 0.0;
    for (var s = 0u; s < live; s = s + 1u) {
        let pbase = (base + s) * PART_STRIDE;
        let pm = part[pbase];
        let pl = part[pbase + 1u];
        let m_new = max(m, pm);
        let corr = exp(m - m_new);
        let corr_i = exp(pm - m_new);
        l = l * corr + pl * corr_i;
        if (d < hd) {
            o = o * corr + part[pbase + 2u + d] * corr_i;
        }
        m = m_new;
    }

    if (d < hd) {
        let inv = select(0.0, 1.0 / l, l > 0.0);
        ctx[(b * p.n_heads + h) * hd + d] = o * inv;
    }
}
