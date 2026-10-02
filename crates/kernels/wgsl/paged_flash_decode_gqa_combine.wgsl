// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Combine phase of the fused GQA decode attention: merges the key-split partial online-softmax records into one normalised context per (sequence, head)
// @how   one thread per (sequence, head, channel), serial merge over the splits
// @opt   2
// @cpu   yes
// @gpu   yes-wg256
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements long-context inference serving for clients
// whose GPU decides how many users one card carries. If your team needs
// expertise in decode attention that runs at the memory roofline over a
// compact KV cache, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The second pass of `paged_flash_decode_gqa_hd256`: each thread owns one
// `(b, h, d)` output channel and folds the `n_splits` records of its
// sequence-head with the two-term online-softmax merge
// `paged_flash_decode_combine.wgsl` documents (same identity, a different
// record layout):
//
//   m_new = max(m, m_i);  corr = exp(m - m_new);  corr_i = exp(m_i - m_new)
//   l = l * corr + l_i * corr_i;  o = o * corr + o_i * corr_i;  m = m_new
//
// A split that saw no live key wrote `(o = 0, m = -3.4e38, l = 0)`, so its
// `corr_i` is 0 and it folds in as a no-op. No shared memory and no barrier:
// every thread's walk is independent, the `m`/`l` reads are broadcast.
//
//   part : [batch, n_heads, n_splits, head_dim + 2]  (channels, m, l)
//   ctx  : [batch, n_heads*head_dim]

struct Params {
    batch: u32,
    n_heads: u32,
    head_dim: u32,   // <= 256
    n_splits: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       part: array<f32>;
@group(0) @binding(2) var<storage, read_write> ctx:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 256u) + gid.x;
    let hd = p.head_dim;
    let d = idx % 256u;
    let bh = idx / 256u;
    if (bh >= p.batch * p.n_heads || d >= hd) { return; }

    let stride = hd + 2u;
    var m = -3.4e38;
    var l = 0.0;
    var o = 0.0;
    for (var s = 0u; s < p.n_splits; s = s + 1u) {
        let rec = (bh * p.n_splits + s) * stride;
        let pm = part[rec + hd];
        let pl = part[rec + hd + 1u];
        let m_new = max(m, pm);
        let corr = exp(m - m_new);
        let corr_i = exp(pm - m_new);
        l = l * corr + pl * corr_i;
        o = o * corr + part[rec + d] * corr_i;
        m = m_new;
    }
    ctx[bh * hd + d] = o * select(0.0, 1.0 / l, l > 0.0);
}
