// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Native sparse attention: the selection's inverse from (key row, query) pairs sorted by key - per-key offsets and query lists
// @how   one thread per sorted pair (plus one past the end): the key boundaries it straddles get its position as their offset
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype n/a
//
// Swedish Embedded AB implements long-context models trained with sparse
// attention in time and memory linear in the context, for clients whose
// sequences outgrow quadratic attention. If your team needs expertise in
// sparse attention training kernels, you can procure our services by sending
// an email to info@swedishembedded.com.
//
// The last step of inverting a native selection on the device, after
// `lsa_native_pairs` and a stable sort of its pairs by key. The index holds
// `[rows, stride]` selection rows, then `inv_off` (`rows + 1` words), then
// `inv`. Thread `q` (of `pairs + 1`) takes the key `k` of pair `q` (`rows`
// for padding or past the end) and the key `k'` of pair `q - 1` (-1 before
// the first): every key row in `(k', k]` starts at `q`, so each offset is
// written exactly once and `inv_off[rows]` is the count of selected keys;
// the pair's query is copied to `inv[q]`. Keys run in order and a key's
// queries ascend (the sort is stable), so the result is deterministic.

struct Params {
    rows: u32,
    pairs: u32,
    stride: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       keys:  array<u32>;
@group(0) @binding(2) var<storage, read>       vals:  array<u32>;
@group(0) @binding(3) var<storage, read_write> index: array<u32>;

fn key_at(q: u32) -> u32 {
    if (q >= p.pairs) { return p.rows; }
    return min(keys[q], p.rows);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let q = gid.y * (nwg.x * 64u) + gid.x;
    if (q > p.pairs) { return; }
    let off = p.rows * p.stride;
    let inv = off + p.rows + 1u;
    let k = key_at(q);
    var first = 0u;
    if (q > 0u) { first = key_at(q - 1u) + 1u; }
    for (var kk = first; kk <= k; kk = kk + 1u) {
        index[off + kk] = q;
    }
    if (k < p.rows) {
        index[inv + q] = vals[q];
    }
}
