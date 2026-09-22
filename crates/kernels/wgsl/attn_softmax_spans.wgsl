// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row-wise softmax over RAGGED spans in one dispatch
// @how   one thread per score row, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `attn_softmax_cross` for the ragged-span layout: one invocation normalises
// one `[len]` score row of one span into probs. See `attn_scores_spans` for
// why this family exists, for the work table, and for the shared `Params`.
//
// The table entry here is counted in score ROWS (`heads * len` of them per
// span), not in score elements, so `elem0 + lane` is a row index and the row
// starts at `score_base + row * len`.

struct Params {
    n_wg: u32,
    n_heads: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       work:   array<u32>;
@group(0) @binding(2) var<storage, read>       scores: array<f32>;
@group(0) @binding(3) var<storage, read_write> probs:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch), split into
    // the workgroup this thread belongs to and its lane within it. See
    // `attn_scores_spans` for why this is derived from the GLOBAL id rather
    // than read from the workgroup builtins.
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let w = gidx / 64u;
    let lane = gidx % 64u;
    if (w >= p.n_wg) { return; }
    let e4 = 4u * w;
    let len = work[e4 + 1u];
    let sbase = work[e4 + 2u];
    let row = work[e4 + 3u] + lane;
    if (row >= p.n_heads * len) { return; }

    let base = sbase + row * len;
    var mx = -3.4e38;
    for (var j: u32 = 0u; j < len; j = j + 1u) { mx = max(mx, scores[base + j]); }
    var sum = 0.0;
    for (var j: u32 = 0u; j < len; j = j + 1u) { sum = sum + exp(scores[base + j] - mx); }
    let inv = 1.0 / sum;
    for (var j: u32 = 0u; j < len; j = j + 1u) {
        probs[base + j] = exp(scores[base + j] - mx) * inv;
    }
}
