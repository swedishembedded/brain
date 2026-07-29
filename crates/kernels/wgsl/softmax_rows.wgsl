// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Row softmax, one WORKGROUP per row — the long-context attention variant.
//
//   scores: [rows, cols]   probs: [rows, cols]
//   params: rows, cols
//
// attn_softmax_cross assigns one THREAD per row: at [H·chunk, 8192] that is
// 16k serial threads of 3×8192 dependent global loads on a 3840-core card —
// measured as the dominant quadratic term of an 8k encoder forward once the
// score/apply GEMMs went register-tiled. Here 64 threads cooperate per row
// (strided max reduction, strided exp+sum, normalize). Uses MULTIPLE
// workgroup barriers — GPU-only by construction: select it behind
// `DeviceCaps::workgroup_reductions`; the CPU JIT path keeps
// attn_softmax_cross (whose native fast path is rayon-parallel).
//
// Dispatch: rows * 64 invocations (one workgroup per row).

struct Params {
    rows: u32,
    cols: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs:  array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.cols;

    var mx = -3.4e38;
    for (var c = t; c < p.cols; c = c + 64u) {
        mx = max(mx, scores[base + c]);
    }
    partial[t] = mx;
    workgroupBarrier();
    var rowmax = -3.4e38;
    for (var i = 0u; i < 64u; i = i + 1u) {
        rowmax = max(rowmax, partial[i]);
    }
    workgroupBarrier();

    var sum = 0.0;
    for (var c = t; c < p.cols; c = c + 64u) {
        let e = exp(scores[base + c] - rowmax);
        probs[base + c] = e;
        sum = sum + e;
    }
    partial[t] = sum;
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        total = total + partial[i];
    }
    let inv = 1.0 / max(total, 1e-38);
    for (var c = t; c < p.cols; c = c + 64u) {
        probs[base + c] = probs[base + c] * inv;
    }
}
