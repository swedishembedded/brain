// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Row-wise argmax:  out[m] = f32(argmax_n x[m, n])   (tie -> LOWEST n)
//
//   x  : [M, N] row-major
//   out: [M]    row-major, the winning column index as f32
//
// Used to pick the greedy next token from decode logits without shipping the
// whole [batch, vocab] logit block back to the host: the head matmul stays on
// the device and only M indices are read back.
//
// One invocation per ROW (not per element), because an argmax must reduce a
// whole row and this engine has no atomics or workgroup barriers to do a tree
// reduction with. That is the right trade here: the expensive part is the head
// matmul (M*N*K), which `matmul` already parallelises over M*N; this pass is a
// single linear scan of M*N values.
//
// The index is returned as f32 rather than u32 so it rides the engine's
// existing f32-only read-back path. Exact for any vocabulary up to 2^24
// (16.7M), far beyond real tokenizer sizes.

struct Params {
    m: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let m = gid.y * (nwg.x * 64u) + gid.x;
    if (m >= p.m) { return; }
    let base = m * p.n;
    var best_n = 0u;
    var best_v = -3.402823e38; // -f32::MAX
    for (var n: u32 = 0u; n < p.n; n = n + 1u) {
        let v = x[base + n];
        if (v > best_v) {
            best_v = v;
            best_n = n;
        }
    }
    out[m] = f32(best_n);
}
