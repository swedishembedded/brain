// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Row scatter by index — the inverse of the `embed` row-gather for UNIQUE
// indices: out[idx[i], :] = src[i, :]. Rows of `out` not named by any index are
// left untouched (zero the buffer via submit `clears` when zeros are required,
// e.g. scattering the gathered MLM head's gradient back into the full-sequence
// hidden grad). Out-of-range indices are SKIPPED — padding a fixed-capacity
// index buffer with a sentinel (u32::MAX) makes the pad slots inert rather
// than colliding on a real row. No atomics: uniqueness of the in-range `idx`
// is the caller's contract.
//   total = n_idx * d.  One invocation per (i, c).

struct Params {
    n_idx: u32,
    d: u32,
    n_rows_out: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       idx: array<u32>;
@group(0) @binding(2) var<storage, read>       src: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.n_idx * p.d;
    if (i >= total) { return; }
    let c = i % p.d;
    let r = i / p.d;
    let dst = idx[r];
    if (dst >= p.n_rows_out) { return; }
    out[dst * p.d + c] = src[r * p.d + c];
}
