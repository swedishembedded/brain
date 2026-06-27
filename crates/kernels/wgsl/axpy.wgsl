// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Scaled accumulate:  out[i] = out[i] + s * in[i].
// Used to fuse a LoRA adapter delta into a base projection's output
// (y += (alpha/rank) * (B @ (A @ x))) and as the scaled-copy in its backward.
// Elementwise; `out` is read-modify-write, `in` is read-only (distinct buffers).

struct Params {
    n: u32,
    s: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read>       inp: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.n) { return; }
    out[idx] = out[idx] + p.s * inp[idx];
}
