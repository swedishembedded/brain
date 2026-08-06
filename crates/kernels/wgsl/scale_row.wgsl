// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Per-row (per-sample) scalar scale on a row-major [N, M] tensor — spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Per-row (per-sample) scalar scale on a row-major [N, M] tensor — spec:
// docs/world-models/specs/P1.glue.md §3.2/§4.2. total = N*M, m = M.
//   y[i] = s[i / m] * x[i]
// EDM use: c_in/c_skip/c_out/lambda(sigma) row factors. Backward w.r.t. x is
// this same kernel (dx = scale_row(dy, s)); `ds` is deliberately NOT provided
// — EDM c_* are sigma-derived constants, never trained (spec §3.2).
//

struct Params {
    total: u32,
    m: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       s: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    y[i] = s[i / p.m] * x[i];
}
