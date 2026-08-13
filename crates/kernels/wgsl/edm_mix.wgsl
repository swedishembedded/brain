// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  EDM output mix D = c_skip*x + c_out*F - spec
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// EDM output mix D = c_skip*x + c_out*F. Row-major [N, M], total = N*M;
// ab is packed [N,2]: ab[2n] = a[n] (c_skip), ab[2n+1] = b[n] (c_out).
//   n = i / m;  y[i] = ab[2n] * x[i] + ab[2n+1] * f[i]
// Property: a=1, b=0 gives y == x (exact f32 ==). Backward w.r.t. x/f uses
// scale_row with the host-kept unpacked a_vec/b_vec; no `dab` kernel — the
// EDM coefficients are sigma-derived constants, never trained.
// Exactly 4 storage buffers (the packing is what keeps it at the limit).
//

struct Params {
    total: u32,
    m: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:  array<f32>;
@group(0) @binding(2) var<storage, read>       f:  array<f32>;
@group(0) @binding(3) var<storage, read>       ab: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let n = i / p.m;
    y[i] = ab[2u * n] * x[i] + ab[2u * n + 1u] * f[i];
}
