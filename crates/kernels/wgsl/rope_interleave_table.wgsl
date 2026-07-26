// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Table-driven interleaved rotary position embedding (Z-Image / multi-axis).
//
// Unlike rope.wgsl (analytic angle, in place on the fused qkv buffer), the angle
// per (token, pair) comes from a HOST-precomputed cos/sin table — this is what
// lets a single kernel serve any multi-axis partition (Z-Image [32,48,48]). The
// same table row is applied to every head. Interleaved convention: within a head
// the channel pair (2j, 2j+1) is rotated (matches diffusers view_as_complex on
// reshape(..., -1, 2)). Out of place: x -> y (fresh SSA buffer).
//
// x/y layout: [seq, n_heads*head_dim] row-major; element (t,h,m) at
// (t*n_heads + h)*head_dim + m. cos/sin layout: [seq, head_dim/2] row-major;
// (t,j) at t*half + j. One invocation per (t, h, j).

struct Params {
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
    half: u32,        // head_dim / 2
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read>       cost: array<f32>;
@group(0) @binding(3) var<storage, read>       sint: array<f32>;
@group(0) @binding(4) var<storage, read_write> y:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.seq_len * p.n_heads * p.half;
    if (gidx >= total) { return; }

    let j = gidx % p.half;
    let tmp = gidx / p.half;
    let h = tmp % p.n_heads;
    let t = tmp / p.n_heads;

    let base = (t * p.n_heads + h) * p.head_dim + 2u * j;
    let ci = t * p.half + j;
    let c = cost[ci];
    let s = sint[ci];
    let e = x[base];
    let o = x[base + 1u];
    y[base]      = e * c - o * s;
    y[base + 1u] = e * s + o * c;
}
