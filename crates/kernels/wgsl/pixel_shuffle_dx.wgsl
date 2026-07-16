// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Pixel shuffle INPUT gradient, NCHW. A pure permutation, so its adjoint is the
// inverse permutation (space-to-depth) — no accumulation, no atomics.
//   dy : [N, C,      H*S, W*S]
//   dx : [N, C*S*S,  H,   W  ]   read_write (one invocation per INPUT element)
//
//   dx[n, (c*S + sh)*S + sw, h, w] = dy[n, c, h*S + sh, w*S + sw]
//
// Inverting the forward's own mapping: each input element is read by exactly one
// output, so every dx[idx] is written exactly once.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    S: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let cin_tot = p.C * p.S * p.S;
    let total = p.N * cin_tot * p.H * p.W;
    if (idx >= total) { return; }

    // Decode input coordinate (n, cin, h, w).
    let w   = idx % p.W;
    let t1  = idx / p.W;
    let h   = t1 % p.H;
    let t2  = t1 / p.H;
    let cin = t2 % cin_tot;
    let n   = t2 / cin_tot;

    // Invert cin = (c*S + sh)*S + sw.
    let sw = cin % p.S;
    let t3 = cin / p.S;
    let sh = t3 % p.S;
    let c  = t3 / p.S;

    let Ho = p.H * p.S;
    let Wo = p.W * p.S;
    let dy_idx = ((n * p.C + c) * Ho + (h * p.S + sh)) * Wo + (w * p.S + sw);
    dx[idx] = dy[dy_idx];
}
