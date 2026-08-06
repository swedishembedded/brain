// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GroupNorm affine gradients, STAGE 1 of 2 — partial sums for BOTH dgamma and
// dbeta in one pass over `dy`.
//
// `gn_dgamma` and `gn_dbeta` are each ONE invocation per channel, walking
// `N*H*W` elements serially: measured at 97.63 ms / 5.4 GB/s and 72.54 ms /
// 7.2 GB/s on a P40 — 1.5% and 2.1% of the ~346 GB/s roof, 24% of the backward
// between them. Two separate pathologies of the same shape, and each reads the
// whole of `dy` independently.
//
// This computes both from ONE traversal:
//   part[(c*P + t)*2 + 0] += sum over the slice of dy * xhat   (-> dgamma)
//   part[(c*P + t)*2 + 1] += sum over the slice of dy          (-> dbeta)
// with xhat = (x - mean_k) * rstd_k, k = n*G + c/(C/G).
//
// One invocation per (channel, partial) — `C*P` of them instead of `C` — with a
// stride-`P` walk so consecutive lanes read consecutive addresses.
//
// BARRIER-FREE, so `backend-cpu` JITs it and no capability branch is needed.
// The fold (`gn_dgb2`) is what accumulates into the caller's gradient buffer,
// preserving "a parameter gradient accumulates" — this stage only WRITES its
// own scratch.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    P: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       dy:    array<f32>;
@group(0) @binding(3) var<storage, read>       stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> part:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.C * p.P) { return; }

    let c = gidx / p.P;
    let t = gidx % p.P;
    let cpg = p.C / p.G;
    let g = c / cpg;
    let hw = p.H * p.W;
    let m = p.N * hw;            // elements in this channel

    var sg = 0.0;                // dgamma partial
    var sb = 0.0;                // dbeta partial
    for (var i = t; i < m; i = i + p.P) {
        let n = i / hw;
        let off = i % hw;
        let idx = (n * p.C + c) * hw + off;
        let k = n * p.G + g;
        let d = dy[idx];
        sg = sg + d * (x[idx] - stats[2u * k]) * stats[2u * k + 1u];
        sb = sb + d;
    }
    part[(c * p.P + t) * 2u + 0u] = sg;
    part[(c * p.P + t) * 2u + 1u] = sb;
}
