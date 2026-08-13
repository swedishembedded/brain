// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm backward per-group reductions, STAGE 1 of 2 - partial sums
// @how   one thread per partial, strided serial reduction (no barrier)
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// GroupNorm backward per-group reductions, STAGE 1 of 2 — partial sums.
//
// `gn_dsum` computes S1/S2 with ONE invocation per (n,g) group: 32 lanes for a
// 32-group norm, each walking `(C/G)*H*W` elements serially. Measured at
// 229 ms / 2.3 GB/s on a P40 — 0.7% of the ~346 GB/s roof, and 27% of a VQGAN
// training step's backward. This is the same pathology the FORWARD statistics
// had before `gn_part`/`gn_stats2`, and this pair is its adjoint.
//
// Stage 1: one invocation per (group, partial) — `N*G*P` of them — each summing
// a strided slice of its group. Stage 2 (`gn_dsum2`) folds the P partials.
//
// BARRIER-FREE by construction, so `backend-cpu` can JIT it. The cooperative
// alternative would need `workgroupBarrier` and a capability branch; this needs
// neither, and the forward's equivalent pair measured within 2x of cooperative
// on the GPU while being ~3x the serial kernel on the CPU.
//
// Determinism: each partial sums a FIXED strided subset in ascending index
// order, and stage 2 folds them in ascending partial order — so the result is
// reproducible run to run. It is NOT bit-identical to `gn_dsum`'s single
// ascending pass (a different association order is a different fp32 rounding),
// which is the same trade `gn_part` already makes for the forward.
//
//   part[(k*P + t)*2 + 0] = sum over this slice of dyg
//   part[(k*P + t)*2 + 1] = sum over this slice of dyg * xhat
// with xhat = (x - mean_k) * rstd_k.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    P: u32,     // partials per group
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       dyg:   array<f32>;
@group(0) @binding(3) var<storage, read>       stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> part:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.N * p.G * p.P;
    if (gidx >= total) { return; }

    let k = gidx / p.P;          // which (n,g) group
    let t = gidx % p.P;          // which partial within it
    let n = k / p.G;
    let g = k % p.G;
    let cpg = p.C / p.G;
    let m = cpg * p.H * p.W;     // elements in the group
    let base = (n * p.C + g * cpg) * p.H * p.W;

    let mean = stats[2u * k];
    let rstd = stats[2u * k + 1u];

    // Strided so consecutive lanes read consecutive addresses (coalesced),
    // which is the whole point of splitting the group.
    var s1 = 0.0;
    var s2 = 0.0;
    for (var i = t; i < m; i = i + p.P) {
        let d = dyg[base + i];
        s1 = s1 + d;
        s2 = s2 + d * (x[base + i] - mean) * rstd;
    }
    part[(k * p.P + t) * 2u + 0u] = s1;
    part[(k * p.P + t) * 2u + 1u] = s2;
}
