// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GroupNorm statistics, one WORKGROUP per (n,g) group - the parallel, COALESCED twin of gn_stats.wgsl
// @how   256-thread workgroup tile, 3 barriers
// @opt   4
// @cpu   no
// @gpu   yes-wg256
// @npu   yes
// @quant none
// @dtype f32
//
// GroupNorm statistics, one WORKGROUP per (n,g) group — the parallel,
// COALESCED twin of gn_stats.wgsl. Same output layout
// (stats[2k]=mean, stats[2k+1]=rstd, eps inside the sqrt), so gn_apply /
// gn_dsum / gn_dx consume it unchanged.
//
// Why it exists: gn_stats dispatches N*G *invocations* — for a VAE decode that
// is 32 threads, each serially walking up to 1M elements, on a 3840-core card.
// Measured on a FLUX.2 VAE decode of a 64x64 latent to 512x512, gn_stats was
// **a third of the whole decode** across 30 dispatches.
// The two faults are the same ones `rmsnorm_rows` fixed for RMSNorm:
//   * no parallelism — 32 threads is 1/120th of the card, and
//   * no coalescing — each thread walks a contiguous run alone, so a warp's
//     32 lanes sit ~M floats apart and every 32-byte sector serves one float.
// Here the 256 threads of a workgroup stride the group's M elements together
// (thread t takes t, t+256, t+512, ...), so every fetch is fully used, and
// the 32 groups run concurrently.
//
// Numerics: the SAME two-pass formulation as gn_stats (mean first, then
// sum of squared deviations) — NOT the one-pass E[x^2]-mean^2 that gn_part
// uses — so there is no cancellation term to lose at 1M elements per group.
// Only the summation ORDER differs (a 256-way tree instead of one ascending
// run), which is strictly better conditioned. Measured on the FLUX.2 VAE:
// decode output cosine 1.000000 vs the gn_stats graph.
//
// GPU-only: two workgroup barriers (the CPU JIT's single-barrier form cannot
// express it). Select on `DeviceCaps::workgroup_reductions`; the fallback is
// gn_stats, which is exactly what this replaces.
//
// Dispatch: N*G workgroups of 256 = N*G*256 invocations.
// @workgroup_size(256) — 64 would give a group only 2 warps to hide the
// latency of a pure streaming read; 256 is 8 warps per SM at 32 groups, which
// is what turns the reduction from latency-bound into bandwidth-bound (and is
// the same reason the register-tiled GEMMs use 256).

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    G: u32,
    eps: u32,  // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let k = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (k >= p.N * p.G) { return; }
    let n = k / p.G;
    let g = k % p.G;
    let cpg = p.C / p.G;
    let m = cpg * p.H * p.W;               // elements in this group
    let base = (n * p.C + g * cpg) * p.H * p.W;
    let M = f32(m);

    // Pass 1: mean.
    var s = 0.0;
    for (var i = t; i < m; i = i + 256u) {
        s = s + x[base + i];
    }
    partial[t] = s;
    workgroupBarrier();
    var tot = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        tot = tot + partial[i];
    }
    let mean = tot / M;
    workgroupBarrier();

    // Pass 2: population variance about that mean.
    var v = 0.0;
    for (var i = t; i < m; i = i + 256u) {
        let d = x[base + i] - mean;
        v = v + d * d;
    }
    partial[t] = v;
    workgroupBarrier();
    var vtot = 0.0;
    for (var i = 0u; i < 256u; i = i + 1u) {
        vtot = vtot + partial[i];
    }
    if (t == 0u) {
        stats[2u * k] = mean;
        stats[2u * k + 1u] = inverseSqrt(vtot / M + bitcast<f32>(p.eps));
    }
}
