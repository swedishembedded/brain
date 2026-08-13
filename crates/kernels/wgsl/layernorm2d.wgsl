// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Channels-first LayerNorm (ConvNeXt / SAM 2's `LayerNorm2d`), FUSED - the normalisation runs in NCHW, with no permute either side
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Channels-first LayerNorm (ConvNeXt / SAM 2's `LayerNorm2d`), FUSED — the
// normalisation runs in NCHW, with no permute either side.
//
//   x, y : [N, C, H, W]
//   gamma, beta : [C]
//   y[n,c,h,w] = ((x - mean_{n,hw}) * rstd_{n,hw}) * gamma[c] + beta[c]
// where the statistics are over the CHANNEL axis at each spatial position.
//
// Why this exists. The composed form is `nchw_nlc` -> `layernorm_rows` ->
// `nlc_nchw`, and it shipped because the middle stage is coalesced. But
// a measured pass found the two permutes at **67-86% of the
// whole thing**, at 14-33% of the bandwidth roof — and it worsens with `H*W`
// (47.5 GB/s at 65536 against 102.8 at 1024). The reason to reject fusing was
// the strided channel access, and the composition pays that sector
// amplification TWICE (once per permute) to avoid paying it once.
//
// So this kernel pays it once: one invocation per spatial position, walking the
// `C` values at stride `H*W`. It reads x twice (mean, then variance+apply)
// against the composition's read-write-read-write-read-write, so even at equal
// per-access efficiency it moves less.
//
// One invocation per (n, hw). Barrier-free, so `backend-cpu` JITs it.

struct Params {
    N: u32,
    C: u32,
    HW: u32,
    eps: u32,   // bitcast<f32>
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       beta:  array<f32>;
@group(0) @binding(4) var<storage, read_write> y:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    if (gidx >= p.N * p.HW) { return; }

    let n = gidx / p.HW;
    let hw = gidx % p.HW;
    let base = n * p.C * p.HW + hw;      // element (n, 0, hw); channel stride HW
    let cf = f32(p.C);

    // Pass 1: mean over the channel axis.
    var s = 0.0;
    for (var c: u32 = 0u; c < p.C; c = c + 1u) {
        s = s + x[base + c * p.HW];
    }
    let mean = s / cf;

    // Pass 2: variance. Ascending c, so the fold order is fixed.
    var v = 0.0;
    for (var c: u32 = 0u; c < p.C; c = c + 1u) {
        let d = x[base + c * p.HW] - mean;
        v = v + d * d;
    }
    let rstd = inverseSqrt(v / cf + bitcast<f32>(p.eps));

    // Pass 3: normalise + affine.
    for (var c: u32 = 0u; c < p.C; c = c + 1u) {
        let i = base + c * p.HW;
        y[i] = (x[i] - mean) * rstd * gamma[c] + beta[c];
    }
}
