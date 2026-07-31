// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Generic KxK max-pool forward, NCHW, arbitrary STRIDE + symmetric zero-pad.
//   x      : [N, C, H,  W ]   idx = ((n*C + c)*H  + hi)*W  + wi
//   y      : [N, C, Ho, Wo]   idx = ((n*C + c)*Ho + ho)*Wo + wo
//   argmax : [N, C, Ho, Wo]   read_write, same indexing as y — the INPUT flat
//                             index of the winning tap, stored as f32.
//
// One invocation per OUTPUT element (n, c, ho, wo).
//
// The generalization of maxpool5.wgsl, which is this kernel pinned at stride=1
// (K and pad were already parameters there). The window's top-left input
// coordinate is
//   (ho*stride - pad, wo*stride - pad)
// and the caller-computed output size is
//   Ho = (H + 2*pad - K)/stride + 1   (likewise Wo from W)
// Ho/Wo are PASSED IN, never recomputed here: at stride > 1 the output extent no
// longer equals the input extent, so `total` and the y/argmax index math must use
// them. A caller that leaves Ho=H after switching to stride 2 gets a kernel that
// runs, has a plausible shape, and reads out of the window it meant to.
//
// NOTE the Params order: [N, C, H, W, K, stride, pad, Ho, Wo]. `stride` sits
// BEFORE `pad`, matching conv2d_gd/convtr1d's (K, stride, pad, dilation, groups)
// hyperparameter order — it is NOT a suffix-extension of maxpool5's
// [N, C, H, W, K, pad]. Migrating a maxpool5 call site means rewriting the whole
// word list, not appending to it.
//
// Out-of-bounds taps are treated as -inf and never selected; the running max is
// seeded from the FIRST in-bounds tap via `found`, so padding can never win (a
// 0.0 seed would beat an all-negative window). A window with NO in-bounds tap
// (only reachable at pad >= K, which torch itself rejects with `pad <= K/2`)
// keeps the seed and writes y = 0, argmax = 0 — a fabricated pointer at input 0.
// maxpool2d_dx never reads it: see the coverage argument in that kernel's
// header. `argmax` records the winner's
// input flat index for the gather-based backward in maxpool2d_dx.wgsl — exact in
// f32 while N*C*H*W < 2^24, which is the same bound maxpool5 has always carried.

struct Params {
    N:      u32,
    C:      u32,
    H:      u32,
    W:      u32,
    K:      u32,
    stride: u32,
    pad:    u32,
    Ho:     u32,
    Wo:     u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read_write> y:      array<f32>;
@group(0) @binding(3) var<storage, read_write> argmax: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decompose output flat index into (n, c, ho, wo).
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // Signed top-left input coordinate of the KxK window.
    let h0 = i32(ho) * i32(p.stride) - i32(p.pad);
    let w0 = i32(wo) * i32(p.stride) - i32(p.pad);

    var best: f32 = 0.0;
    var best_idx: u32 = 0u;
    var found: bool = false;

    for (var kh: u32 = 0u; kh < p.K; kh = kh + 1u) {
        let hi = h0 + i32(kh);
        if (hi >= 0 && hi < i32(p.H)) {
            for (var kw: u32 = 0u; kw < p.K; kw = kw + 1u) {
                let wi = w0 + i32(kw);
                if (wi >= 0 && wi < i32(p.W)) {
                    let ii = ((n * p.C + c) * p.H + u32(hi)) * p.W + u32(wi);
                    let v = x[ii];
                    if (!found || v > best) {
                        best = v;
                        best_idx = ii;
                        found = true;
                    }
                }
            }
        }
    }

    y[idx] = best;
    argmax[idx] = f32(best_idx);
}
