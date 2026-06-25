// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// 5x5 max-pool, stride 1, symmetric padding (SPPF). Output has the same H,W as
// input. One invocation per OUTPUT element (n,c,ho,wo). NCHW layout, activation
// flat index ((n*C+c)*H+h)*W+w. K and pad are parameterised (K=5, pad=2).
//
// The window's top-left input coordinate is (ho - pad, wo - pad). Out-of-bounds
// taps are treated as -inf and never selected; we initialise the running max
// from the first in-bounds tap so padding can never win. `argmax` stores the
// input flat index of the chosen max (as f32) for the gather-based backward.

struct Params {
    N:   u32,
    C:   u32,
    H:   u32,
    W:   u32,
    K:   u32,
    pad: u32,
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
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }

    // Decompose output flat index into (n,c,ho,wo).
    let wo = idx % p.W;
    let t1 = idx / p.W;
    let ho = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // Signed top-left input coordinate of the KxK window.
    let h0 = i32(ho) - i32(p.pad);
    let w0 = i32(wo) - i32(p.pad);

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
