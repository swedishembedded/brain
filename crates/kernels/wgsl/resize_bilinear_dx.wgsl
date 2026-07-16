// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Bilinear resize INPUT gradient, NCHW.
//   dy : [N, C, Ho, Wo]
//   dx : [N, C, H,  W ]   read_write (one invocation per INPUT element)
//
// The adjoint of resize_bilinear.wgsl. Mathematically this is a SCATTER — each
// output pixel spreads its gradient over 4 input taps — which would need atomics.
// brain's kernels are atomic-free, so it is inverted into a GATHER: each input
// pixel finds the outputs that referenced it and sums their contributions. Every
// dx[idx] is then written by exactly one invocation, exactly as conv2d_dx.wgsl
// does for convolution.
//
// The candidate output range is bounded, not swept: only outputs whose source
// coordinate lands in (i-1, i+1) can touch input row i, so the search window is
// the inverse map of that interval, widened by 2 each side and clamped. Inside
// the window each candidate RECOMPUTES the forward's own mapping and takes the
// matching weight — so the adjoint holds by construction rather than by a second,
// independently-derived formula that could drift from the forward.
//
// Correctness at the edges falls out of that recomputation: the forward's
// high-side clamp (y1 = min(y0+1, H-1)) makes the last row take both taps, and
// the half_pixel clamp (src >= 0) piles extra weight on the first row. Both are
// reproduced because both branches are re-evaluated, not re-derived.
//
// Everything is inlined into main: the wgsl-cpu JIT rejects user-defined function
// calls, so a `src_coord` helper compiles on wgpu and hard-fails on CPU.

struct Params {
    N: u32,
    C: u32,
    H: u32,
    W: u32,
    Ho: u32,
    Wo: u32,
    align_corners: u32,
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
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }

    // Decode input coordinate (n, c, hi, wi).
    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    // ---- candidate output window, y axis: inverse map of (hi-1, hi+1) ----
    // inverse of src: align -> o = s*(out-1)/(in-1) ; half_pixel -> o = (s+0.5)*out/in - 0.5
    var oy_lo_f = 0.0;
    var oy_hi_f = 0.0;
    if (p.align_corners == 1u) {
        if (p.H > 1u) {
            let r = f32(p.Ho - 1u) / f32(p.H - 1u);
            oy_lo_f = (f32(hi) - 1.0) * r - 2.0;
            oy_hi_f = (f32(hi) + 1.0) * r + 2.0;
        } else {
            oy_hi_f = f32(p.Ho);
        }
    } else {
        let r = f32(p.Ho) / f32(p.H);
        oy_lo_f = (f32(hi) - 1.0 + 0.5) * r - 0.5 - 2.0;
        oy_hi_f = (f32(hi) + 1.0 + 0.5) * r - 0.5 + 2.0;
    }
    var ho_lo = 0u;
    if (oy_lo_f > 0.0) { ho_lo = u32(floor(oy_lo_f)); }
    var ho_hi = 0u;
    if (oy_hi_f > 0.0) { ho_hi = min(u32(ceil(oy_hi_f)), p.Ho - 1u); }

    // ---- candidate output window, x axis ----
    var ox_lo_f = 0.0;
    var ox_hi_f = 0.0;
    if (p.align_corners == 1u) {
        if (p.W > 1u) {
            let r = f32(p.Wo - 1u) / f32(p.W - 1u);
            ox_lo_f = (f32(wi) - 1.0) * r - 2.0;
            ox_hi_f = (f32(wi) + 1.0) * r + 2.0;
        } else {
            ox_hi_f = f32(p.Wo);
        }
    } else {
        let r = f32(p.Wo) / f32(p.W);
        ox_lo_f = (f32(wi) - 1.0 + 0.5) * r - 0.5 - 2.0;
        ox_hi_f = (f32(wi) + 1.0 + 0.5) * r - 0.5 + 2.0;
    }
    var wo_lo = 0u;
    if (ox_lo_f > 0.0) { wo_lo = u32(floor(ox_lo_f)); }
    var wo_hi = 0u;
    if (ox_hi_f > 0.0) { wo_hi = min(u32(ceil(ox_hi_f)), p.Wo - 1u); }

    var acc = 0.0;
    for (var ho: u32 = ho_lo; ho <= ho_hi; ho = ho + 1u) {
        // Recompute the forward's y mapping for this candidate.
        var sy = 0.0;
        if (p.align_corners == 1u) {
            if (p.Ho > 1u) { sy = f32(ho) * f32(p.H - 1u) / f32(p.Ho - 1u); }
        } else {
            sy = max((f32(ho) + 0.5) * (f32(p.H) / f32(p.Ho)) - 0.5, 0.0);
        }
        let fy0 = floor(sy);
        let y0 = u32(fy0);
        let y1 = min(y0 + 1u, p.H - 1u);
        let fy = sy - fy0;
        // Weight this output places on input row `hi` (the forward's high-side
        // clamp can collapse both taps onto the same row).
        var wy = 0.0;
        if (y0 == hi) { wy = wy + (1.0 - fy); }
        if (y1 == hi) { wy = wy + fy; }
        if (wy != 0.0) {
            for (var wo: u32 = wo_lo; wo <= wo_hi; wo = wo + 1u) {
                var sx = 0.0;
                if (p.align_corners == 1u) {
                    if (p.Wo > 1u) { sx = f32(wo) * f32(p.W - 1u) / f32(p.Wo - 1u); }
                } else {
                    sx = max((f32(wo) + 0.5) * (f32(p.W) / f32(p.Wo)) - 0.5, 0.0);
                }
                let fx0 = floor(sx);
                let x0 = u32(fx0);
                let x1 = min(x0 + 1u, p.W - 1u);
                let fx = sx - fx0;
                var wx = 0.0;
                if (x0 == wi) { wx = wx + (1.0 - fx); }
                if (x1 == wi) { wx = wx + fx; }
                if (wx != 0.0) {
                    let dy_idx = ((n * p.C + c) * p.Ho + ho) * p.Wo + wo;
                    acc = acc + dy[dy_idx] * wy * wx;
                }
            }
        }
    }
    dx[idx] = acc;
}
