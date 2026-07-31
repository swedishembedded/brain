// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Bilinear grid sample forward — `torch.nn.functional.grid_sample(mode='bilinear',
// padding_mode='zeros')`. Resamples x at arbitrary, DATA-DEPENDENT coordinates.
//   x    : [N, C,  H,  W]   idx = ((n*C + c)*H  + hi)*W  + wi
//   grid : [N, Ho, Wo, 2]   idx = ((n*Ho + ho)*Wo + wo)*2 + a   (a = 0 -> X, 1 -> Y)
//   y    : [N, C,  Ho, Wo]  idx = ((n*C + c)*Ho + ho)*Wo + wo
//
// One invocation per OUTPUT element. The grid is shared by every channel: the
// invocation for (n, c, ho, wo) reads grid[n, ho, wo, :] and samples plane (n, c).
//
// GRID CONVENTION (matching PyTorch exactly, and it is easy to get backwards):
// the LAST axis of `grid` is (X, Y) — component 0 indexes W, component 1 indexes
// H. Normalized to [-1, 1] over the *whole* input extent; -1 is the left/top
// edge, +1 the right/bottom edge.
//
// ALIGN_CORNERS — this is the silent half-pixel bug. Both modes look plausible,
// both are self-consistent, and a gradient check CANNOT tell them apart: the
// backward of the wrong mapping is the correct backward *of the wrong forward*.
// The only defence is numeric parity against the reference, so both modes live
// here and `align_corners` is an explicit Params field with no default.
//   align_corners = 0 (PyTorch's DEFAULT, what the face-alignment similarity
//                      warp and ROI-align use):
//       ix = ((gx + 1) * W - 1) / 2        ix = -0.5 at gx = -1
//       iy = ((gy + 1) * H - 1) / 2
//     -1 and +1 refer to the OUTER EDGES of the corner pixels, so a corner pixel
//     CENTRE sits at gx = -1 + 1/W. Half a pixel away from the other mode.
//   align_corners = 1:
//       ix = ((gx + 1) / 2) * (W - 1)      ix = 0 at gx = -1
//       iy = ((gy + 1) / 2) * (H - 1)
//     -1 and +1 refer to the CENTRES of the corner pixels. (W == 1 collapses to
//     ix = 0 for every gx, and its d(ix)/d(gx) is 0 — PyTorch does the same.)
//
// PADDING_MODE = 'zeros', which in PyTorch is NOT clamp-to-edge and NOT a
// clamped coordinate: the coordinate is left where it lands and each of the four
// corner taps is INDIVIDUALLY dropped when it falls outside [0,H) x [0,W). A tap
// at ix = -0.3 therefore keeps its east neighbour (x0 = -1 dropped, x1 = 0 kept
// with weight fx = 0.7) and the output is *attenuated* near the border rather
// than replicated. Substituting a clamp here — the natural instinct, and what
// resize_bilinear.wgsl legitimately does for its own fixed grid — changes both
// the values and the gradient at every border sample.
//
// Bilinear weights, with fx = ix - floor(ix), fy = iy - floor(iy):
//   NW (y0,x0): (1-fx)(1-fy)   NE (y0,x1): fx(1-fy)
//   SW (y1,x0): (1-fx)fy       SE (y1,x1): fx*fy
// x0 = floor(ix), x1 = x0+1, y0 = floor(iy), y1 = y0+1 — SIGNED, because a
// coordinate legitimately goes negative under 'zeros' padding. Doing this corner
// math in u32 wraps -1 to 4294967295, which passes an `< W` test only by luck
// and reads a wild address otherwise.
//
// Everything is inlined into main: the wgsl-cpu Cranelift JIT rejects
// user-defined function calls, so an `unnormalize` helper would compile on wgpu
// and hard-fail on the CPU backend. Same reason resize_bilinear.wgsl writes its
// mapping out once per axis.

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
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       grid: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decode output coordinate (n, c, ho, wo) from the linear index.
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let t2 = t1 / p.Ho;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let g_idx = ((n * p.Ho + ho) * p.Wo + wo) * 2u;
    let gx = grid[g_idx];
    let gy = grid[g_idx + 1u];

    // --- unnormalize to input pixel coordinates (inlined per axis; see header) ---
    var ix = 0.0;
    var iy = 0.0;
    if (p.align_corners == 1u) {
        ix = ((gx + 1.0) * 0.5) * f32(p.W - 1u);
        iy = ((gy + 1.0) * 0.5) * f32(p.H - 1u);
    } else {
        ix = ((gx + 1.0) * f32(p.W) - 1.0) * 0.5;
        iy = ((gy + 1.0) * f32(p.H) - 1.0) * 0.5;
    }
    // Guard the f32 -> i32 cast only. Anything outside [-1, W] contributes
    // nothing under 'zeros' padding, so clamping far beyond that window cannot
    // change a single result — it only keeps an absurd grid value (or an inf
    // arriving from an upstream bug) out of an out-of-range conversion, which is
    // implementation-defined in WGSL.
    ix = clamp(ix, -1.0e7, 1.0e7);
    iy = clamp(iy, -1.0e7, 1.0e7);

    let fx0 = floor(ix);
    let fy0 = floor(iy);
    let x0 = i32(fx0);
    let y0 = i32(fy0);
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = ix - fx0;
    let fy = iy - fy0;

    let Wi = i32(p.W);
    let Hi = i32(p.H);
    let row0 = (n * p.C + c) * p.H;   // first row of input plane (n, c)

    var acc = 0.0;
    // NW
    if (y0 >= 0 && y0 < Hi && x0 >= 0 && x0 < Wi) {
        acc = acc + x[(row0 + u32(y0)) * p.W + u32(x0)] * (1.0 - fx) * (1.0 - fy);
    }
    // NE
    if (y0 >= 0 && y0 < Hi && x1 >= 0 && x1 < Wi) {
        acc = acc + x[(row0 + u32(y0)) * p.W + u32(x1)] * fx * (1.0 - fy);
    }
    // SW
    if (y1 >= 0 && y1 < Hi && x0 >= 0 && x0 < Wi) {
        acc = acc + x[(row0 + u32(y1)) * p.W + u32(x0)] * (1.0 - fx) * fy;
    }
    // SE
    if (y1 >= 0 && y1 < Hi && x1 >= 0 && x1 < Wi) {
        acc = acc + x[(row0 + u32(y1)) * p.W + u32(x1)] * fx * fy;
    }
    y[idx] = acc;
}
