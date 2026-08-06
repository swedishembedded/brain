// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bilinear grid sample GRID gradient — the other half of grid_sample.wgsl's backward, w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Bilinear grid sample GRID gradient — the other half of grid_sample.wgsl's
// backward, w.r.t. the sampling coordinates.
//   dy    : [N, C,  Ho, Wo]  idx = ((n*C + c)*Ho + ho)*Wo + wo
//   x     : [N, C,  H,  W]   idx = ((n*C + c)*H  + hi)*W  + wi
//   grid  : [N, Ho, Wo, 2]   idx = ((n*Ho + ho)*Wo + wo)*2 + a   (a = 0 -> X, 1 -> Y)
//   dgrid : [N, Ho, Wo, 2]   same indexing as grid                read_write
//
// One invocation per GRID POSITION (n, ho, wo) — it owns BOTH components of that
// position and writes dgrid[...,0] and dgrid[...,1], so the dispatch is
// threads = N*Ho*Wo — NOT N*Ho*Wo*2 (one per dgrid element) and NOT N*C*Ho*Wo
// (grid_sample.wgsl's count, which the shared Params makes look interchangeable).
// The two components are not split because they share the whole corner
// computation and the entire loop over C: splitting would double every load to
// halve nothing.
// Precedent for a single invocation writing two adjacent outputs is maxpool5.wgsl
// (y + argmax). The invocation -> element map is still a bijection, so the
// terminal write is a plain OVERWRITE and no pre-zeroing is required.
//
// Note dgrid is a per-sample tensor with its own batch axis, so unlike a WEIGHT
// gradient there is nothing to accumulate ACROSS positions — do not model this
// on conv2d_gd_dw.wgsl's accumulate-into-a-pre-zeroed-buffer contract.
//
// THE DERIVATIVE. With fx = ix - floor(ix), fy = iy - floor(iy) and the four
// weights of grid_sample.wgsl,
//     y = (1-fx)(1-fy)*v00 + fx(1-fy)*v01 + (1-fx)fy*v10 + fx*fy*v11
// and d(fx)/d(ix) = 1, so
//     dy/d(ix) = -(1-fy)*v00 + (1-fy)*v01 - fy*v10 + fy*v11
//     dy/d(iy) = -(1-fx)*v00 -     fx*v01 + (1-fx)*v10 + fx*v11
// Under padding_mode='zeros' an out-of-bounds tap contributes NOTHING to either
// sum — its v is not clamped to an edge value, its whole term is dropped, the
// same rule the forward applies. That asymmetric attenuation at the border is
// real signal, not an edge case to be smoothed away: it is what lets a warp learn
// to pull content back inside the frame.
//
// Then the chain rule through the unnormalization, which is where align_corners
// enters a SECOND time (getting the forward right and this factor wrong scales
// every coordinate gradient by W/(W-1) — small, plausible, and it quietly
// mistrains a similarity transform):
//     align_corners = 1:  d(ix)/d(gx) = (W - 1)/2 ,  d(iy)/d(gy) = (H - 1)/2
//     align_corners = 0:  d(ix)/d(gx) =  W / 2    ,  d(iy)/d(gy) =  H / 2
// (W == 1 with align_corners = 1 gives a factor of 0 — the coordinate genuinely
// has no effect there. PyTorch agrees.)
//
// Corners are SIGNED (i32) for the same reason as in the forward: a coordinate
// legitimately goes negative under 'zeros' padding, and u32 corner math turns
// -1 into 4294967295.
//
// Everything is inlined into main: the wgsl-cpu Cranelift JIT rejects
// user-defined function calls.

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
@group(0) @binding(2) var<storage, read>       x: array<f32>;
@group(0) @binding(3) var<storage, read>       grid: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgrid: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.Ho * p.Wo;
    if (idx >= total) { return; }

    // Decode grid position (n, ho, wo) from the linear index.
    let wo = idx % p.Wo;
    let t1 = idx / p.Wo;
    let ho = t1 % p.Ho;
    let n  = t1 / p.Ho;

    let g_idx = idx * 2u;   // == ((n*Ho + ho)*Wo + wo)*2
    let gx = grid[g_idx];
    let gy = grid[g_idx + 1u];

    // --- unnormalize (must match grid_sample.wgsl exactly; see header) ---
    var ix = 0.0;
    var iy = 0.0;
    var mx = 0.0;   // d(ix)/d(gx)
    var my = 0.0;   // d(iy)/d(gy)
    if (p.align_corners == 1u) {
        ix = ((gx + 1.0) * 0.5) * f32(p.W - 1u);
        iy = ((gy + 1.0) * 0.5) * f32(p.H - 1u);
        mx = 0.5 * f32(p.W - 1u);
        my = 0.5 * f32(p.H - 1u);
    } else {
        ix = ((gx + 1.0) * f32(p.W) - 1.0) * 0.5;
        iy = ((gy + 1.0) * f32(p.H) - 1.0) * 0.5;
        mx = 0.5 * f32(p.W);
        my = 0.5 * f32(p.H);
    }
    // Cast guard only — see grid_sample.wgsl. A clamped coordinate is far
    // outside [0,W)/[0,H), so all four taps drop and both gradients are 0,
    // which is the correct derivative of a saturated sample anyway.
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
    let in_x0 = (x0 >= 0 && x0 < Wi);
    let in_x1 = (x1 >= 0 && x1 < Wi);
    let in_y0 = (y0 >= 0 && y0 < Hi);
    let in_y1 = (y1 >= 0 && y1 < Hi);

    var gix = 0.0;
    var giy = 0.0;
    for (var c: u32 = 0u; c < p.C; c = c + 1u) {
        let row0 = (n * p.C + c) * p.H;   // first row of input plane (n, c)
        let go = dy[((n * p.C + c) * p.Ho + ho) * p.Wo + wo];
        // NW
        if (in_y0 && in_x0) {
            let v = x[(row0 + u32(y0)) * p.W + u32(x0)];
            gix = gix - v * (1.0 - fy) * go;
            giy = giy - v * (1.0 - fx) * go;
        }
        // NE
        if (in_y0 && in_x1) {
            let v = x[(row0 + u32(y0)) * p.W + u32(x1)];
            gix = gix + v * (1.0 - fy) * go;
            giy = giy - v * fx * go;
        }
        // SW
        if (in_y1 && in_x0) {
            let v = x[(row0 + u32(y1)) * p.W + u32(x0)];
            gix = gix - v * fy * go;
            giy = giy + v * (1.0 - fx) * go;
        }
        // SE
        if (in_y1 && in_x1) {
            let v = x[(row0 + u32(y1)) * p.W + u32(x1)];
            gix = gix + v * fy * go;
            giy = giy + v * fx * go;
        }
    }

    dgrid[g_idx]      = gix * mx;
    dgrid[g_idx + 1u] = giy * my;
}
