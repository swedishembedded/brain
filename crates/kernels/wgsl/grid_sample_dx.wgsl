// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bilinear grid sample INPUT gradient — the adjoint of grid_sample.wgsl w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Bilinear grid sample INPUT gradient — the adjoint of grid_sample.wgsl w.r.t. x.
//   dy   : [N, C,  Ho, Wo]  idx = ((n*C + c)*Ho + ho)*Wo + wo
//   grid : [N, Ho, Wo, 2]   idx = ((n*Ho + ho)*Wo + wo)*2 + a   (a = 0 -> X, 1 -> Y)
//   dx   : [N, C,  H,  W]   idx = ((n*C + c)*H  + hi)*W  + wi   read_write
//
// One invocation per INPUT element. Terminal write is an OVERWRITE (`dx[idx] =
// acc`), so no pre-zeroing is required.
//
// WHY THIS LOOKS EXPENSIVE, AND WHY IT IS STILL THE RIGHT SHAPE.
// Mathematically the backward is a SCATTER: each output sample spreads its
// gradient over four input taps whose addresses come from the *data* in `grid`.
// That is the textbook atomicAdd, and brain has no atomics (AGENTS.md). The
// universal replacement is to key the invocation to the buffer being WRITTEN and
// gather — exactly what conv2d_gd_dx.wgsl and resize_bilinear_dx.wgsl do.
//
// The difference from those two is that their forward map is AFFINE, so the set
// of outputs that can touch input (hi,wi) is a closed-form bounded window. Here
// the map is an arbitrary run-time tensor: nothing outside `grid` itself says
// which outputs landed near (hi,wi), so the candidate window is the whole output
// plane and each dx element rescans grid[n] — O(Ho*Wo) per element, i.e.
// N*C*H*W*Ho*Wo tap tests overall. There is no cheaper EXACT and atomic-free
// inversion without building an index structure (a per-input-pixel bucket list),
// and building one needs either atomics or a sort — both worse here.
//
// What makes it tolerable rather than a coalescing disaster (docs/kernel-
// checklist.md §C.2): the scan order is (ho, wo) and the plane index (n, c) is
// LOOP-INVARIANT, so lanes that share a plane read the *same* grid address at the
// same iteration — a broadcast, one transaction per warp, not a strided gather.
// Consecutive gidx differ first in wi then hi, so a workgroup covers one plane
// except where it straddles a plane boundary, and it can straddle at most one
// (H*W >= 64) — the small-plane case degrades to a few extra transactions, never
// to a per-lane gather. `dy` is touched only on a MATCH, so the inner loop's
// whole steady-state memory traffic is those two broadcast grid floats. The cost
// is ALU-bound, not bandwidth-bound, which is the good failure mode.
//
// MEASURED, P40, release, mean of 8 dispatches in one submit (so the claim above
// is not just an argument). Throughput is flat across a 200x range of tap counts,
// which is the signature of an ALU-bound kernel — a coalescing fault would fall
// off as the shapes stop sharing a plane:
//   N1 C3   in 112x112 <- grid 112x112 :   8.4 ms   4.72e8 taps   56 Gtap/s
//   N1 C256 in  64x64  <- grid   7x7   :   0.8 ms   5.14e7 taps   61 Gtap/s
//   N1 C3   in 256x256 <- grid 112x112 :  41.4 ms   2.47e9 taps   60 Gtap/s
//   N1 C3   in 512x512 <- grid 112x112 : 119.5 ms   9.87e9 taps   83 Gtap/s
// So the 5-point face-alignment warp backward is single-digit ms at its working
// resolution, and only a full-frame 512^2 backward costs real time.
//
// The documented next step if a profile ever says this dominates: a companion
// kernel that reduces each TILExTILE block of the output grid to a 4-float
// bounding box of (ix, iy), letting this loop reject a whole tile with four
// compares. That is a pure acceleration structure — same result, extra buffer —
// so it belongs in a separate kernel + a selection seam, NOT as a second copy of
// this one. It is deliberately not written yet: for the 5-point similarity warp
// the whole grid is one smooth affine patch, and the numbers above say the
// inversion is not on anyone's critical path at that size.
//
// CORRECTNESS BY RECOMPUTATION. Inside the loop the forward's own mapping and
// corner arithmetic are re-evaluated verbatim (same align_corners branch, same
// signed corners, same clamp guard) and the weight this output places on THIS
// input pixel is read off. The adjoint therefore holds by construction rather
// than by a second, independently derived formula that could drift.
//   x0 == wi -> weight (1-fx);  x1 == wi -> weight fx;  likewise y0/y1 with fy.
// x1 = x0+1 so at most one of the two can match — the branches are exclusive,
// not additive (this is where resize_bilinear_dx.wgsl differs: its high-side
// clamp can collapse both taps onto one row, and 'zeros' padding never clamps).
// No bounds test is needed on the matched corner: (hi, wi) is an existing input
// element, so a corner equal to it is in bounds by definition, which is exactly
// the tap 'zeros' padding would have kept.
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
@group(0) @binding(2) var<storage, read>       grid: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.C * p.H * p.W;
    if (idx >= total) { return; }

    // Decode input coordinate (n, c, hi, wi) from the linear index.
    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let wi_i = i32(wi);
    let hi_i = i32(hi);

    var acc = 0.0;
    for (var ho: u32 = 0u; ho < p.Ho; ho = ho + 1u) {
        for (var wo: u32 = 0u; wo < p.Wo; wo = wo + 1u) {
            let g_idx = ((n * p.Ho + ho) * p.Wo + wo) * 2u;
            let gx = grid[g_idx];
            let gy = grid[g_idx + 1u];

            // Recompute the forward's mapping for this candidate output.
            var ix = 0.0;
            var iy = 0.0;
            if (p.align_corners == 1u) {
                ix = ((gx + 1.0) * 0.5) * f32(p.W - 1u);
                iy = ((gy + 1.0) * 0.5) * f32(p.H - 1u);
            } else {
                ix = ((gx + 1.0) * f32(p.W) - 1.0) * 0.5;
                iy = ((gy + 1.0) * f32(p.H) - 1.0) * 0.5;
            }
            ix = clamp(ix, -1.0e7, 1.0e7);
            iy = clamp(iy, -1.0e7, 1.0e7);

            let fx0 = floor(ix);
            let fy0 = floor(iy);
            let x0 = i32(fx0);
            let y0 = i32(fy0);

            // Cheap integer rejection first: this output touches column `wi`
            // only through its west (x0) or east (x0+1) tap.
            if (x0 == wi_i || (x0 + 1) == wi_i) {
                if (y0 == hi_i || (y0 + 1) == hi_i) {
                    let fx = ix - fx0;
                    let fy = iy - fy0;
                    var wx = fx;
                    if (x0 == wi_i) { wx = 1.0 - fx; }
                    var wy = fy;
                    if (y0 == hi_i) { wy = 1.0 - fy; }
                    let dy_idx = ((n * p.C + c) * p.Ho + ho) * p.Wo + wo;
                    acc = acc + dy[dy_idx] * wx * wy;
                }
            }
        }
    }
    dx[idx] = acc;
}
