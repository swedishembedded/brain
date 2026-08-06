// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Bicubic resize INPUT gradient, NCHW
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Bicubic resize INPUT gradient, NCHW.
//   dy : [N, C, Ho, Wo]  idx = ((n*C + c)*Ho + ho)*Wo + wo
//   dx : [N, C, H,  W ]  idx = ((n*C + c)*H  + hi)*W  + wi   read_write
//
// One invocation per INPUT element. The adjoint of resize_bicubic.wgsl.
//
// Mathematically this is a SCATTER — each output pixel spreads its gradient over
// 16 input taps — which would need atomics. brain's kernels are atomic-free, so it
// is inverted into a GATHER: each input pixel finds the outputs that referenced it
// and sums their contributions. Every dx[idx] is then written by exactly one
// invocation, and the terminal write is a plain overwrite. Same strategy as
// resize_bilinear_dx.wgsl and conv2d_gd_dx.wgsl.
//
// --- the candidate window ----------------------------------------------------------
// The forward's taps are floor(src) + {-1, 0, +1, +2}, so an INTERIOR input row `hi`
// is touched only when floor(src_y) is in [hi-2, hi+1], i.e. src_y in [hi-2, hi+2).
// The window is therefore the inverse coordinate map of that interval, widened by 2
// each side for fp rounding and clamped — the bicubic stencil is twice as wide as
// bilinear's, which is the ONLY change to the window arithmetic (the bilinear
// version inverts (hi-1, hi+1)).
//
// The two BORDER rows are different and this is where a naive port breaks. The
// forward clamps its tap ACCESS to [0, H-1], so row 0 also receives every tap whose
// unclamped index is negative and row H-1 every tap at or past H-1 — from outputs
// arbitrarily far outside the interval above. Those rows are therefore forced to the
// full output extent (`ho_lo = 0` / `ho_hi = Ho-1`). Miss that and the adjoint fails
// only at the four image edges, only for some scale factors, and with a small
// residual — the shape of bug that survives a loose tolerance.
//
// Inside the window nothing is re-derived: each candidate RECOMPUTES the forward's
// own coordinate map, weights and clamps, and accumulates the weight it actually
// places on THIS input pixel (summing across taps, because the border clamp can
// collapse two or three taps onto the same row). So the adjoint holds by
// construction, and an over-wide window is harmless — a candidate that does not
// touch `hi` contributes exactly 0. The window only has to avoid MISSING outputs.
//
// Note the forward does NOT clamp the half_pixel source coordinate to >= 0 (see
// resize_bicubic.wgsl's header: ATen applies that clamp only for !cubic), so this
// kernel must not either. Reproducing the forward verbatim is what guarantees it.
//
// --- two wgsl-cpu JIT constraints, both paid for here ------------------------------
// 1. Everything is inlined into main: the JIT rejects user-defined function calls,
//    so a `src_coord`/`cubic_w` helper compiles on wgpu and hard-fails on CPU.
// 2. `let a = -0.75;` binds a BARE LITERAL. naga does not put literals in a
//    `Statement::Emit` range, so the JIT (which memoises every naga expression
//    handle the first time it evaluates it, wgsl-cpu/src/lib.rs `eval`) materialises
//    `a` in whichever Cranelift block first uses it — and then reuses that same
//    `Value` from sibling blocks. First drafting this kernel with the weight
//    polynomials written INSIDE the four `if` bodies put `a`'s definition in the
//    first `if` block and every later branch referenced a non-dominating value:
//      "define \"resize_bicubic_dx\": Verifier ... uses value v317 from
//       non-dominating inst347"
//    — a hard failure of `wgsl-cpu`'s `all_kernels_compile`, on the CPU backend
//    only. Hence the weights (`cy0..cy3`, `cx0..cx3`) are computed unconditionally
//    in straight-line code that dominates the branches, and only the SELECTION is
//    conditional. Rule for anyone editing this file: a `let` bound to a bare
//    literal must be first used in a block that dominates all its other uses.
//    (A `let` bound to an *expression* — `hm`, `wm` — is emitted at its declaration
//    and is safe.)

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

    // Degenerate output extent. `dy` is empty, so no input has a gradient — but
    // this guard is NOT cosmetic: the border overrides below assign `p.Ho - 1u`
    // / `p.Wo - 1u` UNCONDITIONALLY, which at Ho==0 underflows to 0xFFFFFFFF and
    // turns the gather into a ~4e9-iteration out-of-bounds sweep of `dy`. The
    // forward can't hit this (it dispatches zero threads when Ho*Wo == 0) and
    // resize_bilinear_dx can't either (it only ever `min`s against Ho-1, so the
    // underflow is absorbed), so the hazard is specific to this kernel's border
    // handling. On the CPU backend the loads are unchecked by design
    // (wgsl-cpu::Jit::new trusts the early-return mask), i.e. a hang or a
    // segfault rather than garbage.
    if (p.Ho == 0u || p.Wo == 0u) { dx[idx] = 0.0; return; }

    // Decode input coordinate (n, c, hi, wi).
    let wi = idx % p.W;
    let t1 = idx / p.W;
    let hi = t1 % p.H;
    let t2 = t1 / p.H;
    let c  = t2 % p.C;
    let n  = t2 / p.C;

    let a = -0.75;
    let hm = i32(p.H) - 1;
    let wm = i32(p.W) - 1;

    // ---- candidate output window, y axis: inverse map of (hi-2, hi+2) ----
    // inverse of src: align -> o = s*(out-1)/(in-1) ; half_pixel -> o = (s+0.5)*out/in - 0.5
    var oy_lo_f = 0.0;
    var oy_hi_f = 0.0;
    if (p.align_corners == 1u) {
        if (p.H > 1u) {
            let r = f32(p.Ho - 1u) / f32(p.H - 1u);
            oy_lo_f = (f32(hi) - 2.0) * r - 2.0;
            oy_hi_f = (f32(hi) + 2.0) * r + 2.0;
        } else {
            oy_hi_f = f32(p.Ho);
        }
    } else {
        let r = f32(p.Ho) / f32(p.H);
        oy_lo_f = (f32(hi) - 2.0 + 0.5) * r - 0.5 - 2.0;
        oy_hi_f = (f32(hi) + 2.0 + 0.5) * r - 0.5 + 2.0;
    }
    var ho_lo = 0u;
    if (oy_lo_f > 0.0) { ho_lo = u32(floor(oy_lo_f)); }
    var ho_hi = 0u;
    if (oy_hi_f > 0.0) { ho_hi = min(u32(ceil(oy_hi_f)), p.Ho - 1u); }
    // Border rows absorb every clamped tap; see header.
    if (hi == 0u) { ho_lo = 0u; }
    if (i32(hi) == hm) { ho_hi = p.Ho - 1u; }

    // ---- candidate output window, x axis ----
    var ox_lo_f = 0.0;
    var ox_hi_f = 0.0;
    if (p.align_corners == 1u) {
        if (p.W > 1u) {
            let r = f32(p.Wo - 1u) / f32(p.W - 1u);
            ox_lo_f = (f32(wi) - 2.0) * r - 2.0;
            ox_hi_f = (f32(wi) + 2.0) * r + 2.0;
        } else {
            ox_hi_f = f32(p.Wo);
        }
    } else {
        let r = f32(p.Wo) / f32(p.W);
        ox_lo_f = (f32(wi) - 2.0 + 0.5) * r - 0.5 - 2.0;
        ox_hi_f = (f32(wi) + 2.0 + 0.5) * r - 0.5 + 2.0;
    }
    var wo_lo = 0u;
    if (ox_lo_f > 0.0) { wo_lo = u32(floor(ox_lo_f)); }
    var wo_hi = 0u;
    if (ox_hi_f > 0.0) { wo_hi = min(u32(ceil(ox_hi_f)), p.Wo - 1u); }
    if (wi == 0u) { wo_lo = 0u; }
    if (i32(wi) == wm) { wo_hi = p.Wo - 1u; }

    var acc = 0.0;
    for (var ho: u32 = ho_lo; ho <= ho_hi; ho = ho + 1u) {
        // Recompute the forward's y mapping for this candidate.
        var sy = 0.0;
        if (p.align_corners == 1u) {
            if (p.Ho > 1u) { sy = f32(ho) * f32(p.H - 1u) / f32(p.Ho - 1u); }
        } else {
            sy = (f32(ho) + 0.5) * (f32(p.H) / f32(p.Ho)) - 0.5;
        }
        let byf = floor(sy);
        let by = i32(byf);
        let ty = sy - byf;
        let qy0 = ty + 1.0;
        let qy2 = 1.0 - ty;
        let qy3 = 2.0 - ty;
        // Weights computed UNCONDITIONALLY (see the `a` note in the header), then
        // selected. `wy` is the weight this output places on input row `hi`, summed
        // over the 4 taps: the forward's clamp-to-edge can collapse several taps
        // onto the same row, and every one of them contributes.
        let cy0 = ((a * qy0 - 5.0 * a) * qy0 + 8.0 * a) * qy0 - 4.0 * a;
        let cy1 = ((a + 2.0) * ty - (a + 3.0)) * ty * ty + 1.0;
        let cy2 = ((a + 2.0) * qy2 - (a + 3.0)) * qy2 * qy2 + 1.0;
        let cy3 = ((a * qy3 - 5.0 * a) * qy3 + 8.0 * a) * qy3 - 4.0 * a;
        var wy = 0.0;
        if (clamp(by - 1, 0, hm) == i32(hi)) { wy = wy + cy0; }
        if (clamp(by,     0, hm) == i32(hi)) { wy = wy + cy1; }
        if (clamp(by + 1, 0, hm) == i32(hi)) { wy = wy + cy2; }
        if (clamp(by + 2, 0, hm) == i32(hi)) { wy = wy + cy3; }
        if (wy != 0.0) {
            for (var wo: u32 = wo_lo; wo <= wo_hi; wo = wo + 1u) {
                var sx = 0.0;
                if (p.align_corners == 1u) {
                    if (p.Wo > 1u) { sx = f32(wo) * f32(p.W - 1u) / f32(p.Wo - 1u); }
                } else {
                    sx = (f32(wo) + 0.5) * (f32(p.W) / f32(p.Wo)) - 0.5;
                }
                let bxf = floor(sx);
                let bx = i32(bxf);
                let tx = sx - bxf;
                let qx0 = tx + 1.0;
                let qx2 = 1.0 - tx;
                let qx3 = 2.0 - tx;
                let cx0 = ((a * qx0 - 5.0 * a) * qx0 + 8.0 * a) * qx0 - 4.0 * a;
                let cx1 = ((a + 2.0) * tx - (a + 3.0)) * tx * tx + 1.0;
                let cx2 = ((a + 2.0) * qx2 - (a + 3.0)) * qx2 * qx2 + 1.0;
                let cx3 = ((a * qx3 - 5.0 * a) * qx3 + 8.0 * a) * qx3 - 4.0 * a;
                var wx = 0.0;
                if (clamp(bx - 1, 0, wm) == i32(wi)) { wx = wx + cx0; }
                if (clamp(bx,     0, wm) == i32(wi)) { wx = wx + cx1; }
                if (clamp(bx + 1, 0, wm) == i32(wi)) { wx = wx + cx2; }
                if (clamp(bx + 2, 0, wm) == i32(wi)) { wx = wx + cx3; }
                if (wx != 0.0) {
                    let dy_idx = ((n * p.C + c) * p.Ho + ho) * p.Wo + wo;
                    acc = acc + dy[dy_idx] * wy * wx;
                }
            }
        }
    }
    dx[idx] = acc;
}
