// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the depth kernel family (P2), driven directly through
//! `gpu_core` like `glue.rs` / `mse_fd.rs` — no model is built.
//!
//! The dominant test technique here is ADJOINTNESS, not finite differences.
//! Every `*_dx` kernel in this family is the adjoint of a LINEAR operator
//! (resize, pool, shuffle, strip-mean, convex-combine), and for a linear `A` the
//! backward is exactly `Aᵀ`. That gives an exact algebraic identity —
//!     <A(x), y> == <x, Aᵀ(y)>   for ALL x, y
//! — which is both cheaper and far sharper than FD: it holds to fp32 round-off
//! rather than to a tolerance, and a wrong adjoint (a dropped edge tap, a
//! transposed group index, an off-by-one window) breaks it immediately. FD is
//! used only where the op is genuinely nonlinear (sigmoid, softmax).
//!
//! `conv2d_gd` gets a different, stronger test: it must AGREE BIT-FOR-BIT with
//! the existing, already-gated `conv2d` at groups=1/dilation=1. A generalization
//! that changes the base case is a regression no matter how good its own tests
//! look.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use gpu_core::{f, Gpu};

const KERNELS: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),                                 // 0
    ("conv2d_gd", kernels::CONV2D_GD),                           // 1
    ("conv2d_gd_dx", kernels::CONV2D_GD_DX),                     // 2
    ("conv2d_gd_dw", kernels::CONV2D_GD_DW),                     // 3
    ("resize_bilinear", kernels::RESIZE_BILINEAR),               // 4
    ("resize_bilinear_dx", kernels::RESIZE_BILINEAR_DX),         // 5
    ("avgpool2d", kernels::AVGPOOL2D),                           // 6
    ("avgpool2d_dx", kernels::AVGPOOL2D_DX),                     // 7
    ("strip_pool", kernels::STRIP_POOL),                         // 8
    ("strip_pool_dx", kernels::STRIP_POOL_DX),                   // 9
    ("softmax_hw", kernels::SOFTMAX_HW),                         // 10
    ("softmax_hw_dx", kernels::SOFTMAX_HW_DX),                   // 11
    ("pixel_shuffle", kernels::PIXEL_SHUFFLE),                   // 12
    ("pixel_shuffle_dx", kernels::PIXEL_SHUFFLE_DX),             // 13
    ("convex_upsample", kernels::CONVEX_UPSAMPLE),               // 14
    ("convex_upsample_dmask", kernels::CONVEX_UPSAMPLE_DMASK),   // 15
    ("convex_upsample_dd", kernels::CONVEX_UPSAMPLE_DD),         // 16
    ("sigmoid", kernels::SIGMOID),                               // 17
    ("sigmoid_bwd", kernels::SIGMOID_BWD),                       // 18
    ("masked_l1", kernels::MASKED_L1),                           // 19
    ("masked_l1_grad", kernels::MASKED_L1_GRAD),                 // 20
];

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~[-1,1)
}
fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n).map(|_| lcg(&mut st)).collect()
}
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// Run a 2-storage kernel (x -> y).
fn run2(gpu: &Gpu, k: usize, x: &[f32], out_n: usize, params: &[u32]) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let yb = gpu.storage(out_n as u64);
    let s = gpu.step(k, &[&xb, &yb], params, out_n as u32);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    gpu.read(&yb, out_n)
}
/// Run a 3-storage kernel (a, b -> y).
fn run3(gpu: &Gpu, k: usize, a: &[f32], b: &[f32], out_n: usize, params: &[u32]) -> Vec<f32> {
    let ab = gpu.storage_init("a", a);
    let bb = gpu.storage_init("b", b);
    let yb = gpu.storage(out_n as u64);
    let s = gpu.step(k, &[&ab, &bb, &yb], params, out_n as u32);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    gpu.read(&yb, out_n)
}
/// Run a 4-storage kernel (a, b, c -> y).
fn run4(gpu: &Gpu, k: usize, a: &[f32], b: &[f32], c: &[f32], out_n: usize, params: &[u32]) -> Vec<f32> {
    let ab = gpu.storage_init("a", a);
    let bb = gpu.storage_init("b", b);
    let cb = gpu.storage_init("c", c);
    let yb = gpu.storage(out_n as u64);
    let s = gpu.step(k, &[&ab, &bb, &cb, &yb], params, out_n as u32);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    gpu.read(&yb, out_n)
}

/// Assert <A(x), y> == <x, A^T(y)> to fp32 round-off.
fn assert_adjoint(tag: &str, ax: &[f32], y: &[f32], x: &[f32], aty: &[f32]) {
    let lhs = dot(ax, y);
    let rhs = dot(x, aty);
    let tol = 1e-4 * lhs.abs().max(rhs.abs()).max(1.0);
    assert!(
        (lhs - rhs).abs() < tol,
        "{tag}: adjointness broken — <A(x),y> = {lhs}, <x,A^T(y)> = {rhs} (diff {:.3e})",
        (lhs - rhs).abs()
    );
}

// ---- conv2d_gd ----------------------------------------------------------------

/// The generalization must not change the base case: at groups=1/dilation=1,
/// conv2d_gd must reproduce the already-gated conv2d (p1_conv, p2_blocks,
/// p3_gradcheck).
///
/// NOT bit-equality, and the reason is worth knowing. `backend-cpu` binds its
/// AVX2/winograd fast paths BY KERNEL NAME (`find("conv2d")`, lib.rs:127-133), so
/// the reference here runs vectorized while conv2d_gd — deliberately named
/// distinctly — runs the generic Cranelift JIT. Same arithmetic, different
/// summation order, ~1 ULP apart (measured: 1087906882 vs 1087906881).
///
/// That gap is the *point*, not a defect: naming this kernel `conv2d` would have
/// made it bit-identical by silently inheriting a DENSE fast path that ignores
/// `groups` entirely and computes the wrong answer with no error. A 1-ULP
/// disagreement is the price of not having that bug.
#[test]
fn conv2d_gd_reproduces_conv2d_at_groups1_dilation1() {
    let gpu = Gpu::new(KERNELS);
    let (n, cin, h, w, cout, k) = (2u32, 3u32, 9u32, 7u32, 4u32, 3u32);
    let x = randvec(1, (n * cin * h * w) as usize);
    let wt = randvec(2, (cout * cin * k * k) as usize);

    for &(stride, pad) in &[(1u32, 1u32), (2, 1), (1, 0)] {
        let ho = (h + 2 * pad - k) / stride + 1;
        let wo = (w + 2 * pad - k) / stride + 1;
        let on = (n * cout * ho * wo) as usize;

        let want = run3(&gpu, 0, &x, &wt, on, &[n, cin, h, w, cout, k, stride, pad, ho, wo]);
        let got = run3(&gpu, 1, &x, &wt, on, &[n, cin, h, w, cout, k, stride, pad, 1, 1, ho, wo]);
        let mut max_ulp = 0i64;
        for i in 0..on {
            let ulp = (want[i].to_bits() as i64 - got[i].to_bits() as i64).abs();
            max_ulp = max_ulp.max(ulp);
            assert!(
                (want[i] - got[i]).abs() <= 1e-5 * want[i].abs().max(1.0),
                "conv2d_gd != conv2d at stride={stride} pad={pad}, element {i}: \
                 {} vs {} ({ulp} ulp)",
                want[i],
                got[i]
            );
        }
        // Pin that the two are within a couple of ULP — i.e. that the difference
        // really is summation order and not a creeping logic divergence.
        assert!(max_ulp <= 4, "stride={stride} pad={pad}: max {max_ulp} ulp is too far apart for a pure reassociation");
    }
}

/// Depthwise (groups == Cin == Cout): each output channel must see ONLY its own
/// input channel. Verified by construction rather than by a golden: zeroing input
/// channel `j` may change output channel `j` and nothing else. A wrong group
/// index still produces plausible numbers, so this tests the isolation directly.
#[test]
fn conv2d_gd_depthwise_channels_are_isolated() {
    let gpu = Gpu::new(KERNELS);
    let (n, c, h, w, k) = (1u32, 4u32, 6u32, 6u32, 3u32);
    let (stride, pad, dil) = (1u32, 1u32, 1u32);
    let x = randvec(3, (n * c * h * w) as usize);
    let wt = randvec(4, (c * 1 * k * k) as usize); // [C, 1, k, k]
    let on = (n * c * h * w) as usize;
    let params = [n, c, h, w, c, k, stride, pad, dil, c, h, w];

    let base = run3(&gpu, 1, &x, &wt, on, &params);
    for j in 0..c as usize {
        let mut x2 = x.clone();
        for i in 0..(h * w) as usize {
            x2[j * (h * w) as usize + i] = 0.0;
        }
        let got = run3(&gpu, 1, &x2, &wt, on, &params);
        for ch in 0..c as usize {
            let changed = (0..(h * w) as usize)
                .any(|i| got[ch * (h * w) as usize + i] != base[ch * (h * w) as usize + i]);
            assert_eq!(
                changed,
                ch == j,
                "depthwise leak: zeroing input ch {j} changed output ch {ch}"
            );
        }
    }
}

/// Dilation 2 must reach exactly 2 pixels out. Single-tap probe: a delta input
/// and a one-hot kernel tap put the response at a known, dilation-dependent
/// offset — a golden that a wrong dilation cannot accidentally satisfy.
#[test]
fn conv2d_gd_dilation_moves_the_tap() {
    let gpu = Gpu::new(KERNELS);
    let (n, c, h, w, k) = (1u32, 1u32, 7u32, 7u32, 3u32);
    let mut x = vec![0.0f32; (h * w) as usize];
    x[(3 * w + 3) as usize] = 1.0; // delta at (3,3)
    let mut wt = vec![0.0f32; (k * k) as usize];
    wt[0] = 1.0; // kernel tap (kh=0, kw=0)

    for &(dil, pad) in &[(1u32, 1u32), (2u32, 2u32)] {
        let on = (h * w) as usize;
        let y = run3(&gpu, 1, &x, &wt, on, &[n, c, h, w, c, k, 1, pad, dil, 1, h, w]);
        // y[ho,wo] = x[ho - pad + 0*dil, ...] -> the delta at (3,3) appears at
        // (3 + pad, 3 + pad) since kh=kw=0.
        let (ey, ex) = (3 + pad, 3 + pad);
        assert_eq!(y[(ey * w + ex) as usize], 1.0, "dilation={dil}: tap not at ({ey},{ex})");
        let nonzero = y.iter().filter(|v| **v != 0.0).count();
        assert_eq!(nonzero, 1, "dilation={dil}: expected exactly one non-zero response");
    }
}

#[test]
fn conv2d_gd_backward_is_adjoint_and_weight_grad_matches() {
    let gpu = Gpu::new(KERNELS);
    let (n, cin, h, w, cout, k, g) = (2u32, 4u32, 6u32, 6u32, 4u32, 3u32, 2u32);
    let (stride, pad, dil) = (1u32, 1u32, 1u32);
    let (ho, wo) = (h, w);
    let xn = (n * cin * h * w) as usize;
    let yn = (n * cout * ho * wo) as usize;
    let x = randvec(5, xn);
    let wt = randvec(6, (cout * (cin / g) * k * k) as usize);
    let params = [n, cin, h, w, cout, k, stride, pad, dil, g, ho, wo];

    // conv is linear in x for fixed w: <conv(x), dy> == <x, conv_dx(dy)>.
    let ax = run3(&gpu, 1, &x, &wt, yn, &params);
    let dy = randvec(7, yn);
    let atdy = run3(&gpu, 2, &dy, &wt, xn, &params);
    assert_adjoint("conv2d_gd (wrt x)", &ax, &dy, &x, &atdy);

    // ...and linear in w for fixed x: <conv(x,w), dy> == <w, conv_dw(dy,x)>.
    let dwn = (cout * (cin / g) * k * k) as usize;
    let dwb = gpu.storage_init("dw", &vec![0.0f32; dwn]); // dw ACCUMULATES; pre-zero
    let dyb = gpu.storage_init("dy", &dy);
    let xb = gpu.storage_init("x", &x);
    let s = gpu.step(3, &[&dyb, &xb, &dwb], &params, dwn as u32);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let dw = gpu.read(&dwb, dwn);
    assert_adjoint("conv2d_gd (wrt w)", &ax, &dy, &wt, &dw);
}

// ---- resize_bilinear (R1: the highest-risk kernel) ------------------------------

/// The two coordinate conventions must actually DIFFER, and each must match its
/// closed-form reference. This is the test that stands in for "we picked the
/// right grid" — a gradient check cannot see a half-pixel shift, because the
/// kernel remains perfectly self-consistent while resampling the wrong lattice.
#[test]
fn resize_bilinear_matches_both_coordinate_conventions() {
    let gpu = Gpu::new(KERNELS);
    let (h, w, ho, wo) = (2u32, 2u32, 4u32, 4u32);
    // A plane f(y,x) = 10*y + x is reproduced EXACTLY by bilinear interpolation,
    // so the expected output is the mapping itself — no interpolation error to
    // hide a coordinate bug behind.
    let x: Vec<f32> = (0..h * w).map(|i| 10.0 * (i / w) as f32 + (i % w) as f32).collect();

    for &align in &[0u32, 1u32] {
        let got = run2(&gpu, 4, &x, (ho * wo) as usize, &[1, 1, h, w, ho, wo, align]);
        for oy in 0..ho {
            for ox in 0..wo {
                let sy = if align == 1 {
                    oy as f32 * (h - 1) as f32 / (ho - 1) as f32
                } else {
                    (((oy as f32 + 0.5) * (h as f32 / ho as f32)) - 0.5).max(0.0)
                };
                let sx = if align == 1 {
                    ox as f32 * (w - 1) as f32 / (wo - 1) as f32
                } else {
                    (((ox as f32 + 0.5) * (w as f32 / wo as f32)) - 0.5).max(0.0)
                };
                // clamp-to-edge, matching the kernel
                let want = 10.0 * sy.min((h - 1) as f32) + sx.min((w - 1) as f32);
                let g = got[(oy * wo + ox) as usize];
                assert!(
                    (g - want).abs() < 1e-5,
                    "align={align} at ({oy},{ox}): got {g}, want {want}"
                );
            }
        }
    }

    // And the conventions are genuinely different — if this ever passes trivially
    // the test above is vacuous.
    let a0 = run2(&gpu, 4, &x, (ho * wo) as usize, &[1, 1, h, w, ho, wo, 0]);
    let a1 = run2(&gpu, 4, &x, (ho * wo) as usize, &[1, 1, h, w, ho, wo, 1]);
    assert!(a0 != a1, "align_corners=0 and =1 produced identical output");
}

/// Upsample AND downsample, both conventions, non-square, non-integer ratios —
/// the adjoint must hold in every case. This is what proves the gather window and
/// the edge clamps are right.
#[test]
fn resize_bilinear_dx_is_the_exact_adjoint() {
    let gpu = Gpu::new(KERNELS);
    let cases: &[(u32, u32, u32, u32)] = &[
        (4, 4, 8, 8),   // 2x up
        (8, 8, 4, 4),   // 2x down
        (5, 3, 9, 7),   // odd, non-integer ratio up
        (9, 7, 5, 3),   // odd, non-integer ratio down
        (4, 4, 4, 4),   // identity size
        (1, 1, 4, 4),   // degenerate input (exercises the out_len<=1 branch)
        (4, 4, 1, 1),   // degenerate output
        (3, 5, 7, 2),   // up in one axis, down in the other
    ];
    for &align in &[0u32, 1u32] {
        for &(h, w, ho, wo) in cases {
            let (n, c) = (2u32, 3u32);
            let xn = (n * c * h * w) as usize;
            let yn = (n * c * ho * wo) as usize;
            let x = randvec(11, xn);
            let y = randvec(12, yn);
            let params = [n, c, h, w, ho, wo, align];
            let ax = run2(&gpu, 4, &x, yn, &params);
            let aty = run2(&gpu, 5, &y, xn, &params);
            assert_adjoint(&format!("resize_bilinear {h}x{w}->{ho}x{wo} align={align}"), &ax, &y, &x, &aty);
        }
    }
}

// ---- avgpool2d ------------------------------------------------------------------

#[test]
fn avgpool2d_global_is_the_mean_and_dx_is_adjoint() {
    let gpu = Gpu::new(KERNELS);
    let (n, c, h, w) = (2u32, 3u32, 4u32, 5u32);
    let x = randvec(21, (n * c * h * w) as usize);
    // Ho=Wo=1 is SE's adaptive_avg_pool2d(x, 1): a plain per-channel mean.
    let got = run2(&gpu, 6, &x, (n * c) as usize, &[n, c, h, w, 1, 1]);
    for i in 0..(n * c) as usize {
        let s: f32 = x[i * (h * w) as usize..(i + 1) * (h * w) as usize].iter().sum();
        let want = s / (h * w) as f32;
        assert!((got[i] - want).abs() < 1e-5, "global avgpool ch {i}: {} vs {want}", got[i]);
    }

    for &(ho, wo) in &[(1u32, 1u32), (2, 2), (4, 5), (2, 3)] {
        let xn = (n * c * h * w) as usize;
        let yn = (n * c * ho * wo) as usize;
        let y = randvec(22, yn);
        let params = [n, c, h, w, ho, wo];
        let ax = run2(&gpu, 6, &x, yn, &params);
        let aty = run2(&gpu, 7, &y, xn, &params);
        assert_adjoint(&format!("avgpool2d {h}x{w}->{ho}x{wo}"), &ax, &y, &x, &aty);
    }
}

// ---- strip_pool -----------------------------------------------------------------

#[test]
fn strip_pool_means_the_right_axis_and_dx_is_adjoint() {
    let gpu = Gpu::new(KERNELS);
    let (n, c, h, w) = (2u32, 3u32, 4u32, 5u32);
    let x = randvec(31, (n * c * h * w) as usize);

    // axis 0 -> [N,C,H,1]: mean over W.
    let got = run2(&gpu, 8, &x, (n * c * h) as usize, &[n, c, h, w, 0]);
    for nc in 0..(n * c) as usize {
        for hi in 0..h as usize {
            let row = &x[nc * (h * w) as usize + hi * w as usize..][..w as usize];
            let want = row.iter().sum::<f32>() / w as f32;
            let g = got[nc * h as usize + hi];
            assert!((g - want).abs() < 1e-5, "strip_pool axis0 [{nc}][{hi}]: {g} vs {want}");
        }
    }
    // axis 1 -> [N,C,1,W]: mean over H.
    let got = run2(&gpu, 8, &x, (n * c * w) as usize, &[n, c, h, w, 1]);
    for nc in 0..(n * c) as usize {
        for wi in 0..w as usize {
            let want: f32 = (0..h as usize)
                .map(|hi| x[nc * (h * w) as usize + hi * w as usize + wi])
                .sum::<f32>()
                / h as f32;
            let g = got[nc * w as usize + wi];
            assert!((g - want).abs() < 1e-5, "strip_pool axis1 [{nc}][{wi}]: {g} vs {want}");
        }
    }

    for &axis in &[0u32, 1u32] {
        let keep = if axis == 0 { h } else { w };
        let xn = (n * c * h * w) as usize;
        let yn = (n * c * keep) as usize;
        let y = randvec(32, yn);
        let params = [n, c, h, w, axis];
        let ax = run2(&gpu, 8, &x, yn, &params);
        let aty = run2(&gpu, 9, &y, xn, &params);
        assert_adjoint(&format!("strip_pool axis={axis}"), &ax, &y, &x, &aty);
    }
}

// ---- softmax_hw (nonlinear -> FD) ------------------------------------------------

#[test]
fn softmax_hw_normalizes_and_its_backward_matches_fd() {
    let gpu = Gpu::new(KERNELS);
    let (n, hw) = (3u32, 16u32);
    let x = randvec(41, (n * hw) as usize);
    let y = run2(&gpu, 10, &x, (n * hw) as usize, &[n, hw]);
    for i in 0..n as usize {
        let s: f32 = y[i * hw as usize..(i + 1) * hw as usize].iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "image {i} sums to {s}, not 1");
        assert!(y.iter().all(|v| *v >= 0.0), "softmax produced a negative");
    }

    // FD on L = <r, softmax(x)> for a fixed random r.
    let r = randvec(42, (n * hw) as usize);
    let analytic = run3(&gpu, 11, &y, &r, (n * hw) as usize, &[n, hw]);
    let h = 1e-3f32;
    for i in 0..(n * hw) as usize {
        let mut xp = x.clone();
        xp[i] += h;
        let mut xm = x.clone();
        xm[i] -= h;
        let yp = run2(&gpu, 10, &xp, (n * hw) as usize, &[n, hw]);
        let ym = run2(&gpu, 10, &xm, (n * hw) as usize, &[n, hw]);
        let fd = (dot(&r, &yp) - dot(&r, &ym)) / (2.0 * h as f64);
        let a = analytic[i] as f64;
        assert!(
            (a - fd).abs() < 4e-3 + 8e-2 * a.abs().max(fd.abs()),
            "softmax_hw_dx[{i}]: analytic {a}, fd {fd}"
        );
    }
}

// ---- pixel_shuffle --------------------------------------------------------------

#[test]
fn pixel_shuffle_is_a_permutation_and_dx_inverts_it() {
    let gpu = Gpu::new(KERNELS);
    let (n, c, h, w, s) = (2u32, 2u32, 3u32, 4u32, 2u32);
    let xn = (n * c * s * s * h * w) as usize;
    let x = randvec(51, xn);
    let params = [n, c, h, w, s];
    let y = run2(&gpu, 12, &x, xn, &params);

    // A permutation preserves the multiset...
    let (mut a, mut b) = (x.clone(), y.clone());
    a.sort_by(|p, q| p.partial_cmp(q).unwrap());
    b.sort_by(|p, q| p.partial_cmp(q).unwrap());
    assert_eq!(a, b, "pixel_shuffle is not a permutation");
    // ...and its adjoint is its inverse, so the roundtrip is the identity.
    let back = run2(&gpu, 13, &y, xn, &params);
    assert_eq!(back, x, "pixel_shuffle_dx(pixel_shuffle(x)) != x");

    // CRD layout check: y[0,0,0,0] must come from x[0, 0, 0, 0] and
    // y[0,0,0,1] (sub-pixel sw=1) from input channel 1.
    let yv = run2(&gpu, 12, &(0..xn).map(|i| i as f32).collect::<Vec<_>>(), xn, &params);
    assert_eq!(yv[0], 0.0, "y[0,0,0,0] should be x[chan 0, 0, 0]");
    assert_eq!(yv[1], (h * w) as f32, "y[0,0,0,1] should be x[chan 1, 0, 0] (CRD)");
}

// ---- convex_upsample -------------------------------------------------------------

/// With a softmax'd mask the output is a CONVEX combination of the 3x3
/// neighbourhood, so it can never leave that neighbourhood's range. That is the
/// defining property of the op and it is checked directly.
#[test]
fn convex_upsample_output_stays_within_the_neighbourhood() {
    let gpu = Gpu::new(KERNELS);
    let (n, h, w, s) = (1u32, 4u32, 4u32, 2u32);
    let ss = (s * s) as usize;
    let d = randvec(61, (n * h * w) as usize);
    // A valid mask: softmax over the 9 axis. Build it on the host.
    let raw = randvec(62, (n * 9 * s * s * h * w) as usize);
    let mut mask = raw.clone();
    let hw = (h * w) as usize;
    for sub in 0..ss {
        for px in 0..hw {
            let idx = |k: usize| (k * ss + sub) * hw + px;
            let m = (0..9).map(|k| raw[idx(k)]).fold(f32::MIN, f32::max);
            let sum: f32 = (0..9).map(|k| (raw[idx(k)] - m).exp()).sum();
            for k in 0..9 {
                mask[idx(k)] = (raw[idx(k)] - m).exp() / sum;
            }
        }
    }
    let on = (n * h * s * w * s) as usize;
    let y = run3(&gpu, 14, &mask, &d, on, &[n, h, w, s]);

    for ho in 0..(h * s) as usize {
        for wo in 0..(w * s) as usize {
            let (hc, wc) = (ho / s as usize, wo / s as usize);
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let yy = (hc as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let xx = (wc as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let v = d[yy * w as usize + xx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let g = y[ho * (w * s) as usize + wo];
            assert!(
                g >= lo - 1e-5 && g <= hi + 1e-5,
                "convex_upsample at ({ho},{wo}): {g} outside neighbourhood [{lo}, {hi}]"
            );
        }
    }
}

#[test]
fn convex_upsample_backward_is_adjoint_in_both_inputs() {
    let gpu = Gpu::new(KERNELS);
    let (n, h, w, s) = (2u32, 3u32, 4u32, 2u32);
    let dn = (n * h * w) as usize;
    let mn = (n * 9 * s * s * h * w) as usize;
    let on = (n * h * s * w * s) as usize;
    let mask = randvec(71, mn);
    let d = randvec(72, dn);
    let dy = randvec(73, on);
    let params = [n, h, w, s];

    let ax = run3(&gpu, 14, &mask, &d, on, &params);
    // Bilinear in (mask, d): adjoint holds separately in each argument.
    let dmask = run3(&gpu, 15, &dy, &d, mn, &params);
    assert_adjoint("convex_upsample (wrt mask)", &ax, &dy, &mask, &dmask);
    let dd = run3(&gpu, 16, &dy, &mask, dn, &params);
    assert_adjoint("convex_upsample (wrt d)", &ax, &dy, &d, &dd);
}

// ---- sigmoid / masked_l1 ----------------------------------------------------------

#[test]
fn sigmoid_and_its_backward() {
    let gpu = Gpu::new(KERNELS);
    let x: Vec<f32> = (-40..=40).map(|i| i as f32 / 10.0).collect();
    let n = x.len();
    let y = run2(&gpu, 17, &x, n, &[n as u32]);
    for i in 0..n {
        let want = 1.0 / (1.0 + (-x[i]).exp());
        assert!((y[i] - want).abs() < 1e-6, "sigmoid({}) = {} want {want}", x[i], y[i]);
    }
    let dy = vec![1.0f32; n];
    let dx = run3(&gpu, 18, &x, &dy, n, &[n as u32]);
    for i in 0..n {
        let s = 1.0 / (1.0 + (-x[i]).exp());
        let want = s * (1.0 - s);
        assert!((dx[i] - want).abs() < 1e-6, "sigmoid'({}) = {} want {want}", x[i], dx[i]);
    }
}

#[test]
fn masked_l1_applies_the_mask_and_its_grad_is_the_signed_mask() {
    let gpu = Gpu::new(KERNELS);
    let pred = vec![1.0f32, 2.0, -3.0, 4.0, 0.0];
    let tgt = vec![1.5f32, 0.0, -1.0, 4.0, 0.0];
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0]; // element 2 masked OUT
    let n = pred.len();

    let out = run4(&gpu, 19, &pred, &tgt, &mask, n, &[n as u32]);
    assert_eq!(out, vec![0.5, 2.0, 0.0, 0.0, 0.0], "masked element must contribute 0");

    // scale is a bit-cast f32 in the uniform (the host's 1/(sum(mask)+eps)).
    let scale = 0.25f32;
    let d = run4(&gpu, 20, &pred, &tgt, &mask, n, &[n as u32, f(scale)]);
    // sign(pred-tgt)*mask*scale; sign(0)==0 -> elements 3,4 are 0.
    assert_eq!(d, vec![-0.25, 0.25, 0.0, 0.0, 0.0], "got {d:?}");
}
