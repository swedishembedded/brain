// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the `resize_bicubic` family, driven directly through
//! `gpu_core` like `depth_kernels.rs` / `glue.rs` — no model is built.
//!
//! Three independent techniques, because each catches a different class of bug:
//!
//! 1. **Parity against an f64 CPU oracle** (`bicubic_ref` below). The oracle is
//!    re-derived from ATen's structure rather than sharing code with the kernel —
//!    an oracle that shares code with the thing it checks proves nothing. This is
//!    the ONLY test that can catch a wrong coordinate convention: a kernel that
//!    resamples the wrong grid stays perfectly self-consistent, so neither an
//!    adjoint identity nor a gradient check will ever notice.
//!
//! 2. **Adjointness for `_dx`.** Bicubic resize is a LINEAR operator `A`, so its
//!    backward is exactly `Aᵀ` and `<A(x), y> == <x, Aᵀ(y)>` holds for ALL `x, y`
//!    to fp32 round-off. That is sharper and cheaper than finite differences and
//!    breaks immediately on a dropped edge tap or an off-by-one gather window —
//!    which is the failure mode this family is most exposed to, because the
//!    4-wide stencil's border clamp piles several taps onto rows 0 and H-1.
//!
//! 3. **Finite differences for `_dx`.** Redundant with (2) for a linear op, and
//!    kept deliberately: (2) only proves the backward is the adjoint of *some*
//!    operator consistent with the forward as measured by a dot product, while FD
//!    of the actual scalar loss `L = <y, dy>` through the actual forward kernel
//!    ties the gradient to the forward that ships.
//!
//! Plus two goldens that pin the polynomial itself:
//!   * `a = -0.75` is checked against a hand-computed weight (-0.09375 at t=0.5),
//!     which is what separates PyTorch's -0.75 from TensorFlow's/OpenCV's other
//!     conventions — a wrong `a` still sums to 1, still looks smooth, and still
//!     passes every adjointness and FD check in this file.
//!   * the half_pixel source coordinate is NOT clamped to >= 0 for cubic (ATen's
//!     `area_pixel_compute_source_index` applies that clamp only when `!cubic`),
//!     unlike `resize_bilinear`. Asserted by showing the kernel disagrees with the
//!     clamped variant of the same oracle.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use data::rng::Lcg;
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("resize_bicubic", kernels::RESIZE_BICUBIC),       // 0
    ("resize_bicubic_dx", kernels::RESIZE_BICUBIC_DX), // 1
];
const K_FWD: usize = 0;
const K_DX: usize = 1;

/// Expensive/physical-GPU work is skipped by presence of the variable, never by
/// its value — the repo-wide idiom.
fn skip() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        brain_testutil::skip_unavailable("resize_bicubic kernels (MOE_SKIP_GPU_TESTS)");
        return true;
    }
    false
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

// ---- CPU oracle ------------------------------------------------------------------
// Independently re-derived in f64 from ATen's `UpSample.h`; matches
// `wgsl/resize_bicubic.wgsl`. Deliberately the slow, obvious form.

/// ATen `area_pixel_compute_source_index`. `clamp_neg` is the `!cubic` branch —
/// bicubic passes `false`; it exists only so a test can show the two differ.
fn src_coord(o: u32, out_n: u32, in_n: u32, align: u32, clamp_neg: bool) -> f64 {
    if align == 1 {
        if out_n > 1 {
            o as f64 * (in_n as f64 - 1.0) / (out_n as f64 - 1.0)
        } else {
            0.0
        }
    } else {
        let s = (o as f64 + 0.5) * (in_n as f64 / out_n as f64) - 0.5;
        if clamp_neg && s < 0.0 {
            0.0
        } else {
            s
        }
    }
}

/// ATen `get_cubic_upsample_coefficients` with `A = -0.75`, for taps
/// `floor(src) + {-1, 0, +1, +2}`.
fn cubic4(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    // cubic_convolution1: |s| <= 1
    let c1 = |s: f64| ((A + 2.0) * s - (A + 3.0)) * s * s + 1.0;
    // cubic_convolution2: 1 < |s| < 2
    let c2 = |s: f64| ((A * s - 5.0 * A) * s + 8.0 * A) * s - 4.0 * A;
    [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)]
}

/// ATen `upsample_get_value_bounded`: clamp the ACCESS (replicate border), never
/// drop the tap and never renormalise the weights.
fn at(x: &[f32], base: usize, h: u32, w: u32, iy: i64, ix: i64) -> f64 {
    let cy = iy.clamp(0, h as i64 - 1) as usize;
    let cx = ix.clamp(0, w as i64 - 1) as usize;
    x[base + cy * w as usize + cx] as f64
}

#[allow(clippy::too_many_arguments)]
fn bicubic_ref(
    x: &[f32],
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    ho: u32,
    wo: u32,
    align: u32,
    clamp_neg: bool,
) -> Vec<f32> {
    let mut y = vec![0f32; (n * c * ho * wo) as usize];
    for nc in 0..(n * c) as usize {
        let xbase = nc * (h * w) as usize;
        let ybase = nc * (ho * wo) as usize;
        for oy in 0..ho {
            let sy = src_coord(oy, ho, h, align, clamp_neg);
            let by = sy.floor();
            let wy = cubic4(sy - by);
            let by = by as i64;
            for ox in 0..wo {
                let sx = src_coord(ox, wo, w, align, clamp_neg);
                let bx = sx.floor();
                let wx = cubic4(sx - bx);
                let bx = bx as i64;
                let mut acc = 0.0f64;
                for (ky, &wyk) in wy.iter().enumerate() {
                    let mut row = 0.0f64;
                    for (kx, &wxk) in wx.iter().enumerate() {
                        row += at(x, xbase, h, w, by - 1 + ky as i64, bx - 1 + kx as i64) * wxk;
                    }
                    acc += row * wyk;
                }
                y[ybase + (oy * wo + ox) as usize] = acc as f32;
            }
        }
    }
    y
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

/// Shapes exercised everywhere: up, down, odd, non-integer ratios, degenerate
/// extents, and mixed direction per axis. `(1,1,..)` and `(..,1,1)` are the ones
/// that hit the `out<=1` / `in<=1` branches in both the forward and the window.
const CASES: &[(u32, u32, u32, u32)] = &[
    (4, 4, 8, 8),   // 2x up
    (8, 8, 4, 4),   // 2x down
    (5, 3, 9, 7),   // odd, non-integer ratio up
    (9, 7, 5, 3),   // odd, non-integer ratio down
    (4, 4, 4, 4),   // identity size
    (1, 1, 4, 4),   // degenerate input
    (4, 4, 1, 1),   // degenerate output
    (3, 5, 7, 2),   // up in one axis, down in the other
    (2, 2, 7, 7),   // stencil wider than the image (every tap clamps)
    (16, 16, 5, 5), // >3x downsample
];

// ---- forward --------------------------------------------------------------------

/// The only test that can catch a wrong coordinate convention or a wrong border
/// rule. Everything else in this file is self-consistent with whatever grid the
/// kernel happens to resample.
#[test]
fn resize_bicubic_matches_the_cpu_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c) = (2u32, 3u32);
    for &align in &[0u32, 1u32] {
        for &(h, w, ho, wo) in CASES {
            let x = Lcg::new(101).vec((n * c * h * w) as usize);
            let yn = (n * c * ho * wo) as usize;
            let got = run2(&gpu, K_FWD, &x, yn, &[n, c, h, w, ho, wo, align]);
            let want = bicubic_ref(&x, n, c, h, w, ho, wo, align, false);
            // 3e-5: the oracle accumulates in f64 and the kernel in f32, and the
            // cubic stencil's negative lobes cancel, so the gap is larger than
            // bilinear's. Measured worst relative error over these shapes is
            // 1.3e-6 — 20x of headroom, while every bug this test exists to catch
            // (wrong `a`, wrong coordinate convention, wrong border rule) moves
            // the result by 1e-2 or more.
            for i in 0..yn {
                assert!(
                    (got[i] - want[i]).abs() <= 3e-5 * want[i].abs().max(1.0),
                    "align={align} {h}x{w}->{ho}x{wo} elem {i}: got {}, want {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}

/// The four weights sum to 1 for every `t`, so a constant image must come back
/// exactly constant — including at the borders, where the replicate clamp folds
/// several taps onto one row. A renormalising or tap-dropping border rule fails
/// here even though it fails nowhere else in this file.
#[test]
fn resize_bicubic_reproduces_a_constant_including_at_the_borders() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (h, w) = (6u32, 5u32);
    let x = vec![2.5f32; (h * w) as usize];
    for &align in &[0u32, 1u32] {
        for &(ho, wo) in &[(13u32, 11u32), (3, 2), (6, 5)] {
            let got = run2(&gpu, K_FWD, &x, (ho * wo) as usize, &[1, 1, h, w, ho, wo, align]);
            for (i, &g) in got.iter().enumerate() {
                assert!(
                    (g - 2.5).abs() < 1e-5,
                    "align={align} ->{ho}x{wo} elem {i}: {g} != 2.5"
                );
            }
        }
    }
}

/// The cubic kernel is INTERPOLATING: at `t = 0` the weights collapse to
/// `[0, 1, 0, 0]`. With `align_corners = 1` and `Ho == H`, every source coordinate
/// is an exact integer, so the resize must be a bit-exact identity. A fractional
/// offset anywhere in the coordinate math (a stray +0.5, a half-pixel convention
/// leaking into the align_corners branch) destroys this immediately.
#[test]
fn resize_bicubic_align_corners_same_size_is_a_bit_exact_identity() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w) = (2u32, 3u32, 6u32, 5u32);
    let x = Lcg::new(7).vec((n * c * h * w) as usize);
    let got = run2(&gpu, K_FWD, &x, x.len(), &[n, c, h, w, h, w, 1]);
    assert_eq!(got, x, "align_corners=1 identity resize is not the identity");
}

/// Pins `a = -0.75` numerically, and pins that the stencil overshoots.
///
/// `H = W = 5`, `align_corners = 1`, `Ho = Wo = 9` puts every odd output on a
/// source coordinate of `k + 0.5`, where the weights are exactly
/// `[-0.09375, 0.59375, 0.59375, -0.09375]` (hand-computed from Keys' kernel at
/// `a = -0.75`). With a unit delta at input `(2,2)`:
///   * output `(3,3)` reads it through the `+0.59375` lobe on both axes;
///   * output `(1,3)` reads it through the `-0.09375` lobe on y — a NEGATIVE
///     result from a non-negative input, which no bilinear kernel can produce.
///
/// This is the golden a wrong `a` cannot satisfy: at `a = -0.5` (TensorFlow) the
/// outer lobe is -0.0625 and the inner 0.5625, at `a = -1.0` they are -0.125 and
/// 0.625. All of them still sum to 1 and still pass every other test here.
#[test]
fn resize_bicubic_uses_a_equals_minus_three_quarters() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (h, w, ho, wo) = (5u32, 5u32, 9u32, 9u32);
    let mut x = vec![0f32; (h * w) as usize];
    x[(2 * w + 2) as usize] = 1.0;
    let got = run2(&gpu, K_FWD, &x, (ho * wo) as usize, &[1, 1, h, w, ho, wo, 1]);

    const INNER: f32 = 0.59375; // cubic_convolution1(0.5) at a = -0.75
    const OUTER: f32 = -0.09375; // cubic_convolution2(1.5) at a = -0.75

    let center = got[(3 * wo + 3) as usize];
    assert!(
        (center - INNER * INNER).abs() < 1e-6,
        "inner lobe: got {center}, want {}",
        INNER * INNER
    );
    let overshoot = got[(wo + 3) as usize]; // output (1, 3)
    assert!(
        (overshoot - OUTER * INNER).abs() < 1e-6,
        "outer lobe: got {overshoot}, want {} (a != -0.75?)",
        OUTER * INNER
    );
    assert!(
        overshoot < 0.0,
        "a non-negative input produced no overshoot — this is not a cubic stencil"
    );
}

/// `resize_bilinear` clamps the half_pixel source coordinate to `>= 0`; bicubic
/// must NOT (ATen's `area_pixel_compute_source_index` gates that clamp on
/// `!cubic`). The two differ only in the first row/column of an upsample, by less
/// than a pixel — invisible to every gradient check — so it is asserted directly
/// by running the oracle both ways.
#[test]
fn resize_bicubic_half_pixel_source_coordinate_is_not_clamped_at_zero() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (h, w, ho, wo) = (4u32, 4u32, 8u32, 8u32);
    let x = Lcg::new(31).vec((h * w) as usize);
    let params = [1u32, 1, h, w, ho, wo, 0];
    let got = run2(&gpu, K_FWD, &x, (ho * wo) as usize, &params);

    let unclamped = bicubic_ref(&x, 1, 1, h, w, ho, wo, 0, false);
    let clamped = bicubic_ref(&x, 1, 1, h, w, ho, wo, 0, true);
    // The two conventions must genuinely differ, or this test is vacuous.
    let differs = unclamped
        .iter()
        .zip(&clamped)
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(differs, "oracle variants agree — the case does not exercise a negative source coordinate");

    for i in 0..got.len() {
        assert!(
            (got[i] - unclamped[i]).abs() <= 1e-5 * unclamped[i].abs().max(1.0),
            "elem {i}: kernel {} matches neither; unclamped {} clamped {}",
            got[i],
            unclamped[i],
            clamped[i]
        );
    }
}

/// The two conventions are genuinely different functions; if this ever passes
/// trivially the parity test above is checking one branch twice.
#[test]
fn resize_bicubic_align_corners_changes_the_result() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (h, w, ho, wo) = (4u32, 4u32, 7u32, 7u32);
    let x = Lcg::new(41).vec((h * w) as usize);
    let a0 = run2(&gpu, K_FWD, &x, (ho * wo) as usize, &[1, 1, h, w, ho, wo, 0]);
    let a1 = run2(&gpu, K_FWD, &x, (ho * wo) as usize, &[1, 1, h, w, ho, wo, 1]);
    assert!(a0 != a1, "align_corners=0 and =1 produced identical output");
}

// ---- backward --------------------------------------------------------------------

/// Upsample AND downsample, both conventions, non-square, non-integer ratios — the
/// adjoint must hold in every case. This is what proves the gather window and, in
/// particular, the border override (rows 0 and H-1 absorb every clamped tap from
/// arbitrarily distant outputs) are right.
#[test]
fn resize_bicubic_dx_is_the_exact_adjoint() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for &align in &[0u32, 1u32] {
        for &(h, w, ho, wo) in CASES {
            let (n, c) = (2u32, 3u32);
            let xn = (n * c * h * w) as usize;
            let yn = (n * c * ho * wo) as usize;
            let x = Lcg::new(11).vec(xn);
            let y = Lcg::new(12).vec(yn);
            let params = [n, c, h, w, ho, wo, align];
            let ax = run2(&gpu, K_FWD, &x, yn, &params);
            let aty = run2(&gpu, K_DX, &y, xn, &params);
            assert_adjoint(
                &format!("resize_bicubic {h}x{w}->{ho}x{wo} align={align}"),
                &ax,
                &y,
                &x,
                &aty,
            );
        }
    }
}

/// Adjointness is an identity between two dot products; this ties `_dx` to the
/// forward kernel that actually ships. `L = <y, g>` for a fixed random `g`, so
/// `dL/dx == A^T(g) == dx`, checked by central differences THROUGH the forward
/// kernel. Small shapes and `step_by` sampling keep the dispatch count sane.
#[test]
fn resize_bicubic_dx_matches_finite_differences() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for &align in &[0u32, 1u32] {
        for &(h, w, ho, wo) in &[(5u32, 4u32, 7u32, 6u32), (7, 6, 4, 3)] {
            let (n, c) = (1u32, 2u32);
            let xn = (n * c * h * w) as usize;
            let yn = (n * c * ho * wo) as usize;
            let x = Lcg::new(51).vec(xn);
            let g = Lcg::new(52).vec(yn);
            let params = [n, c, h, w, ho, wo, align];

            let ana = run2(&gpu, K_DX, &g, xn, &params);

            let eps = 1e-2f32;
            for i in (0..xn).step_by(3) {
                let mut xp = x.clone();
                xp[i] += eps;
                let mut xm = x.clone();
                xm[i] -= eps;
                let lp = dot(&run2(&gpu, K_FWD, &xp, yn, &params), &g);
                let lm = dot(&run2(&gpu, K_FWD, &xm, yn, &params), &g);
                let num = (lp - lm) / (2.0 * eps as f64);
                let a = ana[i] as f64;
                let tol = 2e-3 + 1e-2 * num.abs().max(a.abs());
                assert!(
                    (num - a).abs() < tol,
                    "align={align} {h}x{w}->{ho}x{wo} dx[{i}]: numeric {num}, analytic {a}"
                );
            }
        }
    }
}

/// The gather must not leak across batch or channel: zeroing output channel `j`'s
/// gradient may change input channel `j` and nothing else. A transposed or dropped
/// `(n, c)` term in the `dy` index still produces plausible numbers and can still
/// satisfy the global adjoint identity by accident when the plane sizes match.
#[test]
fn resize_bicubic_dx_does_not_leak_across_planes() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (n, c, h, w, ho, wo) = (2u32, 3u32, 5u32, 4u32, 9u32, 7u32);
    let xn = (n * c * h * w) as usize;
    let yn = (n * c * ho * wo) as usize;
    let plane_in = (h * w) as usize;
    let plane_out = (ho * wo) as usize;
    let params = [n, c, h, w, ho, wo, 0];
    let g = Lcg::new(61).vec(yn);
    let base = run2(&gpu, K_DX, &g, xn, &params);
    for j in 0..(n * c) as usize {
        let mut g2 = g.clone();
        for v in g2[j * plane_out..(j + 1) * plane_out].iter_mut() {
            *v = 0.0;
        }
        let got = run2(&gpu, K_DX, &g2, xn, &params);
        for pl in 0..(n * c) as usize {
            let changed = (0..plane_in).any(|i| got[pl * plane_in + i] != base[pl * plane_in + i]);
            assert_eq!(changed, pl == j, "dx leak: zeroing dy plane {j} changed dx plane {pl}");
        }
    }
}
