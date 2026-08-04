// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the `maxpool2d` / `maxpool2d_dx` kernel family, driven
//! directly through `gpu_core` like `depth_kernels.rs` / `glue.rs` — no model is
//! built.
//!
//! `maxpool2d` is the GENERALIZATION of the deleted `maxpool5` (the same kernel
//! pinned at `stride = 1`; `K` and `pad` were already its parameters), and it
//! took over that kernel's only call site, SPPF. Two consequences shape this
//! file:
//!
//! 1. **The base case is a hard gate.** At `K=5, stride=1, pad=2` the kernel is
//!    what SPPF dispatches, so it is checked *bit for bit* in both `y` and
//!    `argmax`. Unlike a reassociated sum, a max-pool is pure SELECTION — no
//!    arithmetic is reordered, so bit-equality is achievable and anything less is
//!    a real divergence, not float noise. A generalization that changes its own
//!    base case is a regression however good its other tests look.
//! 2. **`stride` is the only new degree of freedom, and a wrong stride still
//!    produces plausible numbers.** So it is attacked three ways: a plain-Rust
//!    forward oracle at five (K, stride, pad) configurations including the two
//!    real call sites (Hiera `q_pool` K=2/s=2, SCRFD stem K=3/s=2/pad=1); a
//!    single-tap probe where one hot input pixel must appear at an
//!    independently-derived set of output positions; and a `stride > K` case
//!    where some input pixels are covered by NO window and must receive exactly
//!    zero gradient.
//!
//! The backward is tested two ways. `maxpool2d_dx` is the adjoint of the LINEAR
//! selection operator obtained by FREEZING the forward's argmax, so
//!     <A(x), dy> == <x, Aᵀ(dy)>
//! holds to fp32 round-off for all `x, dy` — cheaper and far sharper than FD, and
//! a dropped coverage bound or an off-by-one in the `ceil((hp-K+1)/stride)`
//! interval breaks it immediately. Central differences are then run as well,
//! because adjointness alone cannot catch an argmax that points at the wrong
//! pixel (a consistently-wrong selection is still a consistent linear map).
//! Max-pool is piecewise linear, so with well-separated random inputs FD is exact
//! away from ties.
//!
//! Run with `BRAIN_DEVICE=cpu` for a headless gating run.

use data::rng::Lcg;
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("maxpool2d", kernels::MAXPOOL2D),       // 0
    ("maxpool2d_dx", kernels::MAXPOOL2D_DX), // 1
];
const K_FWD: usize = 0;
const K_DX: usize = 1;

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

// ---------------------------------------------------------------------------
// Shape helper + plain-Rust oracle. Independently derived from the op's
// definition (torch `MaxPool2d`, ceil_mode=false), NOT transcribed from the
// WGSL — an oracle that shares its derivation with the thing it checks proves
// nothing.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
struct Pool {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    stride: usize,
    pad: usize,
}
impl Pool {
    fn ho(&self) -> usize {
        (self.h + 2 * self.pad - self.k) / self.stride + 1
    }
    fn wo(&self) -> usize {
        (self.w + 2 * self.pad - self.k) / self.stride + 1
    }
    fn x_len(&self) -> usize {
        self.n * self.c * self.h * self.w
    }
    fn y_len(&self) -> usize {
        self.n * self.c * self.ho() * self.wo()
    }
    /// `Params` for both kernels: [N, C, H, W, K, stride, pad, Ho, Wo].
    fn params(&self) -> [u32; 9] {
        [
            self.n as u32,
            self.c as u32,
            self.h as u32,
            self.w as u32,
            self.k as u32,
            self.stride as u32,
            self.pad as u32,
            self.ho() as u32,
            self.wo() as u32,
        ]
    }
}

/// Forward oracle: returns (y, argmax-as-input-flat-index).
fn ref_fwd(p: Pool, x: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let (ho, wo) = (p.ho(), p.wo());
    let mut y = vec![0.0f32; p.y_len()];
    let mut am = vec![0usize; p.y_len()];
    for nn in 0..p.n {
        for cc in 0..p.c {
            for oh in 0..ho {
                for ow in 0..wo {
                    let h0 = oh as isize * p.stride as isize - p.pad as isize;
                    let w0 = ow as isize * p.stride as isize - p.pad as isize;
                    let mut best = f32::NEG_INFINITY;
                    let mut bi = 0usize;
                    let mut found = false;
                    for kh in 0..p.k {
                        let hi = h0 + kh as isize;
                        if hi < 0 || hi >= p.h as isize {
                            continue;
                        }
                        for kw in 0..p.k {
                            let wi = w0 + kw as isize;
                            if wi < 0 || wi >= p.w as isize {
                                continue;
                            }
                            let ii = ((nn * p.c + cc) * p.h + hi as usize) * p.w + wi as usize;
                            if !found || x[ii] > best {
                                best = x[ii];
                                bi = ii;
                                found = true;
                            }
                        }
                    }
                    let oi = ((nn * p.c + cc) * ho + oh) * wo + ow;
                    y[oi] = if found { best } else { 0.0 };
                    am[oi] = bi;
                }
            }
        }
    }
    (y, am)
}

/// Backward oracle: the scatter form (`dx[argmax[o]] += dy[o]`), which is what
/// the gather kernel must equal.
fn ref_dx(p: Pool, am: &[usize], dy: &[f32]) -> Vec<f32> {
    let mut dx = vec![0.0f32; p.x_len()];
    for oi in 0..p.y_len() {
        dx[am[oi]] += dy[oi];
    }
    dx
}

// ---------------------------------------------------------------------------
// Dispatch helpers. Geometry is stated once here so every test uses it:
//   forward  : threads = N*C*Ho*Wo, buffers [x, y, argmax]
//   backward : threads = N*C*H*W,   buffers [dy, argmax, dx]
// ---------------------------------------------------------------------------
fn gpu_fwd(gpu: &Gpu, p: Pool, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let xb = gpu.storage_init("x", x);
    let yb = gpu.storage(p.y_len() as u64);
    let amb = gpu.storage(p.y_len() as u64);
    let s = gpu.step(K_FWD, &[&xb, &yb, &amb], &p.params(), p.y_len() as u32);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    (gpu.read(&yb, p.y_len()), gpu.read(&amb, p.y_len()))
}

/// Forward then backward, sharing the GPU-produced argmax (self-consistency is
/// the point: the backward must invert the window the forward actually used).
fn gpu_fwd_bwd(gpu: &Gpu, p: Pool, x: &[f32], dy: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let xb = gpu.storage_init("x", x);
    let yb = gpu.storage(p.y_len() as u64);
    let amb = gpu.storage(p.y_len() as u64);
    let dyb = gpu.storage_init("dy", dy);
    let dxb = gpu.storage(p.x_len() as u64);
    let sf = gpu.step(K_FWD, &[&xb, &yb, &amb], &p.params(), p.y_len() as u32);
    gpu.submit(&[], &[sf]);
    let sb = gpu.step(K_DX, &[&dyb, &amb, &dxb], &p.params(), p.x_len() as u32);
    gpu.submit(&[], &[sb]);
    gpu.poll_wait();
    (
        gpu.read(&yb, p.y_len()),
        gpu.read(&amb, p.y_len()),
        gpu.read(&dxb, p.x_len()),
    )
}

/// The five configurations under test. The first two are the call sites this
/// family exists for; the last three are the edge cases stride introduces.
fn configs() -> Vec<(&'static str, Pool)> {
    vec![
        // SAM 2 Hiera q_pool: MaxPool2d(k=2, s=2), exact halving.
        ("hiera_q_pool k2s2p0", Pool { n: 2, c: 3, h: 8, w: 8, k: 2, stride: 2, pad: 0 }),
        // SCRFD / ResNet stem: k=3 s=2 pad=1.
        ("scrfd_stem k3s2p1", Pool { n: 1, c: 2, h: 7, w: 9, k: 3, stride: 2, pad: 1 }),
        // The maxpool5 base case (SPPF), same-size output.
        ("sppf k5s1p2", Pool { n: 1, c: 2, h: 6, w: 6, k: 5, stride: 1, pad: 2 }),
        // stride > K: some input pixels are covered by NO window.
        ("gappy k2s3p0", Pool { n: 1, c: 2, h: 8, w: 8, k: 2, stride: 3, pad: 0 }),
        // Non-divisible: the trailing row/column is dropped (ceil_mode=false).
        ("ragged k3s2p0", Pool { n: 1, c: 1, h: 6, w: 7, k: 3, stride: 2, pad: 0 }),
    ]
}

// ===========================================================================
// 1. Base case: the stride-1 behaviour maxpool5 used to provide.
// ===========================================================================

/// `maxpool2d` replaced `maxpool5` (SPPF's K=5/stride=1/pad=2 pool), so the
/// stride-1 base case is a hard gate rather than just another sweep point. Max-pool
/// reorders no arithmetic — it selects — so the tolerance is bit-equality on `y`
/// and exact equality on `argmax`, against the independently-written oracle.
///
/// The distinctive part is the data: a third of the input is forced strictly
/// negative, which is what catches a max seeded from `0.0` instead of from the
/// first in-bounds tap (`found`). With all-positive data a padding-wins bug is
/// invisible, and every SPPF call site pools an already-SiLU'd — i.e. possibly
/// negative — activation.
#[test]
fn stride1_base_case_including_all_negative_windows() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);

    for &(n, c, h, w, k, pad) in &[(1usize, 2usize, 6usize, 6usize, 5usize, 2usize), (2, 3, 5, 7, 3, 1)] {
        let p = Pool { n, c, h, w, k, stride: 1, pad };
        assert_eq!(p.ho(), h, "stride-1 with pad=K/2 must preserve H");
        assert_eq!(p.wo(), w, "stride-1 with pad=K/2 must preserve W");

        let mut x = Lcg::new(101).vec(p.x_len());
        for v in x.iter_mut().take(p.x_len() / 3) {
            *v = -(v.abs()) - 1.0;
        }

        let (y, am) = gpu_fwd(&gpu, p, &x);
        let (ry, ram) = ref_fwd(p, &x);
        assert!(ry.iter().any(|&v| v < 0.0), "k{k}p{pad}: the negative region did not survive to y");

        for i in 0..p.y_len() {
            assert_eq!(
                y[i].to_bits(),
                ry[i].to_bits(),
                "k{k}p{pad}: y[{i}] kernel {} vs oracle {}",
                y[i],
                ry[i]
            );
            assert_eq!(am[i] as usize, ram[i], "k{k}p{pad}: argmax[{i}] diverged");
        }
    }
}

// ===========================================================================
// 2. Forward vs the plain-Rust oracle, at every configuration.
// ===========================================================================

/// Forward parity. `y` is a copy of an input element, so the tolerance is
/// bit-equality, not an epsilon: any difference means a different element was
/// selected, i.e. a wrong window. `argmax` is checked exactly and is additionally
/// required to *point at* the reported max.
#[test]
fn forward_matches_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for (tag, p) in configs() {
        let x = Lcg::new(2024 + p.h as u64 * 31 + p.k as u64).vec(p.x_len());
        let (y, am) = gpu_fwd(&gpu, p, &x);
        let (ry, ram) = ref_fwd(p, &x);
        assert_eq!(y.len(), p.y_len(), "{tag}: Ho/Wo sizing");
        for i in 0..p.y_len() {
            assert_eq!(y[i].to_bits(), ry[i].to_bits(), "{tag}: y[{i}] {} vs {}", y[i], ry[i]);
            let gi = am[i] as usize;
            assert_eq!(gi, ram[i], "{tag}: argmax[{i}] {gi} vs {}", ram[i]);
            assert_eq!(x[gi].to_bits(), ry[i].to_bits(), "{tag}: argmax[{i}] does not point at the max");
        }
    }
}

/// A single-tap probe that a wrong `stride` cannot accidentally satisfy: one hot
/// pixel in a field of a constant floor, whose response must land at exactly the
/// output positions whose window covers it — a set derived here from the window
/// definition directly, not from the kernel's coverage arithmetic.
#[test]
fn single_hot_pixel_lands_at_the_right_outputs() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let p = Pool { n: 1, c: 1, h: 9, w: 9, k: 3, stride: 2, pad: 1 };
    for &(hh, ww) in &[(0usize, 0usize), (4, 3), (8, 8), (5, 0)] {
        let mut x = vec![-1.0f32; p.x_len()];
        x[hh * p.w + ww] = 1.0;
        let (y, _) = gpu_fwd(&gpu, p, &x);
        for oh in 0..p.ho() {
            for ow in 0..p.wo() {
                let h0 = oh as isize * p.stride as isize - p.pad as isize;
                let w0 = ow as isize * p.stride as isize - p.pad as isize;
                let covers = (hh as isize) >= h0
                    && (hh as isize) < h0 + p.k as isize
                    && (ww as isize) >= w0
                    && (ww as isize) < w0 + p.k as isize;
                let want = if covers { 1.0f32 } else { -1.0f32 };
                let got = y[oh * p.wo() + ow];
                assert_eq!(got, want, "hot({hh},{ww}) -> out({oh},{ow}): {got} want {want}");
            }
        }
    }
}

// ===========================================================================
// 3. Backward: exact vs the scatter oracle, adjointness, and finite differences.
// ===========================================================================

/// The gather backward must equal the scatter reference exactly (it is a sum of
/// the same f32 values; only the order of a handful of adds can differ, and at
/// these shapes each dx cell takes at most a few terms).
#[test]
fn dx_matches_scatter_reference() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for (tag, p) in configs() {
        let x = Lcg::new(77 + p.k as u64).vec(p.x_len());
        let dy = Lcg::new(1337 + p.stride as u64).vec(p.y_len());
        let (_, am, dx) = gpu_fwd_bwd(&gpu, p, &x, &dy);
        let am_u: Vec<usize> = am.iter().map(|&v| v as usize).collect();
        let rdx = ref_dx(p, &am_u, &dy);
        for i in 0..p.x_len() {
            assert!(
                (dx[i] - rdx[i]).abs() < 1e-5,
                "{tag}: dx[{i}] {} vs {}",
                dx[i],
                rdx[i]
            );
        }
        // Gradient mass is conserved: every dy entry is delivered exactly once.
        let sum_dx: f64 = dx.iter().map(|&v| v as f64).sum();
        let sum_dy: f64 = dy.iter().map(|&v| v as f64).sum();
        assert!(
            (sum_dx - sum_dy).abs() < 1e-3 * sum_dy.abs().max(1.0),
            "{tag}: gradient mass {sum_dx} vs {sum_dy} — a coverage bound dropped or double-counted dy"
        );
    }
}

/// With the argmax frozen, max-pool IS a linear selection operator `A`, and
/// `maxpool2d_dx` must be exactly `Aᵀ`: `<A(x), dy> == <x, Aᵀ(dy)>` for all
/// `x, dy`, to fp32 round-off. This is the sharp test for the coverage interval
/// `ho in [ceil((hi+pad-K+1)/stride), floor((hi+pad)/stride)] ∩ [0,Ho)` — an
/// off-by-one at either end, or a spurious divisibility filter, breaks it.
#[test]
fn dx_is_the_adjoint_of_the_frozen_argmax_operator() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    for (tag, p) in configs() {
        let x = Lcg::new(555 + p.w as u64).vec(p.x_len());
        let dy = Lcg::new(9001 + p.h as u64).vec(p.y_len());
        let (y, _, dx) = gpu_fwd_bwd(&gpu, p, &x, &dy);
        let lhs = dot(&y, &dy);
        let rhs = dot(&x, &dx);
        let tol = 1e-4 * lhs.abs().max(rhs.abs()).max(1.0);
        assert!(
            (lhs - rhs).abs() < tol,
            "{tag}: adjointness broken — <A(x),dy> = {lhs}, <x,Aᵀ(dy)> = {rhs} (diff {:.3e})",
            (lhs - rhs).abs()
        );
    }
}

/// Central differences on `L = <y, dy>`. Adjointness alone cannot catch an
/// argmax that consistently points at the wrong pixel (a wrong-but-consistent
/// selection is still a consistent linear map); FD re-derives the gradient from
/// the forward itself and does. Max-pool is piecewise linear, so away from ties
/// FD is exact — the tolerance covers fp32 cancellation in `(L+ − L−)/2ε`, not
/// truncation error.
#[test]
fn dx_matches_finite_differences() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let eps = 1e-3f32;
    for (tag, p) in configs() {
        let x = Lcg::new(31337 + p.k as u64 * 7).vec(p.x_len());
        let dy = Lcg::new(4242 + p.stride as u64 * 13).vec(p.y_len());
        let (_, _, dx) = gpu_fwd_bwd(&gpu, p, &x, &dy);

        let loss = |xv: &[f32]| -> f64 {
            let (y, _) = ref_fwd(p, xv);
            dot(&y, &dy)
        };
        // Sample rather than sweep: step_by keeps this cheap while still hitting
        // border, interior and (for stride > K) uncovered pixels.
        let stride_i = (p.x_len() / 24).max(1);
        for i in (0..p.x_len()).step_by(stride_i) {
            let mut xp = x.to_vec();
            xp[i] += eps;
            let mut xm = x.to_vec();
            xm[i] -= eps;
            let num = (loss(&xp) - loss(&xm)) / (2.0 * eps as f64);
            let ana = dx[i] as f64;
            let tol = 1e-2 + 1e-2 * num.abs().max(ana.abs());
            assert!(
                (num - ana).abs() < tol,
                "{tag}: dx[{i}] analytic {ana} vs fd {num}"
            );
        }
    }
}

/// At `stride > K` some input pixels lie in no window at all. Their gradient must
/// be exactly zero, and the count of such pixels must match the window geometry —
/// a `_dx` that widened its coverage interval "to be safe" would hand them
/// gradient and pass every tolerance-based test above.
#[test]
fn uncovered_pixels_get_exactly_zero_gradient() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let p = Pool { n: 1, c: 2, h: 8, w: 8, k: 2, stride: 3, pad: 0 };
    let x = Lcg::new(606).vec(p.x_len());
    let dy: Vec<f32> = (0..p.y_len()).map(|i| 1.0 + i as f32).collect(); // all non-zero
    let (_, _, dx) = gpu_fwd_bwd(&gpu, p, &x, &dy);

    let covered = |i: usize| -> bool {
        let wi = i % p.w;
        let hi = (i / p.w) % p.h;
        let in_h = (0..p.ho()).any(|oh| {
            let h0 = oh * p.stride;
            hi >= h0 && hi < h0 + p.k
        });
        let in_w = (0..p.wo()).any(|ow| {
            let w0 = ow * p.stride;
            wi >= w0 && wi < w0 + p.k
        });
        in_h && in_w
    };
    let mut n_uncovered = 0;
    for i in 0..p.x_len() {
        if !covered(i) {
            n_uncovered += 1;
            assert_eq!(dx[i], 0.0, "uncovered input {i} received gradient {}", dx[i]);
        }
    }
    // Ho = Wo = 3 windows of width 2 at stride 3 cover rows {0,1,3,4,6,7} of 8,
    // so 2 of 8 rows and 2 of 8 cols are uncovered: 8*8 - 6*6 = 28 per channel.
    assert_eq!(n_uncovered, 28 * p.c, "the test's own coverage model drifted");
}

// ===========================================================================
// 4. Exhaustive shape sweep — the five hand-picked configs above are not enough.
// ===========================================================================

/// The named configurations cover the call sites, not the *shape space*. This
/// sweeps every `(K, stride, pad)` a caller can legally ask for over a set of
/// deliberately awkward `(H, W)` — including `K > H` (the whole row is one
/// window), `K = 1` (pool is the identity and `argmax == ii` everywhere),
/// `H = 1` / `W = 1` (degenerate axes), `pad = K - 1` (wider than torch's own
/// `pad <= K/2` limit, so the border logic is exercised past its call sites),
/// and every non-divisible `(H + 2p - K) % stride`.
///
/// Both directions are checked against the plain-Rust oracles with the SAME
/// exactness the single-config tests use: bit-equality on `y`/`argmax` and an
/// exact match on `dx` (the gather sums the same f32 values the scatter does,
/// and at these sizes each cell takes at most a handful of terms). This is what
/// catches a coverage bound that is right at the sampled shapes and off by one
/// somewhere else — the `ceil((hp-K+1)/stride)` interval is the whole risk in
/// the backward and a single ragged shape does not pin it down.
#[test]
fn shape_sweep_forward_and_backward_match_the_oracles() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut checked = 0usize;
    for &(h, w) in &[(1usize, 1usize), (1, 7), (7, 1), (4, 4), (5, 9), (8, 6), (9, 9), (11, 5)] {
        for k in 1..=5usize {
            for stride in 1..=4usize {
                // A pool window must reach past the padding on both sides, so
                // pad <= K-1; torch is stricter (pad <= K/2) but the kernel is
                // not, and the extra column is where the border math shows.
                for pad in 0..k {
                    if h + 2 * pad < k || w + 2 * pad < k {
                        continue;
                    }
                    let p = Pool { n: 2, c: 2, h, w, k, stride, pad };
                    let tag = format!("h{h}w{w}k{k}s{stride}p{pad}");
                    let x = Lcg::new(9_000 + (checked as u64) * 17).vec(p.x_len());
                    let dy = Lcg::new(4_000 + (checked as u64) * 29).vec(p.y_len());
                    let (y, am, dx) = gpu_fwd_bwd(&gpu, p, &x, &dy);

                    let (ry, ram) = ref_fwd(p, &x);
                    for i in 0..p.y_len() {
                        assert_eq!(y[i].to_bits(), ry[i].to_bits(), "{tag}: y[{i}] {} vs {}", y[i], ry[i]);
                        assert_eq!(am[i] as usize, ram[i], "{tag}: argmax[{i}]");
                    }
                    let am_u: Vec<usize> = am.iter().map(|&v| v as usize).collect();
                    let rdx = ref_dx(p, &am_u, &dy);
                    for i in 0..p.x_len() {
                        assert!(
                            (dx[i] - rdx[i]).abs() < 1e-6,
                            "{tag}: dx[{i}] {} vs scatter {}",
                            dx[i],
                            rdx[i]
                        );
                    }
                    // Adjointness of the frozen-argmax operator, per shape.
                    let (lhs, rhs) = (dot(&y, &dy), dot(&x, &dx));
                    assert!(
                        (lhs - rhs).abs() < 1e-4 * lhs.abs().max(rhs.abs()).max(1.0),
                        "{tag}: adjointness broken — {lhs} vs {rhs}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 300, "sweep collapsed to {checked} configurations");
}

/// `pad >= K` lets a window sit entirely in the padding. torch rejects that
/// (`pad <= K/2`), but nothing in the kernel does, and the forward's
/// `found`-seeding then writes `y = 0, argmax = 0` — a fabricated pointer at
/// input 0. The question that matters is whether the *backward* ever reads it.
/// It cannot: an output covers input `(0,0)` on both axes only if row 0 and
/// col 0 are both inside its window, and such a window has an in-bounds tap, so
/// `found` is true and the pointer is real. The all-padding outputs are
/// therefore outside every input's coverage interval and their `dy` is dropped —
/// which is the correct derivative of a constant. Note this is the ONE regime
/// where the gather backward deliberately disagrees with the naive scatter
/// oracle (`dx[argmax[o]] += dy[o]`), so it is checked against the invariant
/// rather than against `ref_dx`.
#[test]
fn all_padding_windows_are_inert_in_both_directions() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    // K=2, pad=2 > K: outputs ho=0 and ho=Ho-1 are pure padding on the h axis.
    let p = Pool { n: 1, c: 1, h: 4, w: 4, k: 2, stride: 1, pad: 2 };
    let x: Vec<f32> = (0..p.x_len()).map(|i| -1.0 - i as f32).collect(); // all negative
    let dy: Vec<f32> = (0..p.y_len()).map(|i| 1.0 + i as f32).collect(); // all non-zero
    let (y, am, dx) = gpu_fwd_bwd(&gpu, p, &x, &dy);

    let mut fabricated = 0usize;
    for oh in 0..p.ho() {
        for ow in 0..p.wo() {
            let h0 = oh as isize - p.pad as isize;
            let w0 = ow as isize - p.pad as isize;
            let live = (0..p.k as isize).any(|d| (0..p.h as isize).contains(&(h0 + d)))
                && (0..p.k as isize).any(|d| (0..p.w as isize).contains(&(w0 + d)));
            let oi = oh * p.wo() + ow;
            if !live {
                fabricated += 1;
                assert_eq!(y[oi], 0.0, "all-padding output {oi} should be the 0 seed");
                assert_eq!(am[oi] as usize, 0, "all-padding output {oi} argmax seed");
            } else {
                assert!(y[oi] < 0.0, "live output {oi} must select a real (negative) tap");
            }
        }
    }
    assert!(fabricated > 0, "config no longer produces an all-padding window");

    // The fabricated argmax=0 pointers must contribute nothing: dx equals the
    // sum over the LIVE outputs only, which is exactly what ref_fwd/ref_dx give
    // once the dead outputs are masked out of dy.
    let (_, ram) = ref_fwd(p, &x);
    let mut dy_live = dy.clone();
    for oh in 0..p.ho() {
        for ow in 0..p.wo() {
            let h0 = oh as isize - p.pad as isize;
            let w0 = ow as isize - p.pad as isize;
            let live = (0..p.k as isize).any(|d| (0..p.h as isize).contains(&(h0 + d)))
                && (0..p.k as isize).any(|d| (0..p.w as isize).contains(&(w0 + d)));
            if !live {
                dy_live[oh * p.wo() + ow] = 0.0;
            }
        }
    }
    let want = ref_dx(p, &ram, &dy_live);
    for i in 0..p.x_len() {
        assert!(
            (dx[i] - want[i]).abs() < 1e-6,
            "dx[{i}] {} vs live-only scatter {} — a padding-only window leaked gradient",
            dx[i],
            want[i]
        );
    }
}

/// `K = 1` is the algebraic identity for the family and pins the index math on
/// its own: at stride 1 / pad 0 the forward must copy `x` verbatim with
/// `argmax[i] == i`, and the backward must copy `dy` verbatim. Any off-by-one in
/// the window origin or in the coverage interval shows up here as a shift rather
/// than as a tolerance failure.
#[test]
fn k1_stride1_is_the_identity_in_both_directions() {
    if skip() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let p = Pool { n: 2, c: 3, h: 5, w: 7, k: 1, stride: 1, pad: 0 };
    assert_eq!((p.ho(), p.wo()), (p.h, p.w));
    let x = Lcg::new(4711).vec(p.x_len());
    let dy = Lcg::new(1174).vec(p.y_len());
    let (y, am, dx) = gpu_fwd_bwd(&gpu, p, &x, &dy);
    for i in 0..p.x_len() {
        assert_eq!(y[i].to_bits(), x[i].to_bits(), "y[{i}] is not a copy of x");
        assert_eq!(am[i] as usize, i, "argmax[{i}] is not the identity");
        assert_eq!(dx[i].to_bits(), dy[i].to_bits(), "dx[{i}] is not a copy of dy");
    }
}
