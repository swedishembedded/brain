// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified depth-estimation training for its clients. If your team needs expertise in
// scale-invariant losses and gradient-checked training objectives, you can procure our services by sending an email
// to info@swedishembedded.com.

//! P4: the ZipDepth training objective, the MiDaS scale-and-shift-invariant MSE plus multi-scale gradient matching
//! (`zipdepth::loss`), against an independent f64 oracle written from the spec.
//!
//! What is pinned:
//!
//! * the closed-form `(s, b)` equals a least-squares solve in another formulation (covariance form), and the
//!   degenerate cases behave as documented (refuse below two valid pixels, fall back on a constant prediction);
//! * `L_ssi` is invariant to `p -> a p + c` for `a > 0` and is zero when `t = a p + c`;
//! * the loss value and the WHOLE analytic gradient (SSI by the envelope theorem, gradient matching through `s` and
//!   `b`) equal the oracle's value and its central differences, entry by entry, on a random image with a ragged mask;
//! * a masked-out pixel has exactly zero gradient and cannot move the loss.
//!
//! ## Where the tolerances come from
//!
//! The oracle runs in f64 on the same f32 inputs, so every discrepancy is binary32 round-off in the pipeline, which is
//! bounded by `rounding::accumulation_bound(k, S)`: `k` rounded operations on the longest chain from an input to the
//! result, `S` the sum of the magnitudes of the terms combined. The chain is the moment element products (2), the cast
//! of `s, b` (1), the residual (3) and the gradient combination (about 8), with every moment sum taken in f64 on the
//! host; `data()` only draws images whose `det / (a00 a11)` exceeds [`MIN_CONDITIONING`], which bounds the amplification
//! of the moment errors in `s, b` by `1 / MIN_CONDITIONING`, so [`CHAIN`] is that product, rounded up.
//!
//! The numeric derivative is Richardson-extrapolated from steps `h` and `h / 2`; the gap between the two estimates
//! bounds its truncation error and `gamma_64(n) |L| / h` its f64 round-off, and both enter each entry's tolerance.
//! The `|dR|` terms have a kink; `data()` rejects images with a pairwise difference within [`KINK_MARGIN`] of it, far
//! beyond both the step's reach and the f32 error in `R`, so every sign is the same in the oracle and the pipeline.

use brain_testutil::rounding;
use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use zipdepth::loss::{DepthLoss, Moments, SCALES};

/// Smallest `det / (a00 a11)` (the variance-to-second-moment ratio of the prediction) a test image may have.
const MIN_CONDITIONING: f64 = 0.1;
/// Rounded operations on the longest chain (about 14) times the `1 / MIN_CONDITIONING` amplification, rounded up.
const CHAIN: usize = 160;
/// No pairwise residual difference of a test image may be closer to zero than this.
const KINK_MARGIN: f64 = 2e-4;
/// Central-difference step on `p`; far below [`KINK_MARGIN`] and far above f64 round-off.
const STEP: f64 = 1e-6;

const N: usize = 2;
const H: usize = 12;
const W: usize = 16;
const HW: usize = H * W;

// ---- the oracle: the spec's formulas, straightforwardly, in f64 -----------------------------------------------

/// `(s, b)` of `argmin sum_M (s p + b - t)^2` from the covariance form, an independent formulation from the
/// moment-determinant one the pipeline uses.
fn lstsq(p: &[f64], t: &[f64], m: &[f32]) -> (f64, f64) {
    let idx: Vec<usize> = (0..p.len()).filter(|&i| m[i] > 0.0).collect();
    let n = idx.len() as f64;
    let (mp, mt) = (idx.iter().map(|&i| p[i]).sum::<f64>() / n, idx.iter().map(|&i| t[i]).sum::<f64>() / n);
    let cov: f64 = idx.iter().map(|&i| (p[i] - mp) * (t[i] - mt)).sum();
    let var: f64 = idx.iter().map(|&i| (p[i] - mp) * (p[i] - mp)).sum();
    let s = cov / var;
    (s, mt - s * mp)
}

/// Which parts of `s, b` the oracle treats as constants: `Frozen(s, b)` holds them, so its derivative is the direct
/// path only (`s g_i`); `Fitted` re-solves them per evaluation, so its derivative is the total.
#[derive(Clone, Copy)]
enum Fit {
    Fitted,
    Frozen([(f64, f64); N]),
}

/// `(L_ssi, L_grad)` batch means over `N` images, plus the smallest `|pairwise residual difference|` met.
fn oracle(p: &[f64], t: &[f64], m: &[f32], fit: Fit) -> (f64, f64, f64) {
    let (mut ssi, mut gm, mut margin) = (0.0, 0.0, f64::INFINITY);
    for n in 0..N {
        let sl = n * HW..(n + 1) * HW;
        let (pn, tn, mn) = (&p[sl.clone()], &t[sl.clone()], &m[sl]);
        let (s, b) = match fit {
            Fit::Fitted => lstsq(pn, tn, mn),
            Fit::Frozen(sb) => sb[n],
        };
        let valid = mn.iter().filter(|&&v| v > 0.0).count() as f64;
        let r: Vec<f64> = (0..HW).map(|i| s * pn[i] + b - tn[i]).collect();
        ssi += (0..HW).filter(|&i| mn[i] > 0.0).map(|i| r[i] * r[i]).sum::<f64>() / (2.0 * valid);
        let mut g = 0.0;
        for k in 0..SCALES {
            let step = 1usize << k;
            for y in (0..H).step_by(step) {
                for x in (0..W).step_by(step) {
                    let i = y * W + x;
                    for j in [(x + step < W).then(|| i + step), (y + step < H).then(|| i + step * W)].into_iter().flatten() {
                        if mn[i] > 0.0 && mn[j] > 0.0 {
                            g += (r[j] - r[i]).abs();
                            margin = margin.min((r[j] - r[i]).abs());
                        }
                    }
                }
            }
        }
        gm += g / valid;
    }
    (ssi / N as f64, gm / N as f64, margin)
}

/// An image batch the checks apply to: well conditioned, a ragged mask (about 75% valid) and no residual
/// difference near the kink. Drawn again until it is, so the check never depends on a lucky seed.
fn data(rng: &mut Lcg) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    loop {
        let p: Vec<f32> = (0..N * HW).map(|_| 0.2 + rng.unit()).collect();
        // The target is an affine image of a DIFFERENT field plus noise, so the fit is nowhere near exact.
        let t: Vec<f32> = (0..N * HW).map(|i| 0.6 * p[(i * 7 + 3) % (N * HW)] + 0.3 + 0.5 * rng.unit()).collect();
        let m: Vec<f32> = (0..N * HW).map(|_| if rng.unit() < 0.75 { 1.0 } else { 0.0 }).collect();
        let well_conditioned = (0..N).all(|n| {
            let sl = n * HW..(n + 1) * HW;
            moments_of(&p[sl.clone()], &t[sl.clone()], &m[sl]).conditioning() > MIN_CONDITIONING
        });
        let (pd, td): (Vec<f64>, Vec<f64>) = (p.iter().map(|&v| v as f64).collect(), t.iter().map(|&v| v as f64).collect());
        if well_conditioned && oracle(&pd, &td, &m, Fit::Fitted).2 > KINK_MARGIN {
            return (p, t, m);
        }
    }
}

/// The five masked moments summed directly (the pipeline gets them from a kernel's term planes).
fn moments_of(p: &[f32], t: &[f32], m: &[f32]) -> Moments {
    let v = |f: &dyn Fn(f64, f64) -> f64| (0..p.len()).filter(|&i| m[i] > 0.0).map(|i| f(p[i] as f64, t[i] as f64)).sum::<f64>();
    Moments { a00: v(&|p, _| p * p), a01: v(&|p, _| p), a11: v(&|_, _| 1.0), b0: v(&|p, t| p * t), b1: v(&|_, t| t) }
}

// ---- the device side -------------------------------------------------------------------------------------------

struct Run {
    ssi: f32,
    grad: f32,
    d_pred: Vec<f32>,
}

fn upload(gpu: &Gpu, name: &str, v: &[f32]) -> DeviceBuffer {
    gpu.storage_init(name, v)
}

fn run(gpu: &Gpu, p: &[f32], t: &[f32], m: &[f32], alpha: f32) -> Run {
    let loss = DepthLoss::new(gpu, N as u32, H as u32, W as u32);
    let d = gpu.storage((N * HW) as u64);
    let v = loss.eval(gpu, &upload(gpu, "p", p), &upload(gpu, "t", t), &upload(gpu, "m", m), alpha, &d).unwrap();
    Run { ssi: v.ssi, grad: v.grad, d_pred: gpu.read(&d, N * HW) }
}

fn gpu() -> Gpu {
    gpu_core::testgpu::dev(zipdepth::net::PIPELINES)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

// ---- (s, b) ---------------------------------------------------------------------------------------------------

#[test]
fn closed_form_scale_and_shift_equal_a_least_squares_solve() {
    let mut rng = Lcg::random();
    let (p, t, m) = data(&mut rng);
    for n in 0..N {
        let sl = n * HW..(n + 1) * HW;
        let fit = moments_of(&p[sl.clone()], &t[sl.clone()], &m[sl.clone()]).solve().unwrap();
        let (s, b) = lstsq(&f64s(&p[sl.clone()]), &f64s(&t[sl.clone()]), &m[sl]);
        // Both are f64 evaluations of the same smooth function of `HW` terms each; they differ by round-off,
        // magnified by the conditioning the data was drawn to bound.
        let tol = rounding::gamma(HW).max(HW as f64 * f64::EPSILON) / MIN_CONDITIONING * (s.abs() + b.abs() + 1.0);
        assert!(fit.exact, "a well-conditioned image must be solved exactly");
        assert!((fit.s - s).abs() <= tol && (fit.b - b).abs() <= tol, "image {n}: ({}, {}) vs ({s}, {b})", fit.s, fit.b);
    }
}

#[test]
fn fewer_than_two_valid_pixels_is_refused_by_name() {
    let one = Moments { a00: 0.25, a01: 0.5, a11: 1.0, b0: 0.5, b1: 1.0 };
    let err = one.solve().unwrap_err();
    assert!(err.contains("valid pixel"), "{err}");
}

#[test]
fn a_constant_prediction_falls_back_to_unit_scale_and_mean_residual() {
    // p == 0.5 on every valid pixel: det = 0, any s fits equally. The documented fallback is s = 1, b = mean(t - p).
    let (p, t) = ([0.5f32; 4], [1.0f32, 2.0, 3.0, 6.0]);
    let fit = moments_of(&p, &t, &[1.0; 4]).solve().unwrap();
    assert!(!fit.exact);
    assert_eq!(fit.s, 1.0);
    assert!((fit.b - (3.0 - 0.5)).abs() < 1e-12, "b = {}", fit.b);
}

// ---- the value ------------------------------------------------------------------------------------------------

#[test]
fn ssi_is_invariant_to_a_positive_affine_map_of_the_prediction() {
    let gpu = gpu();
    let mut rng = Lcg::random();
    let (p, t, m) = data(&mut rng);
    let (a, c) = (2.5f32, -0.3f32);
    let q: Vec<f32> = p.iter().map(|&v| a * v + c).collect();
    let (l0, l1) = (run(&gpu, &p, &t, &m, 0.5).ssi as f64, run(&gpu, &q, &t, &m, 0.5).ssi as f64);
    // `L_ssi` is a mean of `R^2 / 2` with `|R| <= |s p| + |b| + |t|` per pixel, each accumulated over `CHAIN`
    // rounded operations; the two evaluations differ by twice the bound on the loss.
    let r_scale = f64s(&t).iter().cloned().fold(0.0, f64::max) * 3.0;
    let bound = rounding::agreement_bound(CHAIN, 0.5 * r_scale * r_scale);
    assert!((l0 - l1).abs() <= bound, "{l0} vs {l1}, bound {bound}");
    // ...and the oracle agrees on the value itself.
    let (o, _, _) = oracle(&f64s(&p), &f64s(&t), &m, Fit::Fitted);
    assert!((l0 - o).abs() <= rounding::accumulation_bound(CHAIN, 0.5 * r_scale * r_scale), "{l0} vs oracle {o}");
}

#[test]
fn ssi_is_zero_when_the_target_is_an_affine_map_of_the_prediction() {
    let gpu = gpu();
    let mut rng = Lcg::random();
    let (p, _, m) = data(&mut rng);
    let t: Vec<f32> = p.iter().map(|&v| 1.7 * v + 0.4).collect();
    let r = run(&gpu, &p, &t, &m, 0.5);
    // R = s p + b - t is zero up to the round-off of forming it: L = R^2 / 2 <= (gamma * (|s p| + |b| + |t|))^2 / 2.
    let scale = f64s(&t).iter().cloned().fold(0.0, f64::max) * 3.0;
    let r_bound = rounding::accumulation_bound(CHAIN, scale);
    assert!((r.ssi as f64) <= 0.5 * r_bound * r_bound, "ssi {} over {}", r.ssi, 0.5 * r_bound * r_bound);
    assert!((r.grad as f64) <= r_bound * 2.0 * (SCALES as f64) * 2.0, "grad {} must vanish with the residual", r.grad);
}

#[test]
fn loss_value_equals_the_oracle() {
    let gpu = gpu();
    let mut rng = Lcg::random();
    let (p, t, m) = data(&mut rng);
    let r = run(&gpu, &p, &t, &m, 0.5);
    let (ssi, gm, _) = oracle(&f64s(&p), &f64s(&t), &m, Fit::Fitted);
    let tol = |v: f64| rounding::accumulation_bound(CHAIN, v);
    assert!((r.ssi as f64 - ssi).abs() <= tol(ssi), "ssi {} vs {ssi}", r.ssi);
    assert!((r.grad as f64 - gm).abs() <= tol(gm), "grad {} vs {gm}", r.grad);
}

// ---- the gradient ---------------------------------------------------------------------------------------------

/// Central difference of `f` in `p_i` at step `h`.
fn central(f: &dyn Fn(&[f64]) -> f64, p: &[f64], i: usize, h: f64) -> f64 {
    let (mut a, mut b) = (p.to_vec(), p.to_vec());
    a[i] += h;
    b[i] -= h;
    (f(&a) - f(&b)) / (2.0 * h)
}

/// Richardson-extrapolated derivative and an estimate of what is left of its truncation error.
fn derivative(f: &dyn Fn(&[f64]) -> f64, p: &[f64], i: usize) -> (f64, f64) {
    let (d1, d2) = (central(f, p, i, STEP), central(f, p, i, STEP / 2.0));
    ((4.0 * d2 - d1) / 3.0, (d1 - d2).abs() / 3.0)
}

#[test]
fn analytic_gradient_equals_central_differences_of_the_whole_loss() {
    let gpu = gpu();
    let mut rng = Lcg::random();
    let (p, t, m) = data(&mut rng);
    let alpha = 0.5f32;
    let r = run(&gpu, &p, &t, &m, alpha);
    let (pd, td) = (f64s(&p), f64s(&t));
    let a = alpha as f64;
    let total = |p: &[f64]| {
        let (s, g, _) = oracle(p, &td, &m, Fit::Fitted);
        s + a * g
    };
    let ssi = |p: &[f64]| oracle(p, &td, &m, Fit::Fitted).0;
    // (s, b) frozen at their values at `pd`: the derivative of the gradient term then has only its direct path.
    let frozen = Fit::Frozen(std::array::from_fn(|n| lstsq(&pd[n * HW..(n + 1) * HW], &td[n * HW..(n + 1) * HW], &m[n * HW..(n + 1) * HW])));
    let direct = |p: &[f64]| oracle(p, &td, &m, frozen).1;
    let loss_scale = total(&pd).abs();

    let mut worst: f64 = 0.0;
    let mut through_checked = 0;
    assert_eq!(m.len(), N * HW, "one mask entry per pixel");
    for (i, &mi) in m.iter().enumerate() {
        let (g_num, trunc) = derivative(&total, &pd, i);
        let (g_ssi, _) = derivative(&ssi, &pd, i);
        let (g_direct, _) = derivative(&direct, &pd, i);
        let g_through = g_num - g_ssi - a * g_direct;
        // S: the magnitudes of the three components the entry is the sum of; `CHAIN` covers the rounding inside each.
        let s = g_ssi.abs() + a * g_direct.abs() + a * g_through.abs();
        // f64 round-off of the difference quotient: n terms summed, divided by the step.
        let f64_roundoff = (N * HW) as f64 * f64::EPSILON * loss_scale / STEP;
        let tol = rounding::accumulation_bound(CHAIN, s) + trunc + f64_roundoff;
        let err = (r.d_pred[i] as f64 - g_num).abs();
        assert!(err <= tol, "pixel {i} (valid {mi}): analytic {} vs numeric {g_num}, error {err} > {tol}", r.d_pred[i]);
        if mi == 0.0 {
            assert_eq!(r.d_pred[i], 0.0, "masked pixel {i} must receive exactly no gradient");
        } else if g_through.abs() > 10.0 * tol {
            through_checked += 1;
        }
        worst = worst.max(err / tol);
    }
    // The through-(s, b) path must be a real part of what was compared, or the check would pass without it.
    assert!(through_checked > N * HW / 4, "only {through_checked} entries have a through-(s,b) component above the tolerance");
    eprintln!("worst error / tolerance: {worst:.3}");
}

#[test]
fn a_masked_out_pixel_cannot_move_the_loss() {
    let gpu = gpu();
    let mut rng = Lcg::random();
    let (p, t, m) = data(&mut rng);
    let hidden = m.iter().position(|&v| v == 0.0).expect("the mask has a hole");
    let (mut p2, mut t2) = (p.clone(), t.clone());
    p2[hidden] += 5.0;
    t2[hidden] -= 3.0;
    let (a, b) = (run(&gpu, &p, &t, &m, 0.5), run(&gpu, &p2, &t2, &m, 0.5));
    assert_eq!((a.ssi, a.grad), (b.ssi, b.grad));
    assert_eq!(a.d_pred, b.d_pred);
}

#[test]
fn a_constant_prediction_trains_through_the_fallback() {
    // The degenerate branch must still deliver a finite loss and a gradient (the net's output is constant at init).
    let gpu = gpu();
    let mut rng = Lcg::random();
    let (_, t, m) = data(&mut rng);
    let p = vec![0.5f32; N * HW];
    let r = run(&gpu, &p, &t, &m, 0.5);
    assert!(r.ssi.is_finite() && r.grad.is_finite() && r.d_pred.iter().all(|g| g.is_finite()));
    assert!(r.d_pred.iter().any(|&g| g != 0.0));
}

#[test]
fn an_image_with_under_two_valid_pixels_is_refused_with_its_index() {
    let gpu = gpu();
    let mut rng = Lcg::random();
    let (p, t, mut m) = data(&mut rng);
    m[HW..].fill(0.0);
    m[HW] = 1.0;
    let loss = DepthLoss::new(&gpu, N as u32, H as u32, W as u32);
    let d = gpu.storage((N * HW) as u64);
    let err = loss.eval(&gpu, &upload(&gpu, "p", &p), &upload(&gpu, "t", &t), &upload(&gpu, "m", &m), 0.5, &d).unwrap_err();
    assert!(err.contains("image 1") && err.contains("valid pixel"), "{err}");
}
