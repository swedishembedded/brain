// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated finite-difference gradient check for the exact (erf) GELU pair:
//! `gelu_erf` (forward) + `gelu_erf_bwd` (its gradient).
//!
//! These tests do NOT build any model: they drive the WGSL kernels directly via
//! `gpu_core`, exactly like `mse_fd.rs`.
//!
//! WHY THIS FILE EXISTS. brain has two GELUs:
//!   * `gelu` / `gelu_bwd`     — the tanh approximation (GPT-2 style)
//!   * `gelu_erf` / `gelu_erf_bwd` — the exact erf form (torch's default `F.gelu`)
//!
//! Until `gelu_erf_bwd` was added, `gelu_erf` had NO backward at all, so anything
//! training through it had to borrow `gelu_bwd` — the derivative of a *different
//! function*. The two agree to ~1e-3, which is comfortably inside the global
//! gradcheck tolerance (rtol 8e-2). A model that mispaired them would therefore
//! pass every gate while training on the wrong gradient. `mispaired_gelu_bwd_is_
//! wrong_but_within_gradcheck_tolerance` pins that trap so it stays visible.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use gpu_core::Gpu;

// Kernel order passed to Gpu::new; indices below reference these.
static KERNELS: &[(&str, &str)] = &[
    ("gelu_erf", kernels::GELU_ERF),         // 0
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD), // 1
    ("gelu", kernels::GELU),                 // 2
    ("gelu_bwd", kernels::GELU_BWD),         // 3
];

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * 3.0 // ~[-3,3)
}

/// Elementwise forward for kernel `k` (0 = gelu_erf, 2 = gelu).
fn fwd(gpu: &Gpu, k: usize, x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let xb = gpu.storage_init("x", x);
    let out = gpu.storage(n as u64);
    let st = gpu.step(k, &[&xb, &out], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&out, n)
}

/// Elementwise backward for kernel `k` (1 = gelu_erf_bwd, 3 = gelu_bwd) with
/// dout = 1, i.e. returns g'(x) directly.
fn dgelu(gpu: &Gpu, k: usize, x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let xb = gpu.storage_init("x", x);
    let dout = gpu.storage_init("dout", &vec![1.0f32; n]);
    let dx = gpu.storage(n as u64);
    let st = gpu.step(k, &[&xb, &dout, &dx], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&dx, n)
}

/// Reference exact GELU in f64, independent of the kernel's A&S erf polynomial.
fn gelu_erf_ref(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_ref(x / std::f64::consts::SQRT_2))
}

/// High-accuracy erf via its series / continued fraction, good to ~1e-12 over the
/// range we test. Deliberately NOT the A&S 7.1.26 polynomial the kernel uses, so
/// the forward test checks the kernel against real erf rather than against itself.
fn erf_ref(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x < 3.0 {
        // Maclaurin: erf(x) = 2/sqrt(pi) * sum_{n>=0} (-1)^n x^(2n+1) / (n! (2n+1))
        let mut term = x;
        let mut sum = x;
        for n in 1..200 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        s * 2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        s * (1.0 - 1e-12)
    }
}

#[test]
fn gelu_erf_forward_matches_reference_erf() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x: Vec<f32> = vec![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let got = fwd(&gpu, 0, &x);
    for (i, &xi) in x.iter().enumerate() {
        let want = gelu_erf_ref(xi as f64) as f32;
        assert!(
            (got[i] - want).abs() < 1e-5,
            "gelu_erf({xi}): got {}, want {want}",
            got[i]
        );
    }
}

/// The gate: analytic `gelu_erf_bwd` vs central differences of `gelu_erf`.
#[test]
fn gelu_erf_bwd_matches_finite_differences() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut st = 12345u64;
    let x: Vec<f32> = (0..64).map(|_| lcg(&mut st)).collect();

    let analytic = dgelu(&gpu, 1, &x);

    // Central difference of the *kernel's own* forward, so the A&S erf
    // approximation is common-mode and cancels.
    let h = 5e-3f32;
    let xp: Vec<f32> = x.iter().map(|v| v + h).collect();
    let xm: Vec<f32> = x.iter().map(|v| v - h).collect();
    let fp = fwd(&gpu, 0, &xp);
    let fm = fwd(&gpu, 0, &xm);

    for i in 0..x.len() {
        let fd = (fp[i] - fm[i]) / (2.0 * h);
        let a = analytic[i];
        let abs_err = (a - fd).abs();
        let rel_err = abs_err / fd.abs().max(1e-3);
        assert!(
            abs_err < 4e-3 || rel_err < 8e-2,
            "gelu_erf_bwd at x={}: analytic {a}, fd {fd} (abs {abs_err}, rel {rel_err})",
            x[i]
        );
    }
}

/// THE TRAP, pinned and MEASURED: `gelu_bwd` (tanh) is not the derivative of
/// `gelu_erf`, and gradcheck is structurally incapable of noticing.
///
/// The measured worst-case disagreement over x in [-4, 4] is ~8.7e-4 — which is
/// below gradcheck's ATOL (4e-3) *on its own*. Since the gate is
/// `abs_err <= atol + rtol*max(|a|,|n|)` (`Check::within`, src/lib.rs:41) and the
/// rtol term only ever adds slack, the mispairing satisfies it at EVERY point,
/// for ANY rtol. It is not "usually missed"; it cannot be caught.
///
/// (The relative error does reach ~25% near x ~ -2 where gelu' crosses zero, which
/// looks alarming and is irrelevant: rtol is scaled by max(|a|,|n|), so a large
/// relative error against a vanishing derivative produces a vanishing abs_err.
/// This is why the assertion is written against gradcheck's real predicate rather
/// than against a bare relative error.)
///
/// And gradcheck is even blunter than elementwise: `directional_check` contracts a
/// whole tensor to one scalar ⟨r, dL/dW⟩, so per-element errors partly cancel
/// before it ever looks. Hence this kernel exists — correctness here comes from
/// pairing the right derivative, never from the gate.
#[test]
fn gelu_bwd_is_not_gelu_erfs_derivative_and_gradcheck_cannot_tell() {
    const ATOL: f32 = 4e-3;
    const RTOL: f32 = 8e-2;
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x: Vec<f32> = (-40..=40).map(|i| i as f32 / 10.0).collect(); // [-4, 4] step 0.1

    let d_erf = dgelu(&gpu, 1, &x); // correct derivative of gelu_erf
    let d_tanh = dgelu(&gpu, 3, &x); // derivative of the tanh approximation

    let max_abs = d_erf.iter().zip(&d_tanh).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    // (1) The kernels are NOT interchangeable. If someone "simplifies" by deleting
    // gelu_erf_bwd and reusing gelu_bwd, this fires.
    assert!(
        max_abs > 1e-4,
        "gelu_bwd and gelu_erf_bwd agree to {max_abs}; if the tanh and erf \
         derivatives were truly identical, gelu_erf_bwd would be redundant"
    );

    // (2) ...and gradcheck's own predicate accepts the wrong one everywhere.
    let caught: Vec<f32> = (0..x.len())
        .filter(|&i| {
            let (a, n) = (d_erf[i], d_tanh[i]);
            (a - n).abs() > ATOL + RTOL * a.abs().max(n.abs())
        })
        .map(|i| x[i])
        .collect();
    eprintln!(
        "gelu_bwd-as-gelu_erf_bwd: max abs diff {max_abs:.3e} (< atol {ATOL}); \
         gradcheck within(atol={ATOL}, rtol={RTOL}) rejects it at {}/{} points",
        caught.len(),
        x.len()
    );
    assert!(
        caught.is_empty(),
        "the mispairing was rejected at x={caught:?} — if this ever fires the trap \
         has become visible to the gate, which would be good news worth knowing"
    );
    assert!(
        max_abs < ATOL,
        "the trap depends on max abs diff ({max_abs:.3e}) sitting under ATOL ({ATOL})"
    );
}
