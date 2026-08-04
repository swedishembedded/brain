// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated finite-difference gradient check for the QuickGELU pair:
//! `quick_gelu` (forward) + `quick_gelu_bwd` (its gradient, added with the CLIP
//! backward — the forward kernel's header asked for exactly this file).
//!
//! No model is built: the kernels are driven straight through `gpu_core`, the
//! same shape as `gelu_erf_fd.rs` / `mse_fd.rs`.
//!
//! WHY THIS FILE EXISTS. brain now has three GELUs, and their derivatives are
//! three different functions that agree to ~1e-2:
//!   * `gelu` / `gelu_bwd`         — tanh approximation (GPT-2)
//!   * `gelu_erf` / `gelu_erf_bwd` — exact erf (torch's `F.gelu`)
//!   * `quick_gelu` / `quick_gelu_bwd` — x·sigmoid(1.702x) (OpenAI CLIP / CLIP-L)
//!
//! `gelu_erf_fd.rs` already pins the trap that the tanh/erf mispairing is
//! INVISIBLE to `directional_check`'s tolerance. QuickGELU is the loudest member
//! of the family (it disagrees with the other two by ~1e-2, not ~1e-3), and the
//! last test here measures exactly how much of that a gradcheck could see.
//!
//! Elementwise, no reduction, no `workgroupBarrier` — so there is no cooperative
//! twin to select on `DeviceCaps::workgroup_reductions` and nothing here is
//! device-gated. Runs on any device; also run with `BRAIN_DEVICE=cpu`.

use gpu_core::Gpu;

// Kernel order passed to Gpu::new; indices below reference these.
static KERNELS: &[(&str, &str)] = &[
    ("quick_gelu", kernels::QUICK_GELU),         // 0
    ("quick_gelu_bwd", kernels::QUICK_GELU_BWD), // 1
    ("gelu_erf", kernels::GELU_ERF),             // 2
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),     // 3
];
const K_FWD: usize = 0;
const K_BWD: usize = 1;
const K_ERF_BWD: usize = 3;

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * 3.0 // ~[-3,3)
}

/// `quick_gelu` forward. Params: a single `total`; bufs `[x, out]`.
fn fwd(gpu: &Gpu, x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let xb = gpu.storage_init("x", x);
    let out = gpu.storage(n as u64);
    let st = gpu.step(K_FWD, &[&xb, &out], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&out, n)
}

/// Elementwise activation backward with `dout = 1`, i.e. returns `g'(x)`.
/// Params: a single `total`; bufs `[x (pre-activation), dout, dx]`.
fn dact(gpu: &Gpu, k: usize, x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let xb = gpu.storage_init("x", x);
    let dout = gpu.storage_init("dout", &vec![1.0f32; n]);
    let dx = gpu.storage(n as u64);
    let st = gpu.step(k, &[&xb, &dout, &dx], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&dx, n)
}

/// Reference derivative in f64, re-derived from the definition rather than
/// shared with the kernel (an oracle that shares code with the thing it checks
/// proves nothing — AGENTS.md exception 1):
///   g(x) = x·s, s = 1/(1+e^{-1.702x});  g'(x) = s + 1.702·x·s·(1-s)
fn dquick_ref(x: f64) -> f64 {
    let s = 1.0 / (1.0 + (-1.702 * x).exp());
    s + 1.702 * x * s * (1.0 - s)
}

#[test]
fn quick_gelu_bwd_matches_the_analytic_derivative() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x: Vec<f32> = (-40..=40).map(|i| i as f32 / 10.0).collect(); // [-4, 4] step 0.1
    let got = dact(&gpu, K_BWD, &x);
    for (i, &xi) in x.iter().enumerate() {
        let want = dquick_ref(xi as f64) as f32;
        assert!(
            (got[i] - want).abs() < 1e-5,
            "quick_gelu_bwd({xi}): got {}, want {want}",
            got[i]
        );
    }
}

/// The gate: analytic `quick_gelu_bwd` vs central differences of the kernel's
/// OWN forward (so any shared fp32 detail is common-mode and cancels).
#[test]
fn quick_gelu_bwd_matches_finite_differences() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut st = 12345u64;
    let x: Vec<f32> = (0..64).map(|_| lcg(&mut st)).collect();

    let analytic = dact(&gpu, K_BWD, &x);

    let h = 5e-3f32;
    let xp: Vec<f32> = x.iter().map(|v| v + h).collect();
    let xm: Vec<f32> = x.iter().map(|v| v - h).collect();
    let fp = fwd(&gpu, &xp);
    let fm = fwd(&gpu, &xm);

    let mut worst = 0.0f32;
    for i in 0..x.len() {
        let fd = (fp[i] - fm[i]) / (2.0 * h);
        let a = analytic[i];
        let abs_err = (a - fd).abs();
        let rel_err = abs_err / fd.abs().max(1e-3);
        worst = worst.max(rel_err.min(abs_err / 4e-3));
        assert!(
            abs_err < 4e-3 || rel_err < 8e-2,
            "quick_gelu_bwd at x={}: analytic {a}, fd {fd} (abs {abs_err}, rel {rel_err})",
            x[i]
        );
    }
    eprintln!("quick_gelu_bwd vs FD over 64 points: worst normalized error {worst:.3e}");
}

/// `dout` is genuinely applied (not ignored): scaling it scales `dx` linearly.
/// A kernel that dropped the `dout` multiply would still pass the two tests
/// above, which both pass `dout = 1`.
#[test]
fn quick_gelu_bwd_applies_dout() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x: Vec<f32> = (-8..=8).map(|i| i as f32 / 4.0).collect();
    let n = x.len();
    let g1 = dact(&gpu, K_BWD, &x);

    let xb = gpu.storage_init("x", &x);
    let dvals: Vec<f32> = (0..n).map(|i| 0.5 + i as f32).collect();
    let dout = gpu.storage_init("dout", &dvals);
    let dx = gpu.storage(n as u64);
    let st = gpu.step(K_BWD, &[&xb, &dout, &dx], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    let got = gpu.read(&dx, n);
    for i in 0..n {
        let want = g1[i] * dvals[i];
        assert!((got[i] - want).abs() <= 1e-5 * want.abs().max(1.0), "dout scaling at i={i}: {} vs {want}", got[i]);
    }
}

/// Companion to `gelu_erf_fd.rs`'s trap test, for the third family member.
/// Unlike the tanh/erf pair (whose derivatives differ by ~8.7e-4, BELOW
/// gradcheck's 4e-3 ATOL and therefore structurally invisible), QuickGELU's
/// derivative differs from the erf one by enough that the gate *would* reject
/// the mispairing at some points. This test measures which regime we are in and
/// pins it, so a future "simplification" that reuses `gelu_erf_bwd` for
/// `quick_gelu` is caught here rather than by a silently wrong training run.
#[test]
fn quick_gelu_bwd_is_not_gelu_erf_bwd() {
    const ATOL: f32 = 4e-3;
    const RTOL: f32 = 8e-2;
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x: Vec<f32> = (-40..=40).map(|i| i as f32 / 10.0).collect();

    let d_quick = dact(&gpu, K_BWD, &x);
    let d_erf = dact(&gpu, K_ERF_BWD, &x);
    let max_abs = d_quick.iter().zip(&d_erf).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let caught = (0..x.len())
        .filter(|&i| {
            let (a, n) = (d_quick[i], d_erf[i]);
            (a - n).abs() > ATOL + RTOL * a.abs().max(n.abs())
        })
        .count();
    eprintln!(
        "quick_gelu_bwd vs gelu_erf_bwd: max abs diff {max_abs:.3e}; gradcheck \
         within(atol={ATOL}, rtol={RTOL}) rejects the mispairing at {caught}/{} points",
        x.len()
    );
    assert!(
        max_abs > 1e-2,
        "quick_gelu and gelu_erf derivatives agree to {max_abs}; if they were \
         truly identical, quick_gelu_bwd would be redundant"
    );
}
