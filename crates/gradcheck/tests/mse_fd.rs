// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated finite-difference gradient check for the MSE loss kernel family
//! (ADR 0001 §6, PR-10): `mse_value` (loss) + `mse_grad` (its gradient).
//!
//! These tests do NOT build any model: they drive the WGSL kernels directly via
//! `gpu_core`, exactly as the autoencoder `Regression` head will. The forward
//! kernel computes per-element  out[i] = (pred[i]-target[i])^2 / n  whose host
//! sum is the mean squared error; the backward kernel must produce its exact
//! gradient  d_pred[i] = 2*(pred[i]-target[i])/n. We FD-check every analytic
//! `d_pred` entry against the central difference of the summed forward loss.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use data::rng::Lcg;
use gpu_core::Gpu;

// Kernel order passed to Gpu::new; indices below reference these.
static KERNELS: &[(&str, &str)] = &[
    ("mse_value", kernels::MSE_VALUE), // 0
    ("mse_grad", kernels::MSE_GRAD),   // 1
];

/// Run `mse_value` for a given prediction vector; return the summed loss.
fn loss(gpu: &Gpu, pred: &[f32], target: &[f32]) -> f32 {
    let n = pred.len();
    let pred_buf = gpu.storage_init("pred", pred);
    let tgt_buf = gpu.storage_init("target", target);
    let out = gpu.storage(n as u64);
    let st = gpu.step(0, &[&pred_buf, &tgt_buf, &out], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&out, n).iter().sum()
}

/// Run `mse_grad`; return analytic d_pred.
fn grad(gpu: &Gpu, pred: &[f32], target: &[f32]) -> Vec<f32> {
    let n = pred.len();
    let pred_buf = gpu.storage_init("pred", pred);
    let tgt_buf = gpu.storage_init("target", target);
    let d_pred = gpu.storage(n as u64);
    let st = gpu.step(1, &[&pred_buf, &tgt_buf, &d_pred], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&d_pred, n)
}

#[test]
fn mse_value_matches_reference() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let pred = [1.0f32, 2.0, 3.0, -4.0];
    let target = [1.5f32, 0.0, 3.0, -2.0];
    // reference mean squared error
    let n = pred.len() as f32;
    let want: f32 = pred.iter().zip(target.iter()).map(|(&p, &t)| (p - t) * (p - t)).sum::<f32>() / n;
    let got = loss(&gpu, &pred, &target);
    assert!((got - want).abs() < 1e-6, "mse_value {got} != {want}");
}

#[test]
fn mse_grad_matches_finite_differences() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let mut st = Lcg::new(0x5EED_C0DE);
    let n = 37usize; // not a multiple of 64 -> exercises the bounds check
    let pred: Vec<f32> = (0..n).map(|_| st.signed()).collect();
    let target: Vec<f32> = (0..n).map(|_| st.signed()).collect();

    let analytic = grad(&gpu, &pred, &target);

    let eps = 1e-3f32;
    let mut max_err = 0f32;
    for i in 0..n {
        let mut wp = pred.clone();
        wp[i] += eps;
        let lp = loss(&gpu, &wp, &target);
        let mut wm = pred.clone();
        wm[i] -= eps;
        let lm = loss(&gpu, &wm, &target);
        let num = (lp - lm) / (2.0 * eps);
        let abs_err = (num - analytic[i]).abs();
        max_err = max_err.max(abs_err);
    }
    println!("mse_grad FD max abs err: {max_err:.3e}");
    let tol = 1e-3f32;
    assert!(max_err < tol, "mse_grad FD err {max_err} >= {tol}");
}
