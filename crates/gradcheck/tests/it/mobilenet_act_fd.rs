// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Isolated checks of MobileNetV3's activation pairs - `hardswish`,
//! `hardsigmoid`, `relu6` and their `_bwd` kernels - driven directly through
//! `gpu_core` like `gelu_erf_fd.rs`, with no model built.
//!
//! Each forward is compared with an f64 reference written from the defining
//! formula, and each backward with a central finite difference OF THE FORWARD
//! KERNEL ITSELF, so the pair is checked for consistency, not two copies of
//! one formula against each other. Inputs avoid the kinks (where a
//! piecewise function's one-sided derivatives differ and no finite difference
//! has a single answer), and sweep every piece.
//!
//! The hard sigmoid is also pinned against the OTHER common spelling,
//! `clamp(0.2x + 0.5)`: the two disagree almost everywhere, and a checkpoint
//! trained with one and run with the other is off by up to 0.1 per gate.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use data::rng::Lcg;
use gpu_core::Gpu;

static KERNELS: &[(&str, &str)] = &[
    ("hardswish", kernels::HARDSWISH),             // 0
    ("hardswish_bwd", kernels::HARDSWISH_BWD),     // 1
    ("hardsigmoid", kernels::HARDSIGMOID),         // 2
    ("hardsigmoid_bwd", kernels::HARDSIGMOID_BWD), // 3
    ("relu6", kernels::RELU6),                     // 4
    ("relu6_bwd", kernels::RELU6_BWD),             // 5
];

fn fwd(gpu: &Gpu, k: usize, x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let xb = gpu.storage_init("x", x);
    let out = gpu.storage(n as u64);
    let st = gpu.step(k, &[&xb, &out], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&out, n)
}

fn bwd(gpu: &Gpu, k: usize, x: &[f32], dout: &[f32]) -> Vec<f32> {
    let n = x.len();
    let xb = gpu.storage_init("x", x);
    let db = gpu.storage_init("dout", dout);
    let dx = gpu.storage(n as u64);
    let st = gpu.step(k, &[&xb, &db, &dx], &[n as u32], n as u32);
    gpu.submit(&[], &[st]);
    gpu.poll_wait();
    gpu.read(&dx, n)
}

fn hardswish(x: f64) -> f64 {
    x * (x + 3.0).clamp(0.0, 6.0) / 6.0
}
fn hardsigmoid(x: f64) -> f64 {
    (x + 3.0).clamp(0.0, 6.0) / 6.0
}
fn relu6(x: f64) -> f64 {
    x.clamp(0.0, 6.0)
}

/// Points across every piece of the functions, each at least `margin` from
/// the kinks at -3, 0, 3 and 6.
fn sample(n: usize, margin: f32) -> Vec<f32> {
    let mut rng = Lcg::new(17);
    let kinks = [-3.0f32, 0.0, 3.0, 6.0];
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let v = rng.signed() * 9.0;
        if kinks.iter().all(|k| (v - k).abs() > margin) {
            out.push(v);
        }
    }
    out
}

fn check_pair(f_k: usize, b_k: usize, reference: fn(f64) -> f64, name: &str) {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = sample(512, 0.05);
    let y = fwd(&gpu, f_k, &x);
    for (xi, yi) in x.iter().zip(&y) {
        let want = reference(*xi as f64);
        assert!((*yi as f64 - want).abs() < 1e-6 * want.abs().max(1.0), "{name}({xi}) = {yi}, want {want}");
    }
    // backward with a random upstream gradient against the kernel's own FD
    let mut rng = Lcg::new(29);
    let dout: Vec<f32> = (0..x.len()).map(|_| rng.signed()).collect();
    let dx = bwd(&gpu, b_k, &x, &dout);
    let h = 1e-2f32;
    let plus: Vec<f32> = x.iter().map(|v| v + h).collect();
    let minus: Vec<f32> = x.iter().map(|v| v - h).collect();
    let (yp, ym) = (fwd(&gpu, f_k, &plus), fwd(&gpu, f_k, &minus));
    for i in 0..x.len() {
        let fd = (yp[i] as f64 - ym[i] as f64) / (2.0 * h as f64) * dout[i] as f64;
        assert!((dx[i] as f64 - fd).abs() < 2e-3, "{name}_bwd at {}: {} vs finite difference {fd}", x[i], dx[i]);
    }
}

#[test]
fn hardswish_pair_matches_its_formula_and_its_finite_differences() {
    check_pair(0, 1, hardswish, "hardswish");
}

#[test]
fn hardsigmoid_pair_matches_its_formula_and_its_finite_differences() {
    check_pair(2, 3, hardsigmoid, "hardsigmoid");
}

#[test]
fn relu6_pair_matches_its_formula_and_its_finite_differences() {
    check_pair(4, 5, relu6, "relu6");
}

#[test]
fn hardsigmoid_is_torchs_relu6_form_not_the_keras_slope() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let x = [-2.0f32, -1.0, 1.0, 2.0];
    let y = fwd(&gpu, 2, &x);
    let keras = |v: f32| (0.2 * v + 0.5).clamp(0.0, 1.0);
    for (xi, yi) in x.iter().zip(&y) {
        assert!((yi - keras(*xi)).abs() > 0.02, "hardsigmoid({xi}) = {yi} is the Keras curve");
    }
}
