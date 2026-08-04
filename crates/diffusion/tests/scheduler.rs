// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity tests for the flow-matching Euler scheduler.
//!
//! Goldens are closed-form values transcribed from diffusers'
//! `FlowMatchEulerDiscreteScheduler` (static shift) and independently verified
//! by hand for Z-Image's config (`num_train_timesteps=1000`, `shift=3.0`,
//! `sigmas=linspace(1, 1/N, N)`). No torch/diffusers needed: the schedule is
//! deterministic closed-form math.

use diffusion::{default_z_image_sigmas, FlowMatchConfig, FlowMatchEulerScheduler};

fn assert_close(a: &[f32], b: &[f32], eps: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length {} != {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= eps, "{what}[{i}]: {x} != {y} (eps {eps})");
    }
}

#[test]
fn z_image_default_sigmas_linspace() {
    // linspace(1.0, 1/8, 8) inclusive of both endpoints.
    let s = default_z_image_sigmas(8);
    assert_close(
        &s,
        &[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        1e-6,
        "default_z_image_sigmas(8)",
    );
    assert_eq!(default_z_image_sigmas(1), vec![1.0]);
}

#[test]
fn z_image_8step_schedule_matches_diffusers() {
    // shift σ' = 3σ/(1+2σ) applied to linspace(1, 1/8, 8), terminal 0 appended.
    let cfg = FlowMatchConfig { num_train_timesteps: 1000, shift: 3.0 };
    let mut sched = FlowMatchEulerScheduler::new(cfg);
    sched.set_timesteps(&default_z_image_sigmas(8));

    let want_sigmas = [
        1.0, 0.954_545_5, 0.9, 0.833_333_3, 0.75, 0.642_857_1, 0.5, 0.3, 0.0,
    ];
    assert_close(sched.sigmas(), &want_sigmas, 1e-5, "sigmas");

    let want_timesteps = [
        1000.0, 954.545_5, 900.0, 833.333_3, 750.0, 642.857_1, 500.0, 300.0,
    ];
    assert_close(sched.timesteps(), &want_timesteps, 1e-2, "timesteps");
}

#[test]
fn euler_step_is_first_order_increment() {
    let cfg = FlowMatchConfig { num_train_timesteps: 1000, shift: 3.0 };
    let mut sched = FlowMatchEulerScheduler::new(cfg);
    sched.set_timesteps(&default_z_image_sigmas(8));

    // One step: x_next = x + (σ_1 - σ_0)·v.
    let x = vec![2.0f32, -1.0, 0.5];
    let v = vec![1.0f32, 1.0, -2.0];
    let dt = sched.sigmas()[1] - sched.sigmas()[0]; // 0.9545455 - 1.0
    let got = sched.step(&v, &x);
    let want: Vec<f32> = x.iter().zip(&v).map(|(xi, vi)| xi + dt * vi).collect();
    assert_close(&got, &want, 1e-6, "single euler step");
}

#[test]
fn constant_velocity_integrates_exactly_to_minus_v() {
    // Euler with a constant velocity telescopes: Σ(σ_{i+1}-σ_i) = σ_N - σ_0 =
    // 0 - 1 = -1, so integrating v across all steps yields x0 - v exactly.
    let cfg = FlowMatchConfig { num_train_timesteps: 1000, shift: 3.0 };
    let mut sched = FlowMatchEulerScheduler::new(cfg);
    sched.set_timesteps(&default_z_image_sigmas(8));

    let v = vec![0.3f32, -0.7, 1.5, 0.0];
    let mut x = vec![10.0f32, 10.0, 10.0, 10.0];
    for _ in 0..sched.timesteps().len() {
        x = sched.step(&v, &x);
    }
    let want: Vec<f32> = [10.0f32, 10.0, 10.0, 10.0]
        .iter()
        .zip(&v)
        .map(|(xi, vi)| xi - vi)
        .collect();
    assert_close(&x, &want, 1e-5, "full constant-velocity integration");
}
