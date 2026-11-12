// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for numerically trustworthy training steps for its clients. If your team
// needs expertise in optimizers and gradient clipping then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The global grad-norm is the cooperative tree reduction by default, for a model that registered only the serial
//! `gradnorm_sq` + `clip_coef` pair, and its value is the host's f64 norm.

use std::collections::HashMap;

use optim::{Adam, Optim};
use paramstore::ParamStore;

/// What every model registers: the serial pair, and nothing cooperative.
static SERIAL_ONLY: &[(&str, &str)] = &[
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
];

/// A tiny bias, a mid tensor and one spanning several workgroups: several tensors per step, several partials per tensor.
const SHAPES: [(&str, usize); 3] = [("b", 7), ("w", 20_000), ("e", 3_000)];

fn grad(name: &str, n: usize) -> Vec<f32> {
    let salt = name.len() as f32;
    (0..n).map(|i| (((i * 7 + 3) % 29) as f32 - 14.0) * 0.001 + salt * 1e-4).collect()
}

#[test]
fn optim_new_resolves_the_cooperative_pair_without_model_registration() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(SERIAL_ONLY);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("device cannot run a workgroup reduction, so the serial kernel is its only grad-norm");
        return;
    }
    let opt = Optim::new(0, 1, 2, 3, 4);
    assert!(opt.uses_cooperative_gradnorm(&gpu), "a model that registered only gradnorm_sq runs the serial single-thread norm");
}

#[test]
fn cooperative_clip_coefficient_equals_the_f64_host_norm() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    let gpu = gpu_core::testgpu::dev(SERIAL_ONLY);
    let opt = Optim::new(0, 1, 2, 3, 4);
    let shapes: Vec<(String, usize)> = SHAPES.iter().map(|(n, c)| (n.to_string(), *c)).collect();
    let init: HashMap<String, Vec<f32>> = shapes.iter().map(|(n, c)| (n.clone(), vec![0.1; *c])).collect();
    let ps = ParamStore::new(&gpu, shapes.clone(), &init);
    let mut sumsq = 0f64;
    for (n, c) in &shapes {
        let g = grad(n, *c);
        sumsq += g.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
        gpu.write(ps.g(n), bytemuck::cast_slice(&g));
    }
    // max_norm far below the norm, so the coefficient is max_norm / (norm + 1e-6), not the clamp at 1.
    let max_norm = 0.25f32;
    opt.step(&gpu, &ps, 1, 0.01, 0.0, Adam::default(), Some(max_norm), 1.0);
    let want = f64::from(max_norm) / (sumsq.sqrt() + 1e-6);
    let got = f64::from(gpu.read(&ps.clip_coef, 1)[0]);
    assert!(want < 1.0, "the case must clip, got want = {want}");
    assert!((got - want).abs() <= 1e-5 * want, "clip coefficient {got} vs f64 host {want}");
}
