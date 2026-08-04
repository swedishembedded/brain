// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Exact (erf) GELU kernel vs ground-truth torch F.gelu values. Confirms the
//! A&S erf approximation is accurate enough for parity and distinct from the
//! tanh approximation brain's `gelu` uses.
use gpu_core::Gpu;

#[test]
fn gelu_erf_matches_torch_values() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let gpu = Gpu::new_cpu(&[("gelu_erf", kernels::GELU_ERF)]);
    // x, F.gelu(x) from torch (exact erf).
    let cases: [(f32, f32); 9] = [
        (0.0, 0.0),
        (1.0, 0.8413447),
        (-1.0, -0.1586553),
        (2.0, 1.9544997),
        (-2.0, -0.0455003),
        (3.0, 2.9959502),
        (-0.5, -0.1542859),
        (0.5, 0.3457141),
        (-3.0, -0.0040498),
    ];
    let x: Vec<f32> = cases.iter().map(|c| c.0).collect();
    let xb = gpu.storage_init("x", &x);
    let yb = gpu.storage(x.len() as u64);
    gpu.submit(&[], &[gpu.step(0, &[&xb, &yb], &[x.len() as u32], x.len() as u32)]);
    let y = gpu.read(&yb, x.len());
    for (i, (xi, want)) in cases.iter().enumerate() {
        assert!((y[i] - want).abs() < 1e-4, "gelu_erf({xi}) = {} want {want}", y[i]);
    }
    // distinct from the tanh approximation at a point where they differ.
    let tanh_gelu = |v: f32| 0.5*v*(1.0 + (0.797_884_6_f32*(v + 0.044715*v*v*v)).tanh());
    assert!((y[3] - tanh_gelu(2.0)).abs() > 1e-5, "erf and tanh gelu should differ");
}
