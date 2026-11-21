// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the soft cap `y = c * tanh(x / c)` (`kernels::SOFTCAP` / `SOFTCAP_BWD`): the forward equals the host
//! reference, saturates at +-c without NaN for huge inputs, and the backward equals the finite-difference derivative
//! of the forward (f64 oracle).
//!
//! Swedish Embedded AB implements numerically bounded logit and gate heads for its clients. If your team needs
//! expertise in stable training of language models then you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::{random_seed, Lcg};
use model::hostmath::{softcap, softcap_grad};

const PIPES: &[(&str, &str)] = &[("softcap", kernels::SOFTCAP), ("softcap_bwd", kernels::SOFTCAP_BWD)];

fn run(g: &gpu_core::Gpu, x: &[f32], dy: &[f32], cap: f32) -> (Vec<f32>, Vec<f32>) {
    let fwd = g.kernel_index("softcap").expect("softcap registered");
    let bwd = g.kernel_index("softcap_bwd").expect("softcap_bwd registered");
    let n = x.len();
    let params = [n as u32, cap.to_bits()];
    let xb = g.storage_init("x", x);
    let dyb = g.storage_init("dy", dy);
    let yb = g.storage(n as u64);
    let dxb = g.storage(n as u64);
    g.submit(&[], &[g.step(fwd, &[&xb, &yb], &params, n as u32), g.step(bwd, &[&dyb, &xb, &dxb], &params, n as u32)]);
    (g.read(&yb, n), g.read(&dxb, n))
}

#[test]
fn softcap_forward_matches_reference_and_backward_matches_finite_differences() {
    let g = gpu_core::testgpu::dev(PIPES);
    let mut rng = Lcg::new(random_seed());
    for cap in [4.0f32, 25.0, 30.0] {
        let mut x = rng.vec_scaled(37, 3.0 * cap);
        x.extend([0.0, 1e-3 * cap, -cap, 100.0 * cap, -100.0 * cap]);
        let dy = rng.vec_scaled(x.len(), 1.0);
        let (y, dx) = run(&g, &x, &dy, cap);
        for i in 0..x.len() {
            let want = softcap(x[i], cap);
            assert!(y[i].is_finite() && (y[i] - want).abs() <= 1e-5 * cap, "cap {cap}: y[{i}] {} vs {want} at x {}", y[i], x[i]);
            assert!(y[i].abs() <= cap, "cap {cap}: |y| exceeds the cap at x {}", x[i]);
            let (xd, cd) = (f64::from(x[i]), f64::from(cap));
            let h = 1e-4 * cd;
            let fd = cd * ((xd + h) / cd).tanh() - cd * ((xd - h) / cd).tanh();
            let fd = fd / (2.0 * h) * f64::from(dy[i]);
            assert!((f64::from(dx[i]) - fd).abs() < 1e-4, "cap {cap}: dx[{i}] {} vs finite difference {fd} at x {}", dx[i], x[i]);
            assert!((dx[i] - dy[i] * softcap_grad(x[i], cap)).abs() < 1e-5, "cap {cap}: dx[{i}] disagrees with the host derivative");
        }
    }
}
