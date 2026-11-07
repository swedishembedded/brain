// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The learned scalar lerp (`value_lerp`) and its backward (`value_lerp_bwd`, `value_lerp_dlam`), and the clamp
//! backward (`clamp_range_bwd`): the device-side pieces a decoder with a value residual and a clipped feed-forward
//! trains through.
//!
//! Swedish Embedded AB implements training stacks that differentiate through learned residual mixing and clipped
//! activations for its clients. If your team needs expertise in gradient-correct custom operators then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! The forward is compared with the host formula, and every gradient with a central finite difference of that
//! formula in f64, so the gate does not share an implementation with the kernels. The clamp backward is exact:
//! it is checked on values inside, outside and exactly on the bounds.

use data::rng::Lcg;
use gpu_core::f;

const KERNELS: &[(&str, &str)] = &[
    ("value_lerp", kernels::VALUE_LERP),
    ("value_lerp_bwd", kernels::VALUE_LERP_BWD),
    ("value_lerp_dlam", kernels::VALUE_LERP_DLAM),
    ("bias_grad", kernels::BIAS_GRAD),
    ("clamp_range", kernels::CLAMP_RANGE),
    ("clamp_range_bwd", kernels::CLAMP_RANGE_BWD),
];
const K_LERP: usize = 0;
const K_LERP_BWD: usize = 1;
const K_DLAM: usize = 2;
const K_BIAS_GRAD: usize = 3;
const K_CLAMP: usize = 4;
const K_CLAMP_BWD: usize = 5;

fn lerp64(a: f32, b: f32, lam: f32) -> f64 {
    let s = 1.0 / (1.0 + (-f64::from(lam)).exp());
    f64::from(a) + s * (f64::from(b) - f64::from(a))
}

#[test]
fn the_lerp_forward_and_all_three_gradients_match_finite_differences() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (rows, cols) = (5usize, 7usize);
    let n = rows * cols;
    let mut rng = Lcg::new(11);
    let a: Vec<f32> = (0..n).map(|_| rng.signed()).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.signed()).collect();
    let dy: Vec<f32> = (0..n).map(|_| rng.signed()).collect();
    let lam = 0.7f32;

    let (ab, bb, dyb, lamb) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(1));
    gpu.write_f32(&ab, &a);
    gpu.write_f32(&bb, &b);
    gpu.write_f32(&dyb, &dy);
    gpu.write_f32(&lamb, &[lam]);
    let out = gpu.storage(n as u64);
    let (da, db) = (gpu.storage(n as u64), gpu.storage(n as u64));
    let (part, dlam) = (gpu.storage(cols as u64), gpu.storage(1));
    // The operand gradients accumulate: start them from a known nonzero value to prove it.
    gpu.write_f32(&da, &vec![0.25; n]);
    gpu.write_f32(&db, &vec![-0.5; n]);
    gpu.write_f32(&dlam, &[0.125]);
    gpu.submit(
        &[],
        &[
            gpu.step(K_LERP, &[&ab, &bb, &lamb, &out], &[n as u32], n as u32),
            gpu.step(K_LERP_BWD, &[&dyb, &lamb, &da, &db], &[n as u32], n as u32),
            gpu.step(K_DLAM, &[&dyb, &ab, &bb, &lamb, &part], &[rows as u32, cols as u32], cols as u32),
            gpu.step(K_BIAS_GRAD, &[&part, &dlam], &[cols as u32, 1], 1),
        ],
    );

    let got = gpu.read(&out, n);
    for i in 0..n {
        assert!((f64::from(got[i]) - lerp64(a[i], b[i], lam)).abs() < 1e-6, "forward {i}: {} vs {}", got[i], lerp64(a[i], b[i], lam));
    }
    let eps = 1e-3f32;
    let (got_da, got_db) = (gpu.read(&da, n), gpu.read(&db, n));
    for i in 0..n {
        let fd_a = (lerp64(a[i] + eps, b[i], lam) - lerp64(a[i] - eps, b[i], lam)) / (2.0 * f64::from(eps)) * f64::from(dy[i]);
        let fd_b = (lerp64(a[i], b[i] + eps, lam) - lerp64(a[i], b[i] - eps, lam)) / (2.0 * f64::from(eps)) * f64::from(dy[i]);
        assert!((f64::from(got_da[i]) - 0.25 - fd_a).abs() < 1e-5, "da {i}: {} vs {fd_a}", got_da[i] - 0.25);
        assert!((f64::from(got_db[i]) + 0.5 - fd_b).abs() < 1e-5, "db {i}: {} vs {fd_b}", got_db[i] + 0.5);
    }
    let fd_lam: f64 = (0..n).map(|i| (lerp64(a[i], b[i], lam + eps) - lerp64(a[i], b[i], lam - eps)) / (2.0 * f64::from(eps)) * f64::from(dy[i])).sum();
    let got_lam = f64::from(gpu.read(&dlam, 1)[0]) - 0.125;
    assert!((got_lam - fd_lam).abs() < 1e-4 * fd_lam.abs().max(1.0), "lam gradient {got_lam} vs {fd_lam}");
}

#[test]
fn the_clamp_backward_passes_inside_the_bounds_and_blocks_where_it_saturated() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let input: [f32; 9] = [-30.0, -20.0, -19.5, 0.0, 19.5, 20.0, 20.5, 1.0e30, f32::MIN];
    let dy: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let n = input.len() as u32;
    let run = |lo: f32, hi: f32| -> Vec<f32> {
        let (x, g, dx) = (gpu.storage(u64::from(n)), gpu.storage(u64::from(n)), gpu.storage(u64::from(n)));
        gpu.write_f32(&x, &input);
        gpu.write_f32(&g, &dy);
        gpu.write_f32(&dx, &[99.0; 9]);
        gpu.submit(&[], &[gpu.step(K_CLAMP_BWD, &[&x, &g, &dx], &[n, f(lo), f(hi)], n)]);
        gpu.read(&dx, n as usize)
    };
    // Two-sided: the bounds themselves pass (torch.clamp's convention), everything beyond is exactly zero.
    assert_eq!(run(-20.0, 20.0), [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]);
    // One-sided cap, the inert lower bound written as -f32::MAX: only the saturated side blocks.
    assert_eq!(run(-f32::MAX, 20.0), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    // The forward the backward differentiates is `clamp_range`: it is zero-gradient exactly where it is constant.
    let (x, y) = (gpu.storage(u64::from(n)), gpu.storage(u64::from(n)));
    gpu.write_f32(&x, &input);
    gpu.submit(&[], &[gpu.step(K_CLAMP, &[&x, &y], &[n, f(-20.0), f(20.0)], n)]);
    assert_eq!(gpu.read(&y, n as usize)[0], -20.0);
}
