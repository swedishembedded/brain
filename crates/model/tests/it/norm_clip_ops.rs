// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the gamma-free RMSNorm (`kernels::RMSNORM_NOGAIN*`) and the scaled L2 clip
//! (`kernels::L2CLIP_SCALE*`): forwards equal the host references; every backward (dx, and the scale gradient) equals
//! the finite-difference derivative of an f64 oracle of the forward, on vectors both below and above the clip
//! threshold and with the scale on and off.
//!
//! Swedish Embedded AB implements numerically bounded attention and normalisation layers for its clients. If your
//! team needs expertise in stable transformer training then you can procure our services by sending an email to
//! info@swedishembedded.com.

use data::rng::{random_seed, Lcg};
use model::hostmath::{l2clip_scale_rows, rmsnorm_nogain_rows};

const PIPES: &[(&str, &str)] = &[
    ("rmsnorm_nogain", kernels::RMSNORM_NOGAIN),
    ("rmsnorm_nogain_dx", kernels::RMSNORM_NOGAIN_DX),
    ("l2clip_scale", kernels::L2CLIP_SCALE),
    ("l2clip_scale_dx", kernels::L2CLIP_SCALE_DX),
    ("l2clip_scale_ds", kernels::L2CLIP_SCALE_DS),
];

const EPS: f32 = 1e-6;

/// The f64 oracle of the scaled clip, `sum(dy * clip(x * s))`.
fn clip_loss(x: &[f64], s: Option<&[f64]>, dy: &[f64], heads: usize, d: usize, thr: f64) -> f64 {
    let mut total = 0.0;
    for (v, row) in x.chunks_exact(d).enumerate() {
        let a: Vec<f64> = (0..d).map(|c| row[c] * s.map_or(1.0, |s| s[(v % heads) * d + c])).collect();
        let norm = a.iter().map(|t| t * t).sum::<f64>().sqrt();
        let k = (thr / (norm + f64::from(EPS))).min(1.0);
        total += (0..d).map(|c| dy[v * d + c] * a[c] * k).sum::<f64>();
    }
    total
}

fn rms_loss(x: &[f64], dy: &[f64], d: usize) -> f64 {
    let mut total = 0.0;
    for (row, g) in x.chunks_exact(d).zip(dy.chunks_exact(d)) {
        let r = 1.0 / (row.iter().map(|t| t * t).sum::<f64>() / d as f64 + f64::from(EPS)).sqrt();
        total += row.iter().zip(g).map(|(a, b)| a * r * b).sum::<f64>();
    }
    total
}

fn fd(loss: &dyn Fn(&[f64]) -> f64, at: &[f64], i: usize) -> f64 {
    let h = 1e-5;
    let mut p = at.to_vec();
    p[i] += h;
    let up = loss(&p);
    p[i] -= 2.0 * h;
    (up - loss(&p)) / (2.0 * h)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&t| f64::from(t)).collect()
}

fn close(got: f32, want: f64, what: &str, i: usize) {
    assert!((f64::from(got) - want).abs() <= 2e-4 * (1.0 + want.abs()), "{what}[{i}]: kernel {got} vs finite difference {want}");
}

#[test]
fn rmsnorm_nogain_forward_and_dx() {
    let g = gpu_core::testgpu::dev(PIPES);
    let mut rng = Lcg::new(random_seed());
    // d is not a multiple of 64 and exceeds it, rows is not a multiple of anything: the strided loop and the row guard.
    let (rows, d) = (5usize, 150usize);
    let x = rng.vec_scaled(rows * d, 2.0);
    let dy = rng.vec_scaled(rows * d, 1.0);
    let (xb, dyb) = (g.storage_init("x", &x), g.storage_init("dy", &dy));
    let (yb, dxb) = (g.storage((rows * d) as u64), g.storage((rows * d) as u64));
    let params = [d as u32, rows as u32, EPS.to_bits()];
    let groups = gpu_core::Dispatch::Workgroups(rows as u32);
    let fwd = g.kernel_index("rmsnorm_nogain").expect("registered");
    let dxk = g.kernel_index("rmsnorm_nogain_dx").expect("registered");
    g.submit(&[], &[g.dispatch(fwd, &[&xb, &yb], &params, groups), g.dispatch(dxk, &[&xb, &dyb, &dxb], &params, groups)]);
    let (y, dx) = (g.read(&yb, rows * d), g.read(&dxb, rows * d));
    let want = rmsnorm_nogain_rows(&x, rows, d, EPS);
    for i in 0..x.len() {
        assert!((y[i] - want[i]).abs() < 1e-5, "y[{i}] {} vs {}", y[i], want[i]);
    }
    let (x64, dy64) = (f64s(&x), f64s(&dy));
    for i in (0..x.len()).step_by(7) {
        close(dx[i], fd(&|p| rms_loss(p, &dy64, d), &x64, i), "dx", i);
    }
}

#[test]
fn l2clip_scale_forward_dx_and_ds() {
    let g = gpu_core::testgpu::dev(PIPES);
    let mut rng = Lcg::new(random_seed());
    let (heads, d, tokens, thr) = (3usize, 20usize, 7usize, 4.0f32);
    let n_vec = tokens * heads;
    let fwd = g.kernel_index("l2clip_scale").expect("registered");
    let dxk = g.kernel_index("l2clip_scale_dx").expect("registered");
    let dsk = g.kernel_index("l2clip_scale_ds").expect("registered");
    // Vectors 3 times longer than the threshold alternate with vectors well below it, so both branches are exercised.
    let mut x = rng.vec_scaled(n_vec * d, 1.0);
    for (v, row) in x.chunks_exact_mut(d).enumerate() {
        let gain = if v % 2 == 0 { 3.0 } else { 0.2 };
        row.iter_mut().for_each(|t| *t *= gain);
    }
    let scale: Vec<f32> = rng.vec_scaled(heads * d, 0.5).iter().map(|t| t + 1.0).collect();
    let dy = rng.vec_scaled(n_vec * d, 1.0);
    let (x64, s64, dy64) = (f64s(&x), f64s(&scale), f64s(&dy));
    for with_scale in [true, false] {
        let params = [n_vec as u32, heads as u32, d as u32, u32::from(with_scale), thr.to_bits(), EPS.to_bits()];
        let groups = gpu_core::Dispatch::Workgroups(n_vec as u32);
        let (xb, sb, dyb) = (g.storage_init("x", &x), g.storage_init("s", &scale), g.storage_init("dy", &dy));
        let (yb, dxb, gsb) = (g.storage((n_vec * d) as u64), g.storage((n_vec * d) as u64), g.storage((n_vec * d) as u64));
        let dsb = g.storage_init("ds", &vec![0f32; heads * d]);
        let dparams = [n_vec as u32, heads as u32, d as u32];
        g.submit(&[], &[
            g.dispatch(fwd, &[&xb, &sb, &yb], &params, groups),
            g.dispatch(dxk, &[&xb, &sb, &dyb, &dxb, &gsb], &params, groups),
            g.step(dsk, &[&gsb, &dsb], &dparams, (heads * d) as u32),
        ]);
        let (y, dx, ds) = (g.read(&yb, n_vec * d), g.read(&dxb, n_vec * d), g.read(&dsb, heads * d));
        let s_arg = with_scale.then_some(&scale[..]);
        let want = l2clip_scale_rows(&x, s_arg, heads, d, thr, EPS);
        for i in 0..y.len() {
            assert!((y[i] - want[i]).abs() < 1e-5, "scale {with_scale}: y[{i}] {} vs {}", y[i], want[i]);
        }
        let s_oracle = with_scale.then_some(&s64[..]);
        let loss_x = |p: &[f64]| clip_loss(p, s_oracle, &dy64, heads, d, f64::from(thr));
        for i in 0..x.len() {
            close(dx[i], fd(&loss_x, &x64, i), if with_scale { "dx(scaled)" } else { "dx" }, i);
        }
        if with_scale {
            let loss_s = |p: &[f64]| clip_loss(&x64, Some(p), &dy64, heads, d, f64::from(thr));
            for i in 0..scale.len() {
                close(ds[i], fd(&loss_s, &s64, i), "ds", i);
            }
        }
    }
}
