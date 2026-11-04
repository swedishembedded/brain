// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the Snake activation WGSL kernels, run on the CPU
//! (Cranelift JIT) backend so it needs no GPU:
//!   1. forward parity: GPU `snake1d` == the CPU reference oracle;
//!   2. input grad: `snake1d_bwd_dx` == central finite differences of a
//!      scalar loss `L = <y, dy>` w.r.t. `x`;
//!   3. alpha grad: `snake1d_bwd_dalpha` == finite differences w.r.t. `alpha`.

use audio::snake::{snake1d_bwd_dalpha, snake1d_bwd_dx, snake1d_fwd, snake1d_ref, Snake1d, SnakeKernels};
use data::rng::Lcg;
use gpu_core::{BufUsage, DeviceBuffer, Gpu};

const PIPES: &[(&str, &str)] =
    &[("snake1d", kernels::SNAKE1D), ("snake1d_bwd_dx", kernels::SNAKE1D_BWD_DX), ("snake1d_bwd_dalpha", kernels::SNAKE1D_BWD_DALPHA)];
const K: SnakeKernels = SnakeKernels { fwd: 0, bwd_dx: 1, bwd_dalpha: 2 };

fn buf(g: &Gpu, data: &[f32]) -> DeviceBuffer {
    let b = g.buffer("b", (data.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    g.write(&b, bytemuck::cast_slice(data));
    b
}
fn zeros(g: &Gpu, n: usize) -> DeviceBuffer {
    let b = g.buffer("z", (n * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST | BufUsage::COPY_SRC);
    g.write(&b, &vec![0u32; n]);
    b
}

fn check(c: Snake1d, seed: u64) {
    let g = Gpu::new_cpu(PIPES);
    let mut r = Lcg::new(seed);
    let n = c.total() as usize;
    let x = r.vec(n);
    // Alpha strictly positive and away from 0 (the reference DAC checkpoint's
    // trained alphas are always > 0; near-zero alpha makes `1/(alpha+eps)`
    // finite-difference-unstable, not a kernel defect).
    let alpha: Vec<f32> = (0..c.c as usize).map(|i| 0.3 + 0.1 * (i as f32 + r.unit())).collect();
    let dy = r.vec(n); // random cotangent

    let xb = buf(&g, &x);
    let ab = buf(&g, &alpha);
    let yb = zeros(&g, n);
    g.submit(&[], &[snake1d_fwd(&g, &K, &c, &xb, &ab, &yb)]);
    let y_gpu = g.read(&yb, n);
    let y_ref = snake1d_ref(&c, &x, &alpha);
    let max_abs = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    assert!(max_abs(&y_gpu, &y_ref) < 1e-4, "fwd mismatch: {}", max_abs(&y_gpu, &y_ref));

    let dyb = buf(&g, &dy);
    let dxb = zeros(&g, n);
    let dab = zeros(&g, c.c as usize);
    g.submit(&[], &[snake1d_bwd_dx(&g, &K, &c, &dyb, &xb, &ab, &dxb)]);
    g.submit(&[], &[snake1d_bwd_dalpha(&g, &K, &c, &dyb, &xb, &ab, &dab)]);
    let dx_a = g.read(&dxb, n);
    let da_a = g.read(&dab, c.c as usize);

    let loss = |x: &[f32], alpha: &[f32]| -> f32 { snake1d_ref(&c, x, alpha).iter().zip(&dy).map(|(a, b)| a * b).sum() };
    let eps = 1e-3f32;
    let fd = |base: &[f32], i: usize, f: &dyn Fn(&[f32]) -> f32| {
        let mut p = base.to_vec();
        p[i] = base[i] + eps;
        let lp = f(&p);
        p[i] = base[i] - eps;
        let lm = f(&p);
        (lp - lm) / (2.0 * eps)
    };
    for i in (0..n).step_by((n / 17).max(1)) {
        let num = fd(&x, i, &|xx| loss(xx, &alpha));
        assert!((num - dx_a[i]).abs() < 1e-2 + 1e-2 * num.abs().max(dx_a[i].abs()), "dx[{i}] num={num} ana={}", dx_a[i]);
    }
    for (i, &da) in da_a.iter().enumerate() {
        let num = fd(&alpha, i, &|aa| loss(&x, aa));
        assert!((num - da).abs() < 1e-2 + 1e-2 * num.abs().max(da.abs()), "dalpha[{i}] num={num} ana={da}");
    }
}

#[test]
fn snake1d_ncl_small() {
    check(Snake1d { rows: 2, c: 3, inner: 5, eps: 1e-9 }, 1);
}
#[test]
fn snake1d_single_row() {
    check(Snake1d { rows: 1, c: 4, inner: 1, eps: 1e-9 }, 2);
}
#[test]
fn snake1d_wide() {
    check(Snake1d { rows: 3, c: 8, inner: 11, eps: 1e-9 }, 3);
}
