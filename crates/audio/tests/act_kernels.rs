// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for the ELU WGSL kernels. Uses `Gpu::new` (the ambient
//! `BRAIN_DEVICE`-selected backend, defaulting to wgpu when a GPU is present
//! and falling back to the CPU JIT otherwise) so the same test exercises both
//! backends across machines - explicitly re-run with `BRAIN_DEVICE=cpu` for a
//! GPU-free gate (per lessons.md #5: the kernel has no barriers/shared
//! memory, so this is a belt-and-braces check, not one expected to diverge):
//!   1. forward parity: GPU `elu` == the CPU reference oracle at values
//!      spanning positive/negative/zero;
//!   2. input grad: `elu_bwd` == central finite differences of a scalar loss
//!      `L = <y, dy>` w.r.t. `x`.

use audio::act::{elu_bwd, elu_fwd, elu_ref, ActKernels};
use data::rng::Lcg;
use gpu_core::{BufUsage, DeviceBuffer, Gpu};

const PIPES: &[(&str, &str)] = &[("elu", kernels::ELU), ("elu_bwd", kernels::ELU_BWD)];
const K: ActKernels = ActKernels { fwd: 0, bwd: 1 };
const ALPHA: f32 = 1.0;

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
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Values-based check at a few points spanning positive/negative/zero.
#[test]
fn elu_values_at_known_points() {
    let x = [-2.0f32, -1.0, -1e-6, 0.0, 1e-6, 0.5, 2.0];
    let got = elu_ref(&x, ALPHA);
    let want: Vec<f32> = x.iter().map(|&v| if v > 0.0 { v } else { ALPHA * (v.exp() - 1.0) }).collect();
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-6, "elu_ref[{i}]: got {g} want {w}");
    }
    // known closed forms
    assert!((got[3] - 0.0).abs() < 1e-6, "elu(0) == 0");
    assert!(got[0] < 0.0 && got[0] > -ALPHA, "elu(-2) in (-alpha, 0)");
    assert!((got[5] - 0.5).abs() < 1e-6, "elu(x) == x for x>0");
}

fn check(n: usize, seed: u64) {
    let g = Gpu::new(PIPES);
    let mut r = Lcg::new(seed);
    let x = r.vec(n);
    let dy = r.vec(n);

    let xb = buf(&g, &x);
    let yb = zeros(&g, n);
    g.submit(&[], &[elu_fwd(&g, &K, &xb, &yb, n as u32, ALPHA)]);
    let y_gpu = g.read(&yb, n);
    let y_ref = elu_ref(&x, ALPHA);
    assert!(max_abs(&y_gpu, &y_ref) < 1e-4, "fwd mismatch: {}", max_abs(&y_gpu, &y_ref));

    let dyb = buf(&g, &dy);
    let dxb = zeros(&g, n);
    g.submit(&[], &[elu_bwd(&g, &K, &xb, &dyb, &dxb, n as u32, ALPHA)]);
    let dx_a = g.read(&dxb, n);

    let loss = |xx: &[f32]| -> f32 { elu_ref(xx, ALPHA).iter().zip(&dy).map(|(a, b)| a * b).sum() };
    let eps = 1e-3f32;
    for i in (0..n).step_by((n / 17).max(1)) {
        let mut p = x.clone();
        p[i] = x[i] + eps;
        let lp = loss(&p);
        p[i] = x[i] - eps;
        let lm = loss(&p);
        let num = (lp - lm) / (2.0 * eps);
        assert!((num - dx_a[i]).abs() < 1e-2 + 1e-2 * num.abs().max(dx_a[i].abs()), "dx[{i}] num={num} ana={}", dx_a[i]);
    }
}

#[test]
fn elu_small() {
    check(23, 1);
}
#[test]
fn elu_wide() {
    check(257, 2);
}
