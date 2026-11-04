// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The activation side of every activation-quantizing linear is scaled per
//! 32-element group, the weight side's own granularity.
//!
//! Swedish Embedded AB implements accurate quantized inference for its
//! clients. If your team needs expertise in int8 LLM serving that keeps the
//! model's own answers, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! One scale per token lets a single outlier channel set the step for the
//! whole row: LLM activations carry such channels, and every other value of
//! the row then rounds to a handful of levels or to zero. On Qwen3-0.6B that
//! cost a next-token KL of 2.8e-2 against 1.9e-3 with a scale per 32
//! elements, which is bf16's level. The cases here put one outlier in one
//! group of a row and hold every output to the per-group quantization bound.
//! The weights are exactly representable in each tier (each group is
//! `{-c, 0, c}`), so the bound is the activation's alone.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use model::ops::{Ops, Weight};

/// Activation-quantization group, in elements.
const GROUP: usize = 32;
const OUTLIER: f32 = 1000.0;

/// `[n, k]` weights whose every 32-element group is a mix of `-c`, `0` and
/// `c` with `c` the group's own, so the group-wise int8 and int4 tiers hold
/// them exactly. The first group of each row, where the activations carry
/// their outlier, is zero: what the outlier costs its own group is then
/// nothing, and the bound is the other groups' alone.
fn exact_weights(rng: &mut Lcg, n: usize, k: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(n * k);
    for i in 0..n * k / GROUP {
        if i % (k / GROUP) == 0 {
            w.extend([0.0; GROUP]);
            continue;
        }
        let c = 0.01 + (rng.next_u32() % 100) as f32 * 1e-3;
        w.push(c);
        w.extend((1..GROUP).map(|_| [-c, 0.0, c][(rng.next_u32() % 3) as usize]));
    }
    w
}

/// `[m, k]` activations near unit scale, with one outlier in the first group
/// of every row.
fn outlier_rows(rng: &mut Lcg, m: usize, k: usize) -> Vec<f32> {
    let mut x: Vec<f32> = (0..m * k).map(|_| (rng.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect();
    for r in 0..m {
        x[r * k + 3] = OUTLIER;
    }
    x
}

/// The per-group quantization bound on `y[r, j]`: half a step of each
/// activation group times the weight mass it meets, plus fp32 summation.
fn bound(x: &[f32], w: &[f32], k: usize, r: usize, j: usize) -> f64 {
    let mut b = 0.0f64;
    for g in 0..k / GROUP {
        let xs = &x[r * k + g * GROUP..r * k + (g + 1) * GROUP];
        let step = f64::from(xs.iter().fold(0.0f32, |a, v| a.max(v.abs()))) / 127.0;
        let mass: f64 = w[j * k + g * GROUP..j * k + (g + 1) * GROUP].iter().map(|v| f64::from(v.abs())).sum();
        b += 0.5 * step * mass;
    }
    let terms: f64 = (0..k).map(|i| f64::from(x[r * k + i].abs()) * f64::from(w[j * k + i].abs())).sum();
    b + terms * 1e-5
}

fn check(dtype: Dtype, m: usize, n: usize, k: usize) {
    let ops = Ops::new(gpu_core::testgpu::dev(model::ops::kernel_list())).expect("Ops::new");
    let g = ops.gpu();
    let mut rng = Lcg::new(data::rng::random_seed());
    let w = exact_weights(&mut rng, n, k);
    let x = outlier_rows(&mut rng, m, k);
    let weight = Weight::upload(&ops, &w, n, k, dtype);
    let xb = g.storage_init("x", &x);
    let y = g.storage((m * n) as u64);
    let mut steps = Vec::new();
    let act = ops.act(&mut steps, &xb, 0, m as u32, k as u32);
    ops.matmul(&mut steps, &weight, &act, &y, 0);
    g.submit(&[], &steps);
    g.poll_wait();
    let got = g.read(&y, m * n);
    for r in 0..m {
        for j in 0..n {
            let want: f64 = (0..k).map(|i| f64::from(x[r * k + i]) * f64::from(w[j * k + i])).sum();
            let err = (f64::from(got[r * n + j]) - want).abs();
            let b = bound(&x, &w, k, r, j);
            assert!(err <= b, "{dtype:?} {}: m={m} n={n} k={k} y[{r},{j}] = {} off the exact {want} by {err}, past the per-group bound {b}", ops.matmul_kernel(&weight, m as u32), got[r * n + j]);
        }
    }
}

/// Decode rows (the skinny-M GEMV regime) and prefill rows (the tiled GEMM),
/// for each tier that multiplies quantized activations.
#[test]
fn an_outlier_moves_only_its_own_group_s_step() {
    for dtype in [Dtype::I8, Dtype::Q4] {
        for m in [1, 5, 40, 130] {
            check(dtype, m, 96, 512);
        }
    }
}
