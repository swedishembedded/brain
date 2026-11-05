// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A two-level activation ([`Ops::act_two_level`]) multiplies a quantized
//! weight as if the activation were exact to its second int8 level.
//!
//! Swedish Embedded AB implements quantized inference that keeps the
//! full-precision model's results for its clients. If your team needs
//! expertise in low-precision linear algebra on accelerators, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! One int8 level leaves each 32-element group's rounding error at up to half
//! its step. The second level quantizes what the first drops, so its error is
//! at most half of a step 254 times finer. The weights here are exactly
//! representable in each tier (every group is `{-c, 0, c}`), so the bound is
//! the activation's alone, and a single-level product misses it by far. The
//! row count spans several of the second level's bounded passes and ends in a
//! partial one, so a pass folded into the wrong rows fails too.

use data::rng::Lcg;
use gpu_core::select::Dtype;
use model::ops::{Ops, Weight};

/// Activation-quantization group, in elements.
const GROUP: usize = 32;

/// `[n, k]` weights whose every 32-element group is a mix of `-c`, `0` and
/// `c`, which the group-wise int8 and int4 tiers hold exactly.
fn exact_weights(rng: &mut Lcg, n: usize, k: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(n * k);
    for _ in 0..n * k / GROUP {
        let c = 0.01 + (rng.next_u32() % 100) as f32 * 1e-3;
        w.push(c);
        w.extend((1..GROUP).map(|_| [-c, 0.0, c][(rng.next_u32() % 3) as usize]));
    }
    w
}

/// The second level's bound on `y[r, j]`: half of each group's second-level
/// step (at most the first step / 254) times the weight mass it meets, plus
/// fp32 summation.
fn bound(x: &[f32], w: &[f32], k: usize, r: usize, j: usize) -> f64 {
    let mut b = 0.0f64;
    for g in 0..k / GROUP {
        let xs = &x[r * k + g * GROUP..r * k + (g + 1) * GROUP];
        let step = f64::from(xs.iter().fold(0.0f32, |a, v| a.max(v.abs()))) / 127.0 / 254.0;
        let mass: f64 = w[j * k + g * GROUP..j * k + (g + 1) * GROUP].iter().map(|v| f64::from(v.abs())).sum();
        b += 0.5 * step * mass;
    }
    let terms: f64 = (0..k).map(|i| f64::from(x[r * k + i].abs()) * f64::from(w[j * k + i].abs())).sum();
    b + terms * 1e-6
}

/// How many outputs miss the second level's bound, for a two-level or a
/// single-level activation.
fn misses(dtype: Dtype, two_level: bool, m: usize, n: usize, k: usize) -> usize {
    let ops = Ops::new(gpu_core::testgpu::dev(model::ops::two_level_kernel_list())).expect("Ops::new");
    let g = ops.gpu();
    let mut rng = Lcg::new(data::rng::random_seed());
    let w = exact_weights(&mut rng, n, k);
    let x: Vec<f32> = (0..m * k).map(|_| (rng.next_u32() % 2001) as f32 / 1000.0 - 1.0).collect();
    let weight = Weight::upload(&ops, &w, n, k, dtype);
    let xb = g.storage_init("x", &x);
    let y = g.storage((m * n) as u64);
    let mut steps = Vec::new();
    let act = if two_level { ops.act_two_level(&mut steps, &xb, 0, m as u32, k as u32) } else { ops.act(&mut steps, &xb, 0, m as u32, k as u32) };
    ops.matmul(&mut steps, &weight, &act, &y, 0);
    g.submit(&[], &steps);
    g.poll_wait();
    let got = g.read(&y, m * n);
    let mut missed = 0;
    for r in 0..m {
        for j in 0..n {
            let want: f64 = (0..k).map(|i| f64::from(x[r * k + i]) * f64::from(w[j * k + i])).sum();
            if (f64::from(got[r * n + j]) - want).abs() > bound(&x, &w, k, r, j) {
                missed += 1;
            }
        }
    }
    missed
}

#[test]
fn a_two_level_activation_is_exact_to_its_second_level() {
    let m = model::ops::TWO_LEVEL_PASS_ROWS as usize * 2 + 77;
    for dtype in [Dtype::I8, Dtype::Q4] {
        assert_eq!(misses(dtype, true, m, 48, 256), 0, "{dtype:?}: two-level products past the second level's bound");
    }
    assert!(misses(Dtype::I8, false, 40, 48, 256) > 0, "a single-level activation meets the second level's bound - the gate above would be vacuous");
}
