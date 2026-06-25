// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P3 MASTER CORRECTNESS GATE: full-model finite-difference gradient check.
//!
//! Builds the whole tiny YOLOv8 detector (`Yolo::new(YoloConfig::tiny(2), b=4)`)
//! in [`LossMode::Proxy`], sets a fixed random image batch, runs BN in TRAIN
//! mode (batch N=4), and drives the repo's `directional_check` over EVERY
//! parameter tensor — backbone + PAN-FPN neck + 3-scale decoupled head. The
//! proxy loss `L = <r, raw_logits>` exercises every conv/bn/silu/concat/
//! upsample/pool + head conv backward, so this single gate validates the entire
//! architecture's backprop. CPU backend only (`Gpu::new_cpu`).
//!
//! `gradcheck` does not depend on `brain-yolo` (avoiding a cycle), so the check
//! harness lives here and calls `gradcheck::directional_check` directly via the
//! blanket `CheckModel for model::Model` impl.

use gradcheck::{directional_check, Report};
use model::Model;
use yolo::{LossMode, Yolo, YoloConfig};

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

/// Build the tiny detector in Proxy mode with a fixed random image batch and
/// gradient-check it. Same tolerance family as every other brain model.
fn check_yolo(seed: u64) -> Report {
    let cfg = YoloConfig::tiny(2);
    let b = 4u32;
    let init = yolo::init_weights(&cfg, seed);
    let model = Yolo::new(cfg.clone(), b, /*t*/ cfg.input, &init);
    model.set_mode(LossMode::Proxy);

    // Fixed random image batch [N,3,H,W].
    let n = (b * 3 * cfg.input * cfg.input) as usize;
    let img: Vec<f32> = randvec(seed ^ 0xA5A5, n);
    model.set_batch(model::Batch::Tensor { tokens: None, inputs: &img, targets: &[] });

    // eps = 5e-4 (vs the usual 5e-3): the directional check perturbs every
    // element of a tensor at once along a +/-1 direction, so the L2 step is
    // eps*sqrt(numel). The backbone conv tensors here are large (up to ~1e5
    // weights), so eps=5e-3 would sample points ~2.0 away in weight space —
    // deep in the SiLU/BN nonlinear regime, where the central difference no
    // longer agrees with the analytic directional derivative. eps=5e-4 keeps
    // the step ~0.2 (comparable to the other models' small tensors) while the
    // resulting loss delta (~0.1) stays well above fp32 round-off. The PASS
    // tolerance is unchanged: 4e-3 / 8e-2, same as every other brain model.
    directional_check(&model, 5e-4, 3, seed ^ 0x1234)
}

#[test]
fn yolo_analytic_grads_match_finite_differences() {
    let t0 = std::time::Instant::now();
    let report = check_yolo(7);
    let dt = t0.elapsed();
    println!(
        "check_yolo: {} params checked, worst rel-err {:.3e}, elapsed {:.2?}",
        report.checks.len(),
        report.max_rel(),
        dt
    );
    let fails = report.failures(ATOL, RTOL);
    if !fails.is_empty() {
        report.print();
    }
    assert!(
        fails.is_empty(),
        "yolo gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
}

/// Forward-shape gate: tiny config at input 128 -> A = 16^2 + 8^2 + 4^2 = 336
/// anchors; cls logits [N,336,nc], box logits [N,336,4*reg_max].
#[test]
fn forward_shapes() {
    let cfg = YoloConfig::tiny(2);
    let b = 2u32;
    let init = yolo::init_weights(&cfg, 1);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Proxy);

    let n = (b * 3 * cfg.input * cfg.input) as usize;
    let img: Vec<f32> = randvec(123, n);
    model.set_batch(model::Batch::Tensor { tokens: None, inputs: &img, targets: &[] });
    let _ = model.forward();

    let (cls, boxl) = model.raw_logits();
    let a = 16 * 16 + 8 * 8 + 4 * 4; // 336
    assert_eq!(cls.len(), (b as usize) * a * cfg.nc as usize, "cls [N,336,nc]");
    assert_eq!(boxl.len(), (b as usize) * a * (4 * cfg.reg_max) as usize, "box [N,336,4*reg_max]");
}

// Deterministic LCG -> values in (-1, 1) (matches the P1/P2 test generators).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1) | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.next_f32()).collect()
}
