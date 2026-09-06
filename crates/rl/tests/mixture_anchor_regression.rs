// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Anchor-regression gate for `rl::objective::mixture::Mixture`
//! (self-improve roadmap P15): training on a "primary" token distribution
//! ALONE measurably raises a held-out "anchor" dataset's loss (a tiny
//! stand-in for catastrophic forgetting - the tied embedding means pushing
//! probability mass toward the primary distribution's ids pulls it away
//! from every other id, anchor ids included); mixing anchor micro-steps in
//! via `Mixture` keeps that same held-out anchor loss from degrading
//! nearly as much, for the same total step budget and identical initial
//! weights. This is NOT a gradient-correctness test (`Mixture` adds no new
//! gradient math to check) - it is the behavioral gate the phase's spec
//! states explicitly.

use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use model::Objective;
use qwen3::{Qwen, QwenConfig};
use rl::objective::mixture::{Anchor, Mixture};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// Cycle `pattern` out to `len` tokens (`len` need not be a multiple of
/// `pattern.len()`).
fn repeating_tokens(pattern: &[u32], len: usize) -> Vec<u32> {
    (0..len).map(|i| pattern[i % pattern.len()]).collect()
}

/// Run `steps` optimizer steps of `obj` against `model` - the shared
/// harness both the "primary alone" and "mixture" branches below drive
/// identically, so the ONLY difference between the two branches is which
/// `Objective` runs (plain primary-only vs. a `Mixture` of primary+anchor).
fn train_steps<O: Objective<Qwen>>(model: &Qwen, mut obj: O, steps: u32, lr: f32, seed: u64) {
    let mut rng = Rng::new(seed);
    for step in 0..steps {
        model.zero_grads();
        obj.micro_step(model, &mut rng);
        model.adamw_step(step + 1, lr, 0.0, Some(1.0), 1.0);
        model.poll_wait();
    }
}

#[test]
fn mixing_in_the_anchor_keeps_held_out_anchor_loss_from_degrading_as_much_as_primary_alone() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig::tiny(); // vocab = 23, block_size = 12
    let init = qwen3::init_weights(&cfg, 5);
    let batch_cfg = BatchConfig { batch_size: 1, block_size: 8, ..BatchConfig::default() };

    // Disjoint id ranges so training hard on `primary` only ever sees
    // `anchor`'s ids through the tied softmax's normalization, never
    // directly - a real (if miniature) forgetting mechanism, not a fudge.
    let primary_pattern = [15u32, 16, 17, 18, 19, 20, 21, 22];
    let anchor_pattern = [0u32, 1, 2, 3, 4, 5, 6, 7];
    // Same ids, different phase - a genuinely held-out set of windows.
    let anchor_val_pattern = [4u32, 5, 6, 7, 0, 1, 2, 3];

    let anchor_val = || Anchor::new(TokenDataset::new(repeating_tokens(&anchor_val_pattern, 200), &batch_cfg), batch_cfg.clone());

    const STEPS: u32 = 120;
    const LR: f32 = 3e-2;
    const TRAIN_SEED: u64 = 4242;
    const EVAL_BATCHES: u32 = 20;

    // Baseline: the shared initial weights' own anchor loss, before any
    // training - both branches start here.
    let baseline_model = Qwen::new(cfg.clone(), 1, 8, &init);
    let baseline_anchor_loss = anchor_val().eval(&baseline_model, &mut Rng::new(999), EVAL_BATCHES).expect("anchor eval");

    // Branch A: primary objective alone, the same total step budget.
    let model_primary_only = Qwen::new(cfg.clone(), 1, 8, &init);
    let primary_obj = Anchor::new(TokenDataset::new(repeating_tokens(&primary_pattern, 200), &batch_cfg), batch_cfg.clone());
    train_steps(&model_primary_only, primary_obj, STEPS, LR, TRAIN_SEED);
    let primary_only_anchor_loss = anchor_val().eval(&model_primary_only, &mut Rng::new(999), EVAL_BATCHES).expect("anchor eval");

    // Branch B: a Mixture of the SAME primary objective plus an anchor
    // arm (a disjoint anchor TRAIN split from `anchor_val`'s held-out
    // one), same total step budget, same seeds.
    let model_mixture = Qwen::new(cfg.clone(), 1, 8, &init);
    let primary_obj_b = Anchor::new(TokenDataset::new(repeating_tokens(&primary_pattern, 200), &batch_cfg), batch_cfg.clone());
    let anchor_obj_b = Anchor::new(TokenDataset::new(repeating_tokens(&anchor_pattern, 200), &batch_cfg), batch_cfg.clone());
    let mixture: Mixture<Qwen> =
        Mixture::new(vec![(0.5, Box::new(primary_obj_b) as Box<dyn Objective<Qwen>>), (0.5, Box::new(anchor_obj_b) as Box<dyn Objective<Qwen>>)]);
    train_steps(&model_mixture, mixture, STEPS, LR, TRAIN_SEED);
    let mixture_anchor_loss = anchor_val().eval(&model_mixture, &mut Rng::new(999), EVAL_BATCHES).expect("anchor eval");

    assert!(
        primary_only_anchor_loss.is_finite() && mixture_anchor_loss.is_finite() && baseline_anchor_loss.is_finite(),
        "non-finite loss: baseline {baseline_anchor_loss} primary_only {primary_only_anchor_loss} mixture {mixture_anchor_loss}"
    );
    // The setup must actually produce forgetting on the primary-alone
    // branch, or this test would not be exercising anything.
    assert!(
        primary_only_anchor_loss > baseline_anchor_loss + 0.05,
        "expected primary-only training to measurably raise the anchor's held-out loss: baseline {baseline_anchor_loss}, primary-only {primary_only_anchor_loss}"
    );
    // The gate: mixing in the anchor keeps that same held-out loss from
    // degrading nearly as much.
    assert!(
        mixture_anchor_loss < primary_only_anchor_loss - 0.05,
        "expected the Mixture branch's anchor loss ({mixture_anchor_loss}) to be measurably lower than training on the primary objective alone ({primary_only_anchor_loss})"
    );
}
