// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs whose every
// update can be trusted - no weight, moment or schedule position changes unless
// the step held something to learn from - for its clients. If your team needs
// expertise in reliable fine-tuning then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A step whose objective had no supervised contribution must change nothing:
//! not the weights (weight decay), not the AdamW moments, not the applied-step
//! count that drives bias correction. Checked through the shared fit loop on a
//! tiny real `qwen3::Qwen` with a rank-2 LoRA, for every objective whose data
//! source may come up empty.

use model::train::{fit_controlled, FitControl, FitOpts};
use model::{Model, Objective};
use qwen3::{LoraCfg, Qwen, QwenConfig};
use rl::objective::distill::{DistillConfig, DistillExample, DistillTopK};
use rl::objective::dpo::{Dpo, DpoConfig, DpoPair};

const SEQ_LEN: usize = 8;

fn lora_model(rows: u32) -> Qwen {
    let cfg = QwenConfig { lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 11);
    let m = Qwen::new(cfg, rows, SEQ_LEN as u32, &init);
    model::conformance_grad::randomize_adapters(&m, 11);
    m
}

/// Every trained tensor and its AdamW moments, bit for bit.
fn snapshot(m: &Qwen) -> Vec<(String, Vec<u32>, Vec<u32>)> {
    let names = m.optimized_params().expect("a LoRA model exposes its optimised set");
    assert!(!names.is_empty());
    names
        .into_iter()
        .map(|n| {
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
            let moments = m.read_moments(&n).map(|(a, b)| a.into_iter().chain(b).map(f32::to_bits).collect()).unwrap_or_default();
            (n.clone(), bits(m.read_weight(&n)), moments)
        })
        .collect()
}

fn assert_empty_steps_change_nothing<O: Objective<Qwen>>(rows: u32, obj: O) {
    let m = lora_model(rows);
    let before = snapshot(&m);
    let opts = FitOpts { steps: 3, batch_size: rows, block_size: SEQ_LEN as u32, lr: 1e-2, min_lr: 1e-2, weight_decay: 0.1, eval_interval: 0, ..Default::default() };
    let (report, m) = fit_controlled(m, obj, &opts, None, FitControl::default()).expect("the run completes");
    assert_eq!(report.steps_completed, 3);
    assert_eq!(report.applied_steps, 0, "no step held a signal, so none may be applied");
    assert!(snapshot(&m) == before, "weights and moments must be byte-identical after steps with no supervised contribution");
}

#[test]
fn dpo_steps_without_a_pair_change_no_weights_moments_or_step_count() {
    let obj = Dpo::new(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, |_| None::<DpoPair>, |_| (Vec::new(), Vec::new()));
    assert_empty_steps_change_nothing(2, obj);
}

#[test]
fn distill_steps_without_an_example_change_no_weights_moments_or_step_count() {
    let obj = DistillTopK::new(DistillConfig { seq_len: SEQ_LEN }, |_| None::<DistillExample>);
    assert_empty_steps_change_nothing(1, obj);
}

#[test]
fn distill_steps_whose_example_has_no_teacher_signal_change_nothing() {
    // A completion with no teacher entries yields zero forward/backward passes.
    let example = DistillExample { prompt: vec![1, 2, 3], completion: vec![4, 5], teacher: vec![Default::default(); 2] };
    let obj = DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, vec![example]);
    assert_empty_steps_change_nothing(1, obj);
}

#[test]
fn a_mixture_whose_drawn_arms_have_nothing_to_learn_changes_nothing() {
    let dpo = Dpo::new(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, |_| None::<DpoPair>, |_| (Vec::new(), Vec::new()));
    let obj = rl::objective::mixture::Mixture::<Qwen>::new(vec![(1.0, Box::new(dpo))]);
    assert_empty_steps_change_nothing(2, obj);
}
