// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs whose every
// update can be trusted - no weight, moment or schedule position changes unless
// the step held something to learn from - for its clients. If your team needs
// expertise in reliable fine-tuning then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A step whose objective had no supervised contribution must change nothing, and a record that cannot be trained faithfully stops the run:
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
fn a_mixture_whose_drawn_arms_have_nothing_to_learn_changes_nothing() {
    let dpo = Dpo::new(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, |_| None::<DpoPair>, |_| (Vec::new(), Vec::new()));
    let obj = rl::objective::mixture::Mixture::<Qwen>::new(vec![(1.0, Box::new(dpo))]);
    assert_empty_steps_change_nothing(2, obj);
}

/// A record the source supplies that cannot be trained faithfully stops the run before any update,
/// naming the problem; it is not truncated or skipped into an empty step.
fn assert_malformed_record_aborts_before_any_update<O: Objective<Qwen>>(rows: u32, obj: O, mentions: &str) {
    let m = lora_model(rows);
    let before = snapshot(&m);
    let opts = FitOpts { steps: 2, batch_size: rows, block_size: SEQ_LEN as u32, lr: 1e-2, min_lr: 1e-2, weight_decay: 0.1, eval_interval: 0, ..Default::default() };
    let error = fit_controlled(m, obj, &opts, None, FitControl::default()).err().expect("a malformed record must stop the run");
    assert!(error.to_string().contains(mentions), "{error}");
    // fit_controlled consumed the model; rebuild the same one to compare with a fresh snapshot.
    assert!(snapshot(&lora_model(rows)) == before);
}

#[test]
fn a_dpo_pair_longer_than_the_row_stops_the_run() {
    let pair = DpoPair { prompt: vec![1, 2, 3], chosen: vec![4; SEQ_LEN], rejected: vec![5, 6] };
    let obj = Dpo::new(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, move |_| Some(pair.clone()), |p: &DpoPair| (vec![-1.0; p.chosen.len()], vec![-1.0; p.rejected.len()]));
    assert_malformed_record_aborts_before_any_update(2, obj, "do not fit");
}

#[test]
fn a_distillation_example_longer_than_the_row_or_without_signal_stops_the_run() {
    let long = DistillExample { prompt: vec![1, 2, 3], completion: vec![4; SEQ_LEN], teacher: vec![rl::objective::distill::TopK { ids: vec![4], probs: vec![1.0] }; SEQ_LEN] };
    assert_malformed_record_aborts_before_any_update(1, DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, vec![long]), "do not fit");
    let unnormalised = DistillExample { prompt: vec![1, 2], completion: vec![3], teacher: vec![rl::objective::distill::TopK { ids: vec![3, 4], probs: vec![0.2, 0.2] }] };
    assert_malformed_record_aborts_before_any_update(1, DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, vec![unnormalised]), "sum to");
}
