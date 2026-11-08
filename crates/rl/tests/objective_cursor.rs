// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs whose data order
// is a property of the run, not of how often it was probed or restarted, for
// its clients. If your team needs expertise in reproducible fine-tuning then
// you can procure our services by sending an email to info@swedishembedded.com.

//! The loss estimate a run starts from (`Objective::loss_probe`) must not move
//! an objective's data cursor: the first training step trains the first record
//! whether or not the run was probed. Checked on a tiny real `qwen3::Qwen` with
//! a rank-2 LoRA, by comparing the first step's gradients with and without five
//! probes before it, for every objective that walks a fixed dataset.

use data::rng::Rng;
use model::train::{fit_controlled, FitControl, FitOpts, StepReport};
use model::{Model, Objective};
use qwen3::{LoraCfg, Qwen, QwenConfig};
use rl::objective::distill::{DistillConfig, DistillExample, DistillTopK, TopK};
use rl::objective::dpo::{Dpo, DpoConfig, DpoPair};

const SEQ_LEN: usize = 8;

fn lora_model(rows: u32) -> Qwen {
    let cfg = QwenConfig { lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 21);
    let m = Qwen::new(cfg, rows, SEQ_LEN as u32, &init);
    model::conformance_grad::randomize_adapters(&m, 21);
    m
}

fn first_step_gradients<O: Objective<Qwen>>(rows: u32, mut obj: O, probes: usize) -> Vec<Vec<u32>> {
    let mut m = lora_model(rows);
    obj.prepare(&mut m).unwrap();
    obj.begin(7, 0);
    let mut rng = Rng::new(3);
    for _ in 0..probes {
        obj.loss_probe(&m, &mut rng.clone());
    }
    m.zero_grads();
    obj.micro_step(&m, &mut rng);
    m.optimized_params().unwrap().iter().map(|n| m.read_grad(n).into_iter().map(f32::to_bits).collect()).collect()
}

fn pairs() -> Vec<DpoPair> {
    (0..3u32).map(|i| DpoPair { prompt: vec![1, 2], chosen: vec![3 + i, 4], rejected: vec![9 - i, 8] }).collect()
}

fn examples() -> Vec<DistillExample> {
    (0..3u32)
        .map(|i| DistillExample {
            prompt: vec![1, 2],
            completion: vec![3 + i, 4],
            teacher: vec![TopK { ids: vec![3 + i, 20], probs: vec![0.7, 0.3] }, TopK { ids: vec![4, 6], probs: vec![0.6, 0.4] }],
        })
        .collect()
}

#[test]
fn probing_a_dpo_run_does_not_advance_its_pair_cursor() {
    let make = || Dpo::try_from_pairs(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, pairs(), |p: &DpoPair| (vec![-1.0; p.chosen.len()], vec![-1.0; p.rejected.len()])).expect("a valid DPO configuration");
    assert!(first_step_gradients(2, make(), 0) == first_step_gradients(2, make(), 5), "five probes moved the first trained pair");
}

#[test]
fn probing_a_distillation_run_does_not_advance_its_example_cursor() {
    let make = || DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, examples());
    assert!(first_step_gradients(1, make(), 0) == first_step_gradients(1, make(), 5), "five probes moved the first trained example");
}

/// Every trained tensor and its AdamW moments, bit for bit.
fn trained_state(m: &Qwen) -> Vec<(String, Vec<u32>, Vec<u32>)> {
    m.optimized_params()
        .expect("a LoRA model exposes its optimised set")
        .into_iter()
        .map(|n| {
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
            let moments = m.read_moments(&n).map(|(a, b)| a.into_iter().chain(b).map(f32::to_bits).collect()).unwrap_or_default();
            (n.clone(), bits(m.read_weight(&n)), moments)
        })
        .collect()
}

const STEPS: u32 = 7;
const STOP_AFTER: u32 = 4;

/// Train `STEPS` steps straight, and again as `STOP_AFTER` steps, a stop, and a resume into a fresh model and
/// a fresh objective. Three records against five probes and four steps: any restart of the cursor, or any
/// probe that moved it, lands the resumed run on another record.
fn assert_resume_continues_the_stream<O: Objective<Qwen>>(name: &str, rows: u32, make: impl Fn() -> O) {
    let opts = FitOpts { steps: STEPS, batch_size: rows, block_size: SEQ_LEN as u32, lr: 1e-2, min_lr: 1e-2, weight_decay: 0.1, eval_interval: 0, ..Default::default() };
    let (straight_report, straight) = fit_controlled(lora_model(rows), make(), &opts, None, FitControl::default()).expect("the uninterrupted run");

    let dir = std::env::temp_dir().join(format!("objective-cursor-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let state = dir.join("train.state");
    let identity = serde_json::json!({ "run": name });
    let mut stop = |s: &StepReport| s.step < STOP_AFTER;
    let control = FitControl { on_step: Some(&mut stop), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, _) = fit_controlled(lora_model(rows), make(), &opts, None, control).expect("the interrupted run");
    assert!(first.interrupted);
    let control = FitControl { on_step: None, state: Some(&state), state_every: 0, identity, snapshots: None, grad_sync: None };
    let (resumed_report, resumed) = fit_controlled(lora_model(rows), make(), &opts, None, control).expect("the resumed run");
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(resumed_report.resumed_at, Some(STOP_AFTER));
    assert!(trained_state(&straight) == trained_state(&resumed), "{name}: weights and moments after a stop and resume differ from the uninterrupted run");
    assert_eq!(resumed_report.final_loss.map(f32::to_bits), straight_report.final_loss.map(f32::to_bits), "{name}: final loss");
}

#[test]
fn a_resumed_dpo_run_continues_its_pair_stream() {
    assert_resume_continues_the_stream("dpo", 2, || Dpo::try_from_pairs(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, pairs(), |p: &DpoPair| (vec![-1.0; p.chosen.len()], vec![-1.0; p.rejected.len()])).expect("a valid DPO configuration"));
}

#[test]
fn a_resumed_dpo_run_over_an_indexed_source_continues_it() {
    let source = pairs();
    assert_resume_continues_the_stream("dpo-indexed", 2, || {
        let source = source.clone();
        Dpo::try_new(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, move |n: u64| Some(source[(n * 2 % 3) as usize].clone()), |p: &DpoPair| (vec![-1.0; p.chosen.len()], vec![-1.0; p.rejected.len()])).expect("a valid DPO configuration")
    });
}

#[test]
fn a_resumed_distillation_run_continues_its_example_stream() {
    assert_resume_continues_the_stream("distill", 1, || DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, examples()));
}
