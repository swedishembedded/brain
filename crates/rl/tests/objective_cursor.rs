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
            teacher: vec![TopK { ids: vec![3 + i, 5], probs: vec![0.7, 0.3] }, TopK { ids: vec![4, 6], probs: vec![0.6, 0.4] }],
        })
        .collect()
}

#[test]
fn probing_a_dpo_run_does_not_advance_its_pair_cursor() {
    let make = || Dpo::from_pairs(DpoConfig { beta: 0.1, seq_len: SEQ_LEN, nll_weight: 0.0 }, pairs(), |p: &DpoPair| (vec![-1.0; p.chosen.len()], vec![-1.0; p.rejected.len()]));
    assert!(first_step_gradients(2, make(), 0) == first_step_gradients(2, make(), 5), "five probes moved the first trained pair");
}

#[test]
fn probing_a_distillation_run_does_not_advance_its_example_cursor() {
    let make = || DistillTopK::from_examples(DistillConfig { seq_len: SEQ_LEN }, examples());
    assert!(first_step_gradients(1, make(), 0) == first_step_gradients(1, make(), 5), "five probes moved the first trained example");
}
