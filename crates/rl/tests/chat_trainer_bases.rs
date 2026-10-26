// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning open-weight language
// models of any architecture over quantized frozen bases through one training
// stack, for its clients. If your team needs expertise in wiring a new model
// family into quantized post-training then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The `ChatTrainer` seam end to end, from a checkpoint file: every frozen-base format a trainer
//! declares (`ChatTrainer::LORA_BASES`) really trains through `train_lora`, here under a
//! distillation visitor, and a format it does not declare is refused by name before the
//! checkpoint is read. The test logic is generic over the trainer; a family adds a base writer.

mod grpo_common;

use std::path::Path;

use gpu_core::select::Dtype;
use grpo_common::T;
use model::finetune::{ChatTrainer, LoraRequest};
use model::{FitControl, FitOpts, Model};
use rl::fit::DistillFit;
use rl::objective::distill::{DistillConfig, DistillExample, TopK};

/// A supervised example: the teacher says what follows the prompt.
fn example() -> DistillExample {
    DistillExample {
        prompt: vec![1, 2, 3],
        completion: vec![4, 5],
        teacher: vec![TopK { ids: vec![4, 6], probs: vec![0.6, 0.4] }, TopK { ids: vec![5], probs: vec![1.0] }],
    }
}

fn every_declared_base_trains_and_an_undeclared_one_is_refused<Trainer: ChatTrainer>(write_base: impl Fn(&Path)) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    write_base(&base);
    let opts = FitOpts { steps: 2, batch_size: 1, block_size: T as u32, lr: 1e-2, seed: data::rng::random_seed(), ..Default::default() };
    let bases = Trainer::LORA_BASES;
    let lora = |dtype| LoraRequest { rank: 2, alpha: 2.0, continue_from: None, base: dtype };

    for &dtype in bases.bases {
        let mut fit = DistillFit::new(DistillConfig { seq_len: T }, vec![example()], &opts, FitControl::default());
        let (report, _) = Trainer::train_lora(&base, &lora(dtype), &opts, &mut fit).unwrap_or_else(|e| panic!("{} over {dtype:?}: {e}", bases.model));
        assert!(report.initial_loss.is_finite() && report.final_loss.is_some_and(f32::is_finite) && !report.interrupted, "{} over {dtype:?}: {report:?}", bases.model);
    }

    let undeclared = [Dtype::Q4, Dtype::F16].into_iter().find(|d| !bases.bases.contains(d)).expect("some format is undeclared");
    let mut fit = DistillFit::new(DistillConfig { seq_len: T }, vec![example()], &opts, FitControl::default());
    let err = Trainer::train_lora(&base, &lora(undeclared), &opts, &mut fit).err().expect("an undeclared base is refused");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "{err}");
    assert!(err.to_string().contains(bases.model), "{err}");
}

#[test]
fn qwen3_trains_over_every_base_it_declares() {
    every_declared_base_trains_and_an_undeclared_one_is_refused::<qwen3::finetune::Qwen3Trainer>(|path| {
        let cfg = qwen3::QwenConfig { block_size: T as u32, ..qwen3::QwenConfig::tiny_i8() };
        qwen3::Qwen::new(cfg.clone(), 1, T as u32, &qwen3::init_weights(&cfg, data::rng::random_seed())).save(path.to_str().unwrap());
    });
}

#[test]
fn qwen35_trains_over_every_base_it_declares() {
    every_declared_base_trains_and_an_undeclared_one_is_refused::<qwen35::finetune::Qwen35Trainer>(|path| {
        let cfg = qwen35::config::Qwen35Config { block_size: T as u32, ..qwen35::config::Qwen35Config::tiny_i8() };
        <qwen35::model::Qwen35 as Model>::new(cfg.clone(), 1, T as u32, &qwen35::init::init_weights(&cfg, data::rng::random_seed())).save(path.to_str().unwrap());
    });
}

#[test]
fn qwen35moe_trains_over_every_base_it_declares() {
    every_declared_base_trains_and_an_undeclared_one_is_refused::<qwen35moe::finetune::Qwen35MoeTrainer>(|path| {
        let cfg = qwen35moe::config::Qwen35Config { n_layers: 4, block_size: T as u32, ..qwen35moe::config::Qwen35Config::tiny() };
        let init = <qwen35moe::model::Qwen35 as Model>::init_weights(&cfg, data::rng::random_seed());
        <qwen35moe::model::Qwen35 as Model>::new(cfg, 1, T as u32, &init).save(path.to_str().unwrap());
    });
}
