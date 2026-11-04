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
//! A base can also be mixed, each linear at its own tier (`TierPolicy` rules): a family that holds
//! one trains over it with every leaf at its tier, and one that holds a uniform base only refuses it
//! by name.

mod grpo_common;

use std::path::Path;

use gpu_core::select::Dtype;
use grpo_common::T;
use model::finetune::{ChatTrainer, LoraRequest};
use model::ops::TierPolicy;
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

fn lora(base: &TierPolicy) -> LoraRequest<'_> {
    LoraRequest { rank: 2, alpha: 2.0, continue_from: None, base }
}

fn every_declared_base_trains_and_an_undeclared_one_is_refused<Trainer: ChatTrainer>(write_base: impl Fn(&Path)) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    write_base(&base);
    let opts = FitOpts { steps: 2, batch_size: 1, block_size: T as u32, lr: 1e-2, seed: data::rng::random_seed(), ..Default::default() };
    let bases = Trainer::LORA_BASES;

    for &dtype in bases.bases {
        let mut fit = DistillFit::new(DistillConfig { seq_len: T }, vec![example()], &opts, FitControl::default());
        let uniform = TierPolicy::uniform(dtype);
        let (report, _) = Trainer::train_lora(&base, &lora(&uniform), &opts, &mut fit).unwrap_or_else(|e| panic!("{} over {dtype:?}: {e}", bases.model));
        assert!(report.initial_loss.is_finite() && report.final_loss.is_some_and(f32::is_finite) && !report.interrupted, "{} over {dtype:?}: {report:?}", bases.model);
    }

    let undeclared = [Dtype::Q4, Dtype::F16].into_iter().find(|d| !bases.bases.contains(d)).expect("some format is undeclared");
    let mut fit = DistillFit::new(DistillConfig { seq_len: T }, vec![example()], &opts, FitControl::default());
    let uniform = TierPolicy::uniform(undeclared);
    let err = Trainer::train_lora(&base, &lora(&uniform), &opts, &mut fit).err().expect("an undeclared base is refused");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "{err}");
    assert!(err.to_string().contains(bases.model), "{err}");

    let mixed_undeclared = TierPolicy::uniform(Dtype::F32).with(&["mlp"], undeclared);
    let mut fit = DistillFit::new(DistillConfig { seq_len: T }, vec![example()], &opts, FitControl::default());
    let err = Trainer::train_lora(&base, &lora(&mixed_undeclared), &opts, &mut fit).err().expect("a rule naming an undeclared base is refused");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "{err}");
}

/// The mixed base a family that holds one leaf per tier trains over.
fn mixed() -> TierPolicy {
    TierPolicy::uniform(Dtype::F32).with(&["attn"], Dtype::I8).with(&["mlp"], Dtype::NF4)
}

fn mixed_base_is_refused_by_name<Trainer: ChatTrainer>(write_base: impl Fn(&Path)) {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    write_base(&base);
    let opts = FitOpts { steps: 1, batch_size: 1, block_size: T as u32, lr: 1e-2, seed: data::rng::random_seed(), ..Default::default() };
    let mut fit = DistillFit::new(DistillConfig { seq_len: T }, vec![example()], &opts, FitControl::default());
    let declared = Trainer::LORA_BASES.bases;
    assert!(declared.len() >= 2, "{}: a mixed base needs two declared tiers", Trainer::LORA_BASES.model);
    let mixed_declared = TierPolicy::uniform(declared[0]).with(&["mlp"], declared[declared.len() - 1]);
    let err = Trainer::train_lora(&base, &lora(&mixed_declared), &opts, &mut fit).err().expect("a mixed base is refused");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported, "{err}");
    assert!(err.to_string().contains(Trainer::LORA_BASES.model) && err.to_string().contains("mixed"), "{err}");
}

#[test]
fn qwen3_trains_over_a_mixed_base_with_every_leaf_at_its_tier() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let base = dir.path().join("base.safetensors");
    let cfg = qwen3::QwenConfig { block_size: T as u32, ..qwen3::QwenConfig::tiny_i8() };
    qwen3::Qwen::new(cfg.clone(), 1, T as u32, &qwen3::init_weights(&cfg, data::rng::random_seed())).save(base.to_str().unwrap());
    let opts = FitOpts { steps: 2, batch_size: 1, block_size: T as u32, lr: 1e-2, seed: data::rng::random_seed(), ..Default::default() };
    let mut fit = DistillFit::new(DistillConfig { seq_len: T }, vec![example()], &opts, FitControl::default());
    let (report, trained) = qwen3::finetune::Qwen3Trainer::train_lora(&base, &lora(&mixed()), &opts, &mut fit).expect("qwen3 holds a mixed base");
    assert!(report.final_loss.is_some_and(f32::is_finite), "{report:?}");
    let qwen3::finetune::Trained::Single(m) = trained else { panic!("a tiny model trains on one card") };
    for l in 0..cfg.n_layers {
        for (leaf, want) in [("attn.wq.weight", Dtype::I8), ("attn.wo.weight", Dtype::I8), ("mlp.up.weight", Dtype::NF4), ("mlp.down.weight", Dtype::NF4)] {
            assert_eq!(m.linear_dtype_of(&format!("blocks.{l}.{leaf}")), Some(want), "blocks.{l}.{leaf}");
        }
    }
}

#[test]
fn qwen3_trains_over_every_base_it_declares() {
    every_declared_base_trains_and_an_undeclared_one_is_refused::<qwen3::finetune::Qwen3Trainer>(|path| {
        let cfg = qwen3::QwenConfig { block_size: T as u32, ..qwen3::QwenConfig::tiny_i8() };
        qwen3::Qwen::new(cfg.clone(), 1, T as u32, &qwen3::init_weights(&cfg, data::rng::random_seed())).save(path.to_str().unwrap());
    });
}

fn write_qwen35(path: &Path) {
    let cfg = qwen35::config::Qwen35Config { block_size: T as u32, ..qwen35::config::Qwen35Config::tiny_i8() };
    <qwen35::model::Qwen35 as Model>::new(cfg.clone(), 1, T as u32, &qwen35::init::init_weights(&cfg, data::rng::random_seed())).save(path.to_str().unwrap());
}

#[test]
fn qwen35_trains_over_every_base_it_declares() {
    every_declared_base_trains_and_an_undeclared_one_is_refused::<qwen35::finetune::Qwen35Trainer>(write_qwen35);
}

#[test]
fn qwen35_refuses_a_mixed_base_by_name() {
    mixed_base_is_refused_by_name::<qwen35::finetune::Qwen35Trainer>(write_qwen35);
}

#[test]
fn qwen35moe_trains_over_every_base_it_declares() {
    every_declared_base_trains_and_an_undeclared_one_is_refused::<qwen35moe::finetune::Qwen35MoeTrainer>(|path| {
        let cfg = qwen35moe::config::Qwen35Config { n_layers: 4, block_size: T as u32, ..qwen35moe::config::Qwen35Config::tiny() };
        let init = <qwen35moe::model::Qwen35 as Model>::init_weights(&cfg, data::rng::random_seed());
        <qwen35moe::model::Qwen35 as Model>::new(cfg, 1, T as u32, &init).save(path.to_str().unwrap());
    });
}

