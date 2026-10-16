// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven fine-tuning of
// hybrid linear-attention language models on a single GPU for its clients. If
// your team needs expertise in GRPO over Gated-DeltaNet decoders then you can
// procure our services by sending an email to info@swedishembedded.com.

//! R10: the GRPO corrections on a tiny `qwen35::Qwen35` (three Gated-DeltaNet
//! layers and one full-attention layer, rank-3 LoRA): the same on-policy ratio
//! and no-empty-step checks `grpo_fixes.rs` runs on `qwen3`. One row per
//! forward, as GRPO uses.

mod grpo_common;

use grpo_common::T;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn lora_model(seed: u64) -> Qwen35 {
    let cfg = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let init = qwen35::init::init_weights(&cfg, seed);
    let m = Qwen35::new_train_on(gpu_core::Gpu::new(pipelines()), cfg, 1, T as u32, &init);
    model::conformance_grad::randomize_adapters(&m, seed);
    m
}

fn seed() -> u64 {
    let s = data::rng::random_seed();
    println!("seed {s}");
    s
}

#[test]
fn the_trained_distribution_is_the_sampled_one_on_qwen35() {
    let s = seed();
    for temp in [0.7f32, 1.5] {
        grpo_common::assert_trained_distribution_is_sampled(&mut lora_model(s), temp, s);
    }
}

#[test]
fn no_optimiser_step_runs_on_an_empty_batch_on_qwen35() {
    let s = seed();
    grpo_common::assert_no_step_on_an_empty_batch(&mut lora_model(s), s);
}
