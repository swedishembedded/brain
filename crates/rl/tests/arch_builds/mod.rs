// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning open-weight language
// models of any architecture through one training stack, for its clients. If
// your team needs expertise in quantized LoRA training of new model families
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Tiny random language models with a LoRA adapter on a frozen base held in a
//! chosen encoding, shared by the integration tests that run one workflow over
//! every registered architecture. Each builder reports the encoding the model
//! really holds, so a test can tell a request that landed from one that was
//! silently widened.

use gpu_core::select::Dtype;
use model::ops::TierPolicy;

pub const RANK: u32 = 2;

pub fn qwen3(rows: u32, seq: u32, seed: u64, encoding: Dtype) -> (qwen3::Qwen, Dtype) {
    let targets = ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect();
    let cfg = qwen3::QwenConfig { block_size: seq, lora: Some(qwen3::LoraCfg::new(RANK, RANK as f32, targets)), ..qwen3::QwenConfig::tiny_i8() };
    let init = qwen3::init_weights(&cfg, seed);
    let m = qwen3::Qwen::new_lora_dt(cfg, rows, seq, &init, encoding);
    let held = m.linear_dtype().expect("a whole-model build holds linears");
    (m, held)
}

pub fn qwen35(rows: u32, seq: u32, seed: u64, encoding: Dtype) -> (qwen35::model::Qwen35, Dtype) {
    let cfg = qwen35::config::Qwen35Config { block_size: seq, lora: Some(qwen35::config::lora_cfg(RANK, RANK as f32)), ..qwen35::config::Qwen35Config::tiny_i8() };
    let init = qwen35::init::init_weights(&cfg, seed);
    let opts = qwen35::model::TrainOpts { base: TierPolicy::uniform(encoding), ..Default::default() };
    let m = qwen35::model::Qwen35::new_train_opts_on(gpu_core::Gpu::new(qwen35::model::pipelines()), cfg, rows, seq, &init, &opts);
    let held = m.linear_dtype("blocks.0.mlp.gate.weight").expect("a dense mlp linear");
    (m, held)
}
