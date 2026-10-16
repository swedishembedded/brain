// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! QLoRA: a Qwen3.5 LoRA trainer whose frozen base is held in 4-bit NF4. The
//! only approximation is the weight rounding, so a fp32 build over the values
//! the NF4 base actually holds (read back from it) computes the same loss and
//! adapter gradients, and the 4-bit base still learns.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn cfg() -> Qwen35Config {
    // `tiny_i8` keeps every contraction dimension a multiple of the NF4 group.
    Qwen35Config { block_size: T as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny_i8() }
}

fn build(cfg: &Qwen35Config, init: &std::collections::HashMap<String, Vec<f32>>, base: Dtype) -> Qwen35 {
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, T as u32, init, &opts(base))
}

#[test]
fn the_nf4_base_trains_the_adapters_a_fp32_build_over_its_dequantised_values_does() {
    let cfg = cfg();
    let init = weights(&cfg, 0.3);
    let nf4 = build(&cfg, &init, Dtype::NF4);
    assert_eq!(nf4.linear_dtype("blocks.0.mlp.gate.weight"), Some(Dtype::NF4));
    assert_eq!(nf4.linear_dtype("tok.weight"), Some(Dtype::BF16), "the embedding and head are held in bf16, not 4 bits");
    // What the NF4 base really holds, as the fp32 base of the reference.
    let mut held = init.clone();
    for name in cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| nf4.linear_dtype(n).is_some()) {
        held.insert(name.clone(), nf4.read_weight(&name));
    }
    assert_ne!(held["blocks.0.mlp.gate.weight"], init["blocks.0.mlp.gate.weight"], "NF4 rounds the weights");
    let want = loss_and_grads(&build(&cfg, &held, Dtype::F32));
    assert_same_training_signal(loss_and_grads(&nf4), want, 1e-4, 0.9999);
}

#[test]
fn a_4_bit_base_still_overfits_a_fixed_batch() {
    let cfg = cfg();
    let init = weights(&cfg, 0.0);
    let m = build(&cfg, &init, Dtype::NF4);
    m.set_batch(&TOKENS, &TARGETS);
    let before = m.forward();
    for step in 1..=120 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 3e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let after = m.forward();
    assert!(after < before * 0.8, "NF4-base LoRA did not learn: {before} -> {after}");
}
