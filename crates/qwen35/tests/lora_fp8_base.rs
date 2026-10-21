// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA fine-tuning of large
// language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A Qwen3.5 LoRA trainer whose frozen base is held as block-scaled FP8 (E4M3 or E5M2), the
//! encoding of the released checkpoint. The only approximation is the weight rounding, so an fp32
//! build over the values the FP8 base actually holds (read back from it) computes the same loss and
//! adapter gradients, and the FP8 base still learns.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn cfg() -> Qwen35Config {
    // `tiny_i8` keeps every contraction dimension a multiple of the quantisation group.
    Qwen35Config { block_size: T as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny_i8() }
}

fn build(cfg: &Qwen35Config, init: &std::collections::HashMap<String, Vec<f32>>, base: Dtype) -> Qwen35 {
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, T as u32, init, &opts(base))
}

#[test]
fn the_fp8_base_trains_the_adapters_a_fp32_build_over_its_decoded_values_does() {
    for fp8 in [Dtype::F8E4M3, Dtype::F8E5M2] {
        let cfg = cfg();
        let init = weights(&cfg, 0.3);
        let packed = build(&cfg, &init, fp8);
        assert_eq!(packed.linear_dtype("blocks.0.mlp.gate.weight"), Some(fp8), "the base is held as asked, not widened");
        // What the FP8 base really holds, as the fp32 base of the reference.
        let mut held = init.clone();
        for name in cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| packed.linear_dtype(n).is_some()) {
            held.insert(name.clone(), packed.read_weight(&name));
        }
        assert_ne!(held["blocks.0.mlp.gate.weight"], init["blocks.0.mlp.gate.weight"], "{fp8:?} rounds the weights");
        let want = loss_and_grads(&build(&cfg, &held, Dtype::F32));
        assert_same_training_signal(loss_and_grads(&packed), want, 1e-4, 0.9999);
    }
}

#[test]
fn an_fp8_base_still_overfits_a_fixed_batch() {
    let cfg = cfg();
    let init = weights(&cfg, 0.0);
    let m = build(&cfg, &init, Dtype::F8E4M3);
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
    assert!(after < before * 0.8, "FP8-base LoRA did not learn: {before} -> {after}");
}
