// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A Qwen3.5 LoRA trainer whose frozen base is held in bf16 (half the bytes
//! of the fp32 base) trains the same adapters the fp32 build does. The base
//! values are bf16-representable, so the only difference between the two
//! builds is the storage tier of the base matmuls: the loss and every
//! adapter gradient agree, and a fresh adapter (B = 0) leaves the base
//! model's logits exactly as they were.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn cpu() -> Gpu {
    Gpu::new_cpu(pipelines())
}

fn build(cfg: &Qwen35Config, init: &std::collections::HashMap<String, Vec<f32>>, base: Dtype) -> Qwen35 {
    Qwen35::new_train_opts_on(cpu(), cfg.clone(), 1, T as u32, init, &opts(base))
}

#[test]
fn the_bf16_base_trains_the_adapters_the_fp32_base_does() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    let want = loss_and_grads(&build(&cfg, &init, Dtype::F32));
    let half = build(&cfg, &init, Dtype::BF16);
    assert_eq!(half.linear_dtype("blocks.0.mlp.gate.weight"), Some(Dtype::BF16), "the base linears really are bf16");
    assert_eq!(half.linear_dtype("blocks.3.self_attn.q_proj.weight"), Some(Dtype::BF16));
    assert_same_training_signal(loss_and_grads(&half), want, 1e-4, 0.9999);
}

#[test]
fn a_fresh_adapter_leaves_the_bf16_base_logits_unchanged() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.0);
    let with_adapter = build(&cfg, &init, Dtype::BF16).logits_all(&TOKENS);
    let base_cfg = Qwen35Config { lora: None, ..cfg };
    let base_init: std::collections::HashMap<_, _> = init.iter().filter(|(n, _)| !n.contains(".lora_")).map(|(n, v)| (n.clone(), v.clone())).collect();
    let base = Qwen35::new_on_dt(cpu(), base_cfg, 1, T as u32, &base_init, &TierPolicy::uniform(Dtype::BF16));
    assert_eq!(with_adapter, base.logits_all(&TOKENS), "B = 0 adds exactly nothing to the bf16 base");
}

/// The frozen base is untouched by training and reads back as the values it
/// was built from (bf16-representable, so the packed tier is lossless).
#[test]
fn a_bf16_base_stays_frozen_and_reads_back() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    let m = build(&cfg, &init, Dtype::BF16);
    m.set_batch(&TOKENS, &TARGETS);
    for step in 1..=3 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 1e-1, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let mut adapters_moved = false;
    for name in cfg.param_list().into_iter().map(|(n, _)| n) {
        let now = m.read_weight(&name);
        if name.contains(".lora_") {
            adapters_moved |= now != init[&name];
        } else {
            assert_eq!(now, init[&name], "frozen base weight {name} changed (or does not read back)");
        }
    }
    assert!(adapters_moved, "no adapter moved");
}

/// Train, save the adapter alone, rebuild a bf16-base trainer from the base
/// plus the saved adapter: it must compute the trained model's logits.
#[test]
fn an_adapter_trained_on_a_bf16_base_survives_save_and_reload() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.0);
    let m = build(&cfg, &init, Dtype::BF16);
    m.set_batch(&TOKENS, &TARGETS);
    for step in 1..=8 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let trained = m.logits_all(&TOKENS);
    let path = std::env::temp_dir().join(format!("brain-qwen35-bf16-adapter-{}.safetensors", std::process::id()));
    qwen35::lora::save_adapter(path.to_str().unwrap(), &m, "test/adapter", "test/base", None).unwrap();
    let adapter = checkpoint::load(path.to_str().unwrap()).by_role("");
    std::fs::remove_file(&path).unwrap();
    assert!(adapter.keys().all(|k| k.contains(".lora_")), "the adapter file carries only adapter tensors");

    let mut reload = init.clone();
    reload.extend(adapter);
    let again = build(&cfg, &reload, Dtype::BF16).logits_all(&TOKENS);
    let fresh = build(&cfg, &init, Dtype::BF16).logits_all(&TOKENS);
    let dist = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f32>() / a.len() as f32;
    assert!(dist(&trained, &again) < 1e-6, "reloaded adapter does not reproduce the trained model: {}", dist(&trained, &again));
    assert!(dist(&trained, &fresh) > 1e-2, "training did not move the logits ({})", dist(&trained, &fresh));
}

#[test]
fn a_lora_trainer_on_a_bf16_base_overfits_a_fixed_batch() {
    // The model is a few hundred elements wide: on every core of the machine its
    // 120 steps spent 12 CPU-minutes waking a pool for 6 s of arithmetic.
    Gpu::with_cpu_threads(2, || {
        let cfg = lora_config();
        let init = weights(&cfg, 0.0);
        let m = build(&cfg, &init, Dtype::BF16);
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
        assert!(after < before * 0.8, "bf16-base LoRA did not learn: {before} -> {after}");
    });
}

/// Full-parameter training stays fp32-only; only LoRA may sit on a narrow base.
#[test]
#[should_panic(expected = "frozen base")]
fn a_narrow_base_without_lora_is_refused() {
    let cfg = Qwen35Config { lora: None, ..lora_config() };
    let init = weights(&cfg, 0.0);
    let _ = build(&cfg, &init, Dtype::BF16);
}
