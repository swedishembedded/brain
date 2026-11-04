// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for showing that a trained adapter
// changes what a very large model says, without a second copy of the model,
// for its clients. If your team needs expertise in evaluating fine-tuned
// language models you can procure our services by sending an email to
// info@swedishembedded.com.

//! `AdapterSampler` decodes greedily from a streamed base with an adapter
//! overlaid (no fold into a second copy of the weights). Its answer must be
//! the answer of the independent KV-cache decode of the adapter folded into
//! the base; with the adapter switched off it must be the base's own; and a
//! trained adapter must make the two differ.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;
use qwen35::adapter_sample::AdapterSampler;
use gpu_core::select::Dtype;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

const NEW: usize = 6;
const PROMPT: usize = 5;

/// A fresh directory per call: the tests run in parallel and each builds its own fixture.
fn tmp(name: &str) -> std::path::PathBuf {
    static CALLS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-qwen35-adapter-sample-{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A tiny base checkpoint file and an adapter trained on it (rank 5, coprime
/// with the tiny dims) to continue a fixed token pattern.
fn base_and_adapter() -> (std::path::PathBuf, std::path::PathBuf, Vec<u32>) {
    let dir = tmp("fixture");
    let cfg = Qwen35Config::tiny();
    // seed: any base works; the test compares two decode paths of the same weights
    let base_init = qwen35::init::init_weights(&cfg, 11);
    let base = dir.join("base.safetensors");
    Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, cfg.block_size, &base_init).save(base.to_str().unwrap());

    let lora = Qwen35Config { lora: Some(lora_cfg(5, 8.0)), ..Qwen35Config::tiny() };
    let mut init: HashMap<String, Vec<f32>> = qwen35::init::init_weights(&lora, 11);
    init.extend(base_init);
    let t = cfg.block_size;
    let x: Vec<u32> = (0..t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), lora, 1, t, &init);
    m.set_batch(&x, &y);
    for step in 1..=12 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        m.poll_wait();
    }
    let adapter = dir.join("adapter.safetensors");
    qwen35::lora::save_adapter(adapter.to_str().unwrap(), &m, "test/adapter", "test/base", None).unwrap();
    (base, adapter, x[..PROMPT].to_vec())
}

/// Greedy continuation by the KV-cache decode of a plain model built from `weights`.
fn kv_greedy(weights: &HashMap<String, Vec<f32>>, prompt: &[u32]) -> Vec<u32> {
    let cfg = Qwen35Config::tiny();
    let m = Qwen35::new_on(Gpu::new(pipelines()), cfg, 1, (prompt.len() + NEW) as u32, weights);
    // seed: unused, temperature 0 is greedy
    qwen35::sample::generate_kv(&m, prompt, NEW, 0.0, 0, 1.0, &[], &mut Rng::new(0))
}

#[test]
fn the_overlay_decodes_as_the_folded_adapter_and_switches_off_to_the_base() {
    let (base, adapter, prompt) = base_and_adapter();
    let plain = checkpoint::load(base.to_str().unwrap()).by_role("");
    let mut folded = plain.clone();
    qwen35::lora::fold_adapter_into(&mut folded, adapter.to_str().unwrap()).unwrap();
    let (want_on, want_off) = (kv_greedy(&folded, &prompt), kv_greedy(&plain, &prompt));
    assert_ne!(want_on, want_off, "the fixture adapter must change the base's greedy output, or this test proves nothing");

    let mut s = AdapterSampler::open(&base, Some(&adapter), Dtype::F32, (PROMPT + NEW) as u32).expect("open");
    assert_eq!(s.greedy(&prompt, NEW, &[]).unwrap(), want_on, "adapter on");
    s.set_adapter(false);
    assert_eq!(s.greedy(&prompt, NEW, &[]).unwrap(), want_off, "adapter off");
    s.set_adapter(true);
    assert_eq!(s.greedy(&prompt, NEW, &[]).unwrap(), want_on, "adapter back on");
}

#[test]
fn a_sampler_without_an_adapter_is_the_base() {
    let (base, _adapter, prompt) = base_and_adapter();
    let plain = checkpoint::load(base.to_str().unwrap()).by_role("");
    let s = AdapterSampler::open(&base, None, Dtype::F32, (PROMPT + NEW) as u32).expect("open");
    assert_eq!(s.greedy(&prompt, NEW, &[]).unwrap(), kv_greedy(&plain, &prompt));
}

#[test]
fn a_prompt_that_does_not_fit_the_built_length_is_refused() {
    let (base, _adapter, prompt) = base_and_adapter();
    let s = AdapterSampler::open(&base, None, Dtype::F32, (PROMPT + NEW) as u32).expect("open");
    let err = s.greedy(&prompt, NEW + 1, &[]).unwrap_err();
    assert!(err.contains("length"), "{err}");
}
