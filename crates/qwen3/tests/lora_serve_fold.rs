// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The path `crates/cli/src/resident_llm.rs`'s `QwenResident::activate` takes
//! to serve a named LoRA adapter: fold the adapter into the base tensors
//! (`qwen3::lora::fold_adapter_into`, already proven exact against the live
//! unfolded forward by `crates/qwen3/tests/lora_adapter_file.rs`), then build
//! a DECODE-ONLY KV-cache model from the folded tensors via
//! `Qwen::from_tensors_decode` -- the API this test exercises, which exists
//! for adapter serving. Proves the decode-only construction from an in-memory
//! folded tensor map produces the SAME greedy generation as the live,
//! unfolded, batched-forward trained model -- i.e. that serving a folded
//! adapter through the KV-cache path is behaviorally identical to training it.

use std::collections::HashMap;

use data::rng::Rng;
use qwen3::{LoraCfg, Qwen, QwenConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-serve-fold-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn decode_only_model_from_folded_tensors_matches_the_live_trained_forward() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }

    let base_cfg = QwenConfig::tiny();
    let base_init = qwen3::init_weights(&base_cfg, 21);
    // rank=3 coprime with tiny()'s head_dim=8/d_model=16 (avoiding a degenerate rank).
    let lora_cfg = QwenConfig { lora: Some(LoraCfg::attn(3, 6.0)), ..QwenConfig::tiny() };
    let mut init: HashMap<String, Vec<f32>> = qwen3::init_weights(&lora_cfg, 21);
    for (k, v) in &base_init {
        init.insert(k.clone(), v.clone());
    }
    let x: Vec<u32> = (0..12).map(|i| (i * 7 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 7 + 2) % 23).collect();
    let trained = Qwen::new(lora_cfg, 1, 12, &init);
    trained.set_batch(&x, &y);
    for step in 1..=8 {
        trained.zero_grads();
        trained.forward();
        trained.backward();
        trained.adamw_step(step, 5e-2, 0.0, Some(1.0), 1.0);
        trained.poll_wait();
    }

    let adapter_path = tmp("adapter").join("adapter.safetensors");
    qwen3::lora::save_adapter(adapter_path.to_str().unwrap(), &trained, "test/adapter", "test/base", None).unwrap();

    // The base tensors an inference load would see: every non-adapter param
    // (exactly what `checkpoint::load(weights).by_role("")` returns in
    // `QwenResident::activate`).
    let mut base_tensors: HashMap<String, Vec<f32>> =
        trained.ps.params.iter().filter(|(name, _)| !(name.ends_with(".lora_a") || name.ends_with(".lora_b"))).map(|(name, _)| (name.clone(), trained.read_weight(name))).collect();
    qwen3::lora::fold_adapter_into(&mut base_tensors, adapter_path.to_str().unwrap()).expect("fold_adapter_into");

    let prompt = &x[..6];
    let live_greedy = qwen3::sample::generate(&trained, prompt, 6, 0.0, 0, 1.0, None, &mut Rng::new(1));

    // The new bit: decode-only construction from the folded tensor map,
    // exactly what `Qwen::from_tensors_decode` (QwenResident::activate's
    // adapter path) builds.
    let served = Qwen::from_tensors_decode(QwenConfig::tiny(), &base_tensors, 32);
    let served_greedy = qwen3::sample::generate_kv(&served, prompt, 6, 0.0, 0, 1.0, None, &mut Rng::new(1));

    assert_eq!(
        served_greedy, live_greedy,
        "decode-only generation from the folded-adapter tensors diverged from the live trained model's greedy generation"
    );
}
