// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A LoRA fine-tune must change what the model actually outputs after a save
//! plus reload cycle, not just in the live process. Mirrors
//! `qwen3/tests/lora_roundtrip.rs`. No streaming `load_inference` exists yet
//! for this crate (that lands with CLI/serving, M11), so reload here goes
//! through the generic `checkpoint::load` plus `Qwen35::new_on` path
//! directly, exactly what a future `Qwen35::load_inference` would do
//! internally.

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-lora-roundtrip-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A LoRA-trained model's saved checkpoint must reproduce the SAME logits
/// when reloaded fresh, and those logits must differ from the untrained base
/// by a real margin (the gate a silently-dropped adapter would still pass if
/// only the reload-matches check ran).
#[test]
fn lora_adapter_survives_save_and_reload() {
    if skip() {
        return;
    }
    let base_cfg = Qwen35Config::tiny();
    let base_init = qwen35::init::init_weights(&base_cfg, 11);

    // rank=5 is coprime with tiny()'s head_dim=40/d_model=96/intermediate=112 -
    // a degenerate rank equal to one of those would hide a shape-transposition bug.
    let lora_cfg_ = Qwen35Config { lora: Some(lora_cfg(5, 8.0)), ..Qwen35Config::tiny() };
    // Fresh init for the LoRA-extended param set, then overwrite with the same
    // base weights used above - exactly what a `qwen35::finetune::finetune`
    // would do when seeding from a checkpoint.
    let mut init: HashMap<String, Vec<f32>> = qwen35::init::init_weights(&lora_cfg_, 11);
    for (k, v) in &base_init {
        init.insert(k.clone(), v.clone());
    }

    let t = base_cfg.block_size;
    let x: Vec<u32> = (0..t).map(|i| (i * 5 + 1) % base_cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 5 + 2) % base_cfg.vocab).collect();

    let trained = Qwen35::new_train_on(Gpu::new(pipelines()), lora_cfg_.clone(), 1, t, &init);
    trained.set_batch(&x, &y);
    // Move the adapters off the B=0 init so the fold is non-trivial.
    for step in 1..=8 {
        trained.zero_grads();
        trained.forward();
        trained.backward();
        trained.adamw_step(step, 5e-2, 0.0, Some(1.0), 1.0);
        trained.poll_wait();
    }
    let logits_before_save = trained.logits_all(&x);

    let path = tmp("adapter").join("qwen35_lora.safetensors");
    trained.save(path.to_str().unwrap());

    let c = checkpoint::load(path.to_str().unwrap());
    let reloaded_cfg = Qwen35Config::from_json(&c.header["config"]);
    assert!(reloaded_cfg.lora.is_some(), "the lora field must survive the checkpoint round-trip");
    let tensors = c.by_role("");
    let reloaded = Qwen35::new_on(Gpu::new(pipelines()), reloaded_cfg, 1, t, &tensors);
    let logits_after_reload = reloaded.logits_all(&x);

    let base = Qwen35::new_on(Gpu::new(pipelines()), base_cfg, 1, t, &base_init);
    let logits_base = base.logits_all(&x);

    // The reloaded checkpoint must reproduce the trained model, not the base.
    let close = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f32>() / a.len() as f32 };
    let reload_err = close(&logits_before_save, &logits_after_reload);
    let base_diff = close(&logits_before_save, &logits_base);

    assert!(reload_err < 1e-3, "reloaded LoRA checkpoint does not reproduce the trained model: mean abs diff {reload_err:.6}");
    assert!(
        base_diff > 1e-2,
        "training did not move logits away from the base model (mean abs diff {base_diff:.6}); \
         the gate above would pass even if LoRA adapters are silently dropped at load"
    );
}

/// A checkpoint saved before LoRA support existed (no `lora` key in its
/// config JSON) must still load as a plain (non-LoRA) model, not error or
/// silently invent a rank.
#[test]
fn checkpoint_without_lora_key_loads_as_plain_model() {
    if skip() {
        return;
    }
    let cfg = Qwen35Config::tiny();
    let init = qwen35::init::init_weights(&cfg, 3);
    let t = cfg.block_size;
    let m = Qwen35::new_on(Gpu::new(pipelines()), cfg.clone(), 1, t, &init);

    let path = tmp("no_lora").join("qwen35_base.safetensors");
    m.save(path.to_str().unwrap());

    let c = checkpoint::load(path.to_str().unwrap());
    assert!(c.header["config"].get("lora").is_none(), "a non-LoRA save must not emit a `lora` key at all");
    let reloaded_cfg = Qwen35Config::from_json(&c.header["config"]);
    assert!(reloaded_cfg.lora.is_none(), "a checkpoint with no `lora` key must load as lora: None");
}
