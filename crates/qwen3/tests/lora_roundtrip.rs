// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A LoRA fine-tune must change what the model actually outputs after a save +
//! reload cycle -- not just in the live process. Before this test, a LoRA
//! checkpoint's adapters were written to disk but silently dropped on the next
//! `load_inference`: `QwenConfig::to_json` never emitted `lora`, so `from_json`
//! always rebuilt the param list without `*.lora_a`/`*.lora_b`, and `lora_fwd`
//! never dispatched.

use std::collections::HashMap;

use qwen3::{LoraCfg, Qwen, QwenConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-roundtrip-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// rank=3 is coprime with tiny()'s head_dim=8 and d_model=16 (a degenerate
/// rank equal to head_dim or d_model would hide a whole
/// shape-transposition bug class).
#[test]
fn lora_adapter_survives_save_and_reload() {
    if skip() {
        eprintln!("skip: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let base_cfg = QwenConfig::tiny();
    let base_init = qwen3::init_weights(&base_cfg, 11);

    let lora_cfg = QwenConfig { lora: Some(LoraCfg::attn(3, 6.0)), ..QwenConfig::tiny() };
    // Fresh init for the LoRA-extended param set, then overwrite with the same
    // base weights used above -- exactly what `qwen3::finetune::finetune` does
    // when seeding from a checkpoint.
    let mut init: HashMap<String, Vec<f32>> = qwen3::init_weights(&lora_cfg, 11);
    for (k, v) in &base_init {
        init.insert(k.clone(), v.clone());
    }

    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();

    let trained = Qwen::new(lora_cfg.clone(), 1, 12, &init);
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

    let path = tmp("adapter").join("qwen_lora.safetensors");
    trained.save(path.to_str().unwrap());

    let reloaded = Qwen::load_inference(path.to_str().unwrap(), 1, 12);
    let logits_after_reload = reloaded.logits_all(&x);

    let base = Qwen::new(base_cfg, 1, 12, &base_init);
    let logits_base = base.logits_all(&x);

    // The reloaded checkpoint must reproduce the trained model, not the base.
    let close = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f32>() / a.len() as f32
    };
    let reload_err = close(&logits_before_save, &logits_after_reload);
    let base_diff = close(&logits_before_save, &logits_base);

    assert!(
        reload_err < 1e-3,
        "reloaded LoRA checkpoint does not reproduce the trained model: mean abs diff {reload_err:.6}"
    );
    assert!(
        base_diff > 1e-2,
        "training did not move logits away from the base model (mean abs diff {base_diff:.6}); \
         the gate above would pass even if LoRA adapters are silently dropped at load"
    );
}

/// A checkpoint saved before this fix (no `lora` key in its config JSON) must
/// still load as a plain (non-LoRA) model, not error or silently invent ranks.
#[test]
fn checkpoint_without_lora_key_loads_as_plain_model() {
    if skip() {
        eprintln!("skip: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 3);
    let m = Qwen::new(cfg.clone(), 1, 12, &init);

    let path = tmp("no_lora").join("qwen_base.safetensors");
    m.save(path.to_str().unwrap());

    let c = checkpoint::load(path.to_str().unwrap());
    assert!(
        c.header["config"].get("lora").is_none(),
        "a non-LoRA save must not emit a `lora` key at all"
    );
    let reloaded_cfg = QwenConfig::from_json(&c.header["config"]);
    assert!(reloaded_cfg.lora.is_none(), "a checkpoint with no `lora` key must load as lora: None");
}
