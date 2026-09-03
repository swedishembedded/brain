// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The path `brain qwen35 infer --adapter FILE` takes to serve a LoRA
//! adapter: fold it into the base tensors (`qwen35::lora::fold_adapter_into`,
//! already proven exact against the live unfolded forward by
//! `crates/qwen35/tests/lora_adapter_file.rs`), then build the SAME
//! `Qwen35::new_on` KV-cache-capable model `brain qwen35 infer` already uses
//! for a plain (non-adapted) checkpoint -- no separate decode-only
//! constructor exists for this crate the way `qwen3::Qwen::from_tensors_
//! decode` does, because `new_on` already IS the decode/KV-cache path here.
//! Mirrors `qwen3/tests/lora_serve_fold.rs`.
//!
//! Two things must both hold: (1) greedy decode from the folded tensors must
//! match the live trained model's greedy decode exactly (the folded-serve
//! path is behaviorally identical to training), and (2) that decode must
//! genuinely DIVERGE from the same base served with alpha forced away (i.e.
//! the unfolded base with no adapter at all) -- the gate a fold that silently
//! no-ops (wrong tensor names, scale of zero, wrong sign) would still pass if
//! only (1) ran.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-lora-serve-fold-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn folded_serve_matches_live_training_and_diverges_from_the_unadapted_base() {
    if skip() {
        return;
    }

    let base_cfg = Qwen35Config::tiny();
    let base_init = qwen35::init::init_weights(&base_cfg, 21);
    // rank=3, coprime with tiny()'s head_dim=40/d_model=96, avoiding a degenerate rank.
    let lora_cfg_ = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let mut init: HashMap<String, Vec<f32>> = qwen35::init::init_weights(&lora_cfg_, 21);
    for (k, v) in &base_init {
        init.insert(k.clone(), v.clone());
    }
    let t = base_cfg.block_size;
    let x: Vec<u32> = (0..t).map(|i| (i * 7 + 1) % base_cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 7 + 2) % base_cfg.vocab).collect();
    let trained = Qwen35::new_train_on(Gpu::new(pipelines()), lora_cfg_, 1, t, &init);
    trained.set_batch(&x, &y);
    for step in 1..=8 {
        trained.zero_grads();
        trained.forward();
        trained.backward();
        trained.adamw_step(step, 5e-2, 0.0, Some(1.0), 1.0);
        trained.poll_wait();
    }

    let adapter_path = tmp("adapter").join("adapter.safetensors");
    qwen35::lora::save_adapter(adapter_path.to_str().unwrap(), &trained, "test/adapter", "test/base", None).unwrap();

    // Exactly what `checkpoint::load(weights).by_role("")` returns, feeding
    // `crates/cli/src/qwen35_cli.rs`'s `infer` -- the real serving load path.
    let mut base_tensors: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, _) in trained.cfg.param_list() {
        if !name.ends_with(".lora_a") && !name.ends_with(".lora_b") {
            base_tensors.insert(name.clone(), trained.read_weight(&name));
        }
    }
    let unadapted_tensors = base_tensors.clone();

    qwen35::lora::fold_adapter_into(&mut base_tensors, adapter_path.to_str().unwrap()).expect("fold_adapter_into");

    let prompt = &x[..6];
    let live_greedy = qwen35::sample::generate_kv(&trained, prompt, 6, 0.0, 0, 1.0, &[], &mut Rng::new(1));

    // The folded-serve path -- `Qwen35::new_on` over the folded tensor map,
    // same constructor `brain qwen35 infer` uses for a plain checkpoint.
    let served_cfg = Qwen35Config::tiny(); // lora: None -- a plain, folded inference model
    let served = Qwen35::new_on(Gpu::new(pipelines()), served_cfg, 1, t, &base_tensors);
    let served_greedy = qwen35::sample::generate_kv(&served, prompt, 6, 0.0, 0, 1.0, &[], &mut Rng::new(1));
    assert_eq!(
        served_greedy, live_greedy,
        "greedy generation from the folded-adapter tensors diverged from the live trained model's greedy generation"
    );

    // Non-triviality: the SAME base served WITHOUT the adapter fold (the
    // "alpha forced to 0" case) must produce different output -- otherwise
    // the fold above could have silently been a no-op and this test would
    // never have caught it.
    let unadapted = Qwen35::new_on(Gpu::new(pipelines()), Qwen35Config::tiny(), 1, t, &unadapted_tensors);
    let unadapted_greedy = qwen35::sample::generate_kv(&unadapted, prompt, 6, 0.0, 0, 1.0, &[], &mut Rng::new(1));
    assert_ne!(
        served_greedy, unadapted_greedy,
        "folded-adapter serving produced the SAME output as the unadapted base -- the adapter is not doing anything"
    );
}
