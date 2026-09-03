// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen35::lora::save_adapter` must write ONLY the `.lora_a`/`.lora_b`
//! tensors (not the frozen base -- that's the whole point of an adapter file
//! being small), with a `ModelCard` a consumer can use to find
//! rank/alpha/targets and reload it. `fold_adapter_into` must reproduce the
//! LIVE (unfolded) LoRA forward exactly, so serving a folded base is
//! equivalent to serving the trained model. Mirrors
//! `qwen3/tests/lora_adapter_file.rs` -- the qwen3 precedent this port
//! follows -- adapted to `Qwen35`'s hybrid GDN+GQA+dense-MLP shapes and its
//! own `Qwen35::new_train_on`/`new_on` GPU-taking constructors.

use std::collections::HashMap;

use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen35-lora-adapter-file-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A tiny trained LoRA model (rank 5, coprime with `tiny()`'s
/// head_dim=40/d_model=96/intermediate=112, to avoid a degenerate rank
/// hiding a shape-transposition bug) plus the input it was trained on.
fn trained_model() -> (Qwen35, Vec<u32>) {
    let base_cfg = Qwen35Config::tiny();
    let base_init = qwen35::init::init_weights(&base_cfg, 11);
    let lora_cfg_ = Qwen35Config { lora: Some(lora_cfg(5, 8.0)), ..Qwen35Config::tiny() };
    let mut init: HashMap<String, Vec<f32>> = qwen35::init::init_weights(&lora_cfg_, 11);
    for (k, v) in &base_init {
        init.insert(k.clone(), v.clone());
    }
    let t = base_cfg.block_size;
    let x: Vec<u32> = (0..t).map(|i| (i * 7 + 1) % base_cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 7 + 2) % base_cfg.vocab).collect();
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), lora_cfg_, 1, t, &init);
    m.set_batch(&x, &y);
    for step in 1..=8 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 5e-2, 0.0, Some(1.0), 1.0);
        m.poll_wait();
    }
    (m, x)
}

#[test]
fn adapter_file_contains_only_lora_tensors_and_is_far_smaller_than_a_full_checkpoint() {
    if skip() {
        return;
    }
    let (trained, _x) = trained_model();
    let dir = tmp("adapter_only");

    let adapter_path = dir.join("adapter.safetensors");
    qwen35::lora::save_adapter(
        adapter_path.to_str().unwrap(),
        &trained,
        "Qwen/Qwen3.8-27B:swedishembedded-com:generic-sft:latest",
        "Qwen/Qwen3.8-27B",
        Some("sha256:test"),
    )
    .expect("save_adapter");

    let full_path = dir.join("full.safetensors");
    trained.save(full_path.to_str().unwrap());

    let adapter_bytes = std::fs::metadata(&adapter_path).unwrap().len();
    let full_bytes = std::fs::metadata(&full_path).unwrap().len();
    assert!(
        adapter_bytes * 5 < full_bytes,
        "adapter file ({adapter_bytes}B) should be far smaller than the full checkpoint ({full_bytes}B) -- \
         a regression back to saving every param would defeat the point of an adapter-only save"
    );

    let st = checkpoint::st::load_safetensors(adapter_path.to_str().unwrap()).expect("load adapter safetensors");
    assert!(
        st.tensors.keys().all(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")),
        "adapter file must contain ONLY lora_a/lora_b tensors, found: {:?}",
        st.tensors.keys().collect::<Vec<_>>()
    );
    let card = st.card().expect("adapter file must carry a ModelCard");
    assert_eq!(card.family, "qwen35", "family tag must not collide with the MoE sibling's own \"qwen35moe\" tag");
    let a = card.adapter.expect("card must carry an Adapter descriptor");
    assert_eq!(a.kind, "lora");
    assert_eq!(a.rank, Some(5));
    assert_eq!(a.alpha, Some(8.0));
    assert_eq!(a.base.as_deref(), Some("Qwen/Qwen3.8-27B"));
    assert_eq!(a.dataset_id.as_deref(), Some("sha256:test"));
    assert_eq!(card.variant_of.as_deref(), Some("Qwen/Qwen3.8-27B"));
}

#[test]
fn folding_the_adapter_into_the_base_reproduces_the_live_lora_forward() {
    if skip() {
        return;
    }
    let (trained, x) = trained_model();
    let live_logits = trained.logits_all(&x);

    let adapter_path = tmp("fold").join("adapter.safetensors");
    qwen35::lora::save_adapter(adapter_path.to_str().unwrap(), &trained, "test/adapter", "test/base", None).unwrap();

    // The base tensors an inference load would see: every non-adapter param
    // (exactly what `checkpoint::load(weights).by_role("")` returns). Note
    // `Qwen35::param_names()` on a LoRA build lists ONLY the trainable
    // adapters (see `lora_freezes_base.rs`), never the frozen base -- so the
    // base names come from the config's own `param_list()` instead, exactly
    // like `qwen35::finetune::finetune`'s init map does.
    let mut base_tensors: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, numel) in trained.cfg.param_list() {
        if !name.ends_with(".lora_a") && !name.ends_with(".lora_b") {
            let w = trained.read_weight(&name);
            assert_eq!(w.len(), numel, "{name}: read_weight length disagrees with param_list");
            base_tensors.insert(name, w);
        }
    }

    qwen35::lora::fold_adapter_into(&mut base_tensors, adapter_path.to_str().unwrap()).expect("fold_adapter_into");

    let base_cfg = Qwen35Config::tiny(); // lora: None -- a plain, folded inference model
    let t = base_cfg.block_size;
    let folded = Qwen35::new_on(Gpu::new(pipelines()), base_cfg, 1, t, &base_tensors);
    let folded_logits = folded.logits_all(&x);

    let mean_abs_diff: f32 =
        live_logits.iter().zip(&folded_logits).map(|(a, b)| (a - b).abs()).sum::<f32>() / live_logits.len() as f32;
    assert!(
        mean_abs_diff < 1e-3,
        "folded-base forward does not match the live unfolded LoRA forward: mean abs diff {mean_abs_diff:.6}"
    );
}
