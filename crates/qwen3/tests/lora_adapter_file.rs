// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `qwen3::lora::save_adapter` must write ONLY the `.lora_a`/`.lora_b` tensors
//! (not the frozen base -- that's the whole point of an adapter file being
//! small), with a `ModelCard` a consumer can use to find rank/alpha/targets
//! and reload it. `fold_adapter_into` must reproduce the LIVE (unfolded)
//! `lora_fwd` forward exactly, so serving a folded base is equivalent to
//! serving the trained model.

use std::collections::HashMap;

use qwen3::{LoraCfg, Qwen, QwenConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-lora-adapter-file-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn trained_model() -> (Qwen, Vec<u32>) {
    let base_cfg = QwenConfig::tiny();
    let base_init = qwen3::init_weights(&base_cfg, 21);
    let lora_cfg = QwenConfig { lora: Some(LoraCfg::attn(3, 6.0)), ..QwenConfig::tiny() };
    let mut init: HashMap<String, Vec<f32>> = qwen3::init_weights(&lora_cfg, 21);
    for (k, v) in &base_init {
        init.insert(k.clone(), v.clone());
    }
    let x: Vec<u32> = (0..12).map(|i| (i * 7 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 7 + 2) % 23).collect();
    let m = Qwen::new(lora_cfg, 1, 12, &init);
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
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (trained, _x) = trained_model();
    let dir = tmp("adapter_only"); // one tmp() call -- it wipes its directory on every call

    let adapter_path = dir.join("adapter.safetensors");
    qwen3::lora::save_adapter(
        adapter_path.to_str().unwrap(),
        &trained,
        "Qwen/Qwen3-0.6B:swedishembedded-com:generic-sft:latest",
        "Qwen/Qwen3-0.6B",
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
    let a = card.adapter.expect("card must carry an Adapter descriptor");
    assert_eq!(a.kind, "lora");
    assert_eq!(a.rank, Some(3));
    assert_eq!(a.alpha, Some(6.0));
    assert_eq!(a.base.as_deref(), Some("Qwen/Qwen3-0.6B"));
    assert_eq!(a.dataset_id.as_deref(), Some("sha256:test"));
    assert_eq!(card.variant_of.as_deref(), Some("Qwen/Qwen3-0.6B"));
}

#[test]
fn folding_the_adapter_into_the_base_reproduces_the_live_lora_forward() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (trained, x) = trained_model();
    let live_logits = trained.logits_all(&x);

    let adapter_path = tmp("fold").join("adapter.safetensors");
    qwen3::lora::save_adapter(adapter_path.to_str().unwrap(), &trained, "test/adapter", "test/base", None).unwrap();

    // The base tensors an inference load would see: every non-adapter param.
    let mut base_tensors: HashMap<String, Vec<f32>> = trained
        .ps
        .params
        .iter()
        .filter(|(name, _)| !(name.ends_with(".lora_a") || name.ends_with(".lora_b")))
        .map(|(name, _)| (name.clone(), trained.read_weight(name)))
        .collect();

    qwen3::lora::fold_adapter_into(&mut base_tensors, adapter_path.to_str().unwrap()).expect("fold_adapter_into");

    let base_cfg = QwenConfig::tiny(); // no lora: None -- a plain, folded inference model
    let folded = Qwen::new(base_cfg, 1, 12, &base_tensors);
    let folded_logits = folded.logits_all(&x);

    let mean_abs_diff: f32 =
        live_logits.iter().zip(&folded_logits).map(|(a, b)| (a - b).abs()).sum::<f32>() / live_logits.len() as f32;
    assert!(
        mean_abs_diff < 1e-3,
        "folded-base forward does not match the live unfolded LoRA forward: mean abs diff {mean_abs_diff:.6}"
    );
}

/// Train, save a full checkpoint, reload it, and extract the adapter - the
/// order `brain qwen3 finetune --lora` actually does it in, and the one
/// nothing covered.
///
/// `Model::param_names` is the OPTIMISED set: trainable plus offload, never
/// frozen. An inference build freezes everything, adapters included, so a
/// checkpoint reloaded that way carries its adapters and reports none, and
/// `save_adapter` fails with "no .lora_a/.lora_b tensors in the param store"
/// on a file that visibly contains 392 of them. The reload has to be the
/// shape that gives the adapters a trainable role.
#[test]
fn an_adapter_survives_a_full_checkpoint_round_trip() {
    if skip() {
        brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
        return;
    }
    let (trained, _x) = trained_model();
    let dir = tmp("roundtrip");
    let full = dir.join("full.safetensors");
    trained.save(full.to_str().expect("utf-8 path"));
    drop(trained);

    let reader = checkpoint::weightio::WeightReader::open(full.to_str().expect("utf-8 path")).expect("reopen");
    let cfg = QwenConfig::from_json(&reader.config());
    assert!(cfg.lora.is_some(), "the saved config must carry the adapter shape, or nothing below can work");
    let shard = qwen3::Shard::whole(cfg.n_layers as usize);
    let block = cfg.block_size;
    let reloaded = Qwen::new_shard(cfg, 1, block, &reader, true, shard);
    drop(reader);

    let adapter = dir.join("adapter.safetensors");
    qwen3::lora::save_adapter(adapter.to_str().expect("utf-8 path"), &reloaded, "test/adapter", "test/base", None)
        .expect("a reloaded LoRA checkpoint must still yield its adapter");

    let st = checkpoint::st::load_safetensors(adapter.to_str().expect("utf-8 path")).expect("read back");
    assert!(!st.tensors.is_empty(), "the adapter file is empty");
    assert!(
        st.tensors.keys().all(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")),
        "{:?}",
        st.tensors.keys().collect::<Vec<&String>>()
    );
}
