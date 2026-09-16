// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `audio` surface, so it compiles only with it -
// the same reason `tests/image_pipeline.rs` gates itself on `image`.
#![cfg(feature = "audio")]

//! End-to-end coverage of `TranscribePipeline::from_pretrained`'s resolution
//! path against a real local, synthetic, fully offline fixture reproducing
//! `crates/qwen3asr/src/spec.rs`'s own (private) classification schema (an
//! `HfDir` whose `config.json` declares `architectures:
//! ["Qwen3ASRForConditionalGeneration"]`) - mirroring `tests/
//! image_pipeline.rs`'s established pattern.
//!
//! ## Why this stops short of a successful `.transcribe(...)`
//!
//! `qwen3asr::caps::QwenAsrProvider::load` needs both a real checkpoint
//! (`Qwen3Asr::from_hf_windowed`) and a real tokenizer
//! (`data::qwen_tokenizer::QwenBpe::from_dir`) - the same two real-content
//! ceilings `TextGenerationPipeline`'s and `EmbeddingPipeline`'s own tests
//! document. So this test proves resolution through to `QwenAsrProvider::load`
//! being reached with the right directory, then a clean `Error::Backend`,
//! never a panic.

use std::path::{Path, PathBuf};

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

fn scratch_root(tag: &str) -> Scratch {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-sdk-transcribe-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Scratch(dir)
}

fn with_models_dir<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", root);
    let out = f();
    std::env::remove_var("BRAIN_MODELS_DIR");
    out
}

fn mark_locally_present(root: &Path, vendor: &str, repo: &str, family: &str) {
    let dir = root.join(vendor).join(repo);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": format!("{vendor}/{repo}"), "family": family, "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
}

/// The exact declared-architecture schema `Qwen3AsrSpec::classify` reads,
/// plus a loose safetensors shard so a real `inventory::scan` collapses the
/// directory into an `HfDir` record at all (the same requirement `tests/
/// forecast_pipeline.rs`'s kronos fixture documents).
fn write_qwen3asr_hfdir(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let config = serde_json::json!({"architectures": ["Qwen3ASRForConditionalGeneration"]});
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
}

#[test]
fn from_pretrained_resolves_qwen3asr_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("qwen3asr");
    write_qwen3asr_hfdir(&root.join("Qwen").join("Qwen3-ASR-1.7B"));
    mark_locally_present(&root, "local", "asr-sdk-test", "qwen3asr");

    let err = with_models_dir(&root, || brain::TranscribePipeline::from_pretrained("local/asr-sdk-test").unwrap_err());
    match &err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from incomplete checkpoint/tokenizer construction, got {other:?}"),
    }
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::TranscribePipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}
