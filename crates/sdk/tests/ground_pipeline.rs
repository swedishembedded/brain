// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "vision")]

//! End-to-end coverage of `GroundingPipeline::from_pretrained` against a
//! local, synthetic, fully offline models-directory fixture - mirroring
//! `tests/tts_pipeline.rs`'s established pattern for this exact class of
//! ceiling.
//!
//! ## Why this stops short of a successful `.ground(...)`
//!
//! `florence2::spec::Florence2Spec::classify` only reads `config.json`'s
//! `model_type` field (header-only - see that module's own doc), so
//! resolution succeeds against a fixture carrying no real vision-tower/BART
//! tensor content at all. `FlorenceSession::load` then fails CLEANLY
//! (`Error::Backend`, never a panic) reading that incomplete tensor set -
//! `florence2::import::build_param_source`'s own `source.get(&name)
//! .ok_or_else(...)?` contract, unlike `qwen3vl`'s paramstore upload path
//! (`tests/vlm_pipeline.rs`'s own documented ceiling). So this test proves
//! construction through to that clean failure, the same "real resolution,
//! clean failure on fake content" shape `tests/tts_pipeline.rs`/
//! `tests/video_pipeline.rs` already establish.

use std::path::{Path, PathBuf};

fn scratch_root(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-sdk-ground-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn with_models_dir<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", root);
    let out = f();
    std::env::remove_var("BRAIN_MODELS_DIR");
    out
}

const TOY_VOCAB: usize = 64;

fn write_tokenizer_json(path: &Path) {
    let vocab: serde_json::Map<String, serde_json::Value> = (0..TOY_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
    std::fs::write(path, serde_json::to_vec(&serde_json::json!({"version": "1.0", "model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
}

/// The exact three files `florence2::spec::Florence2Spec::classify`/
/// `florence2::caps::Florence2Provider::RELEASE_FILES` require -
/// `model.safetensors` carries one dummy tensor (this pipeline never reaches
/// a real weight read - see this file's module doc), never florence2's real
/// vision-tower/BART manifest.
fn write_florence2_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"model_type": "florence2"})).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
    write_tokenizer_json(&dir.join("tokenizer.json"));
}

/// `resolve()`'s own root inference needs a second, unrelated vendor - a
/// SIBLING of the fixture's own vendor directory, under `root` - to land on
/// `root` rather than collapsing onto the fixture's single repo directory,
/// the same requirement every other spec fixture in this workspace
/// documents.
fn write_unrelated_sibling(root: &Path) {
    let dir = root.join("other-vendor");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("unrelated.gguf"), b"not a real gguf").unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::GroundingPipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// The full facade path against a real, content-classified local fixture,
/// with no network access at any point: reference parses,
/// `loader::resolve_structured` finds the checkpoint via
/// `Florence2Spec::classify`'s own `config.json` `model_type` check, and
/// `GroundingPipelineBuilder::load` builds a real `GroundingPipeline`.
/// `.ground(...)` then reaches `FlorenceSession::load`'s tensor-manifest
/// check, which fails cleanly on the fixture's deliberately incomplete
/// tensor set rather than resolution itself failing.
#[test]
fn from_pretrained_resolves_florence2_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("resolve");
    write_florence2_checkpoint(&root.join("microsoft").join("Florence-2-base"));
    write_unrelated_sibling(&root);

    let pipe = with_models_dir(&root, || brain::GroundingPipeline::from_pretrained("microsoft/Florence-2-base"));
    let err = pipe.expect_err("a checkpoint with fake tensor content must fail at construction, not at resolution").to_string();
    assert!(!err.is_empty(), "must name what went wrong");
}

/// A locally-present repo whose OWN `brain.manifest.json` declares neither
/// the real role name (the generic single-role `{"weights": ...}` shape
/// every other spec's own "missing role" test fixture uses) surfaces as
/// `Error::Missing`, naming the arch - proven the same way
/// `tests/depth_pipeline.rs`/`tests/detection_pipeline.rs` prove it for
/// their own single-role specs.
#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    let dir = root.join("microsoft").join("Florence-2-empty-test");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": "microsoft/Florence-2-empty-test", "family": "irrelevant", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
    write_unrelated_sibling(&root);

    let err = with_models_dir(&root, || brain::GroundingPipeline::from_pretrained("microsoft/Florence-2-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => assert_eq!(m.arch, "florence2"),
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}
