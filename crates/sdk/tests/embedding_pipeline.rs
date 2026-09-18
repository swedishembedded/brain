// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `vision` surface, so it compiles only with it -
// the same reason `tests/image_pipeline.rs` gates itself on `image`.
#![cfg(feature = "vision")]

//! End-to-end coverage of `EmbeddingPipeline::from_pretrained`'s resolution
//! path against a real local, synthetic, fully offline fixture - mirroring
//! `tests/image_pipeline.rs`'s pattern and, for the exact classification
//! schema, `crates/clip/src/spec.rs`'s own (private) `write_pipeline` test
//! fixture (an SDXL-layout root: `model_index.json` naming
//! `StableDiffusionXLPipeline`, plus four component directories).
//!
//! ## Why this stops short of a successful `.embed(...)`
//!
//! `clip::caps::Session::load` needs a REAL tokenizer under `tokenizer/`
//! (`data::clip_bpe::ClipBpe::from_dir` reads a real vocab/merges schema,
//! the same class of fixture gap `TextGenerationPipeline`'s own tests
//! document for `QwenBpe`). So this test proves resolution all the way to
//! `Session::load` being reached with the right directory - then fails
//! CLEANLY (a typed `brain::Error`, never a panic) on the fixture's
//! deliberately empty tokenizer directories.

use std::path::{Path, PathBuf};

/// A fixture models directory that deletes itself when the test ends.
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
    let dir = std::env::temp_dir().join(format!("brain-sdk-embedding-pipeline-{tag}-{}-{n}", std::process::id()));
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

/// The real SDXL-layout tower root `clip::spec::ClipSpec::classify` looks
/// for: a `model_index.json` naming `StableDiffusionXLPipeline`, plus the
/// four component directories - empty here, since only their EXISTENCE
/// (not their content) decides classification.
fn write_sdxl_tower_root(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model_index.json"), serde_json::to_vec(&serde_json::json!({"_class_name": "StableDiffusionXLPipeline"})).unwrap()).unwrap();
    for c in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"] {
        std::fs::create_dir_all(dir.join(c)).unwrap();
    }
}

#[test]
fn from_pretrained_resolves_clip_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("clip");
    write_sdxl_tower_root(&root.join("stabilityai").join("stable-diffusion-xl-base-1.0"));
    mark_locally_present(&root, "local", "embedding-sdk-test", "clip");

    let err = with_models_dir(&root, || brain::EmbeddingPipeline::from_pretrained("local/embedding-sdk-test").unwrap_err());
    match &err {
        // A `Backend` error this deep means resolution found the "towers"
        // role and `Session::load` was reached - only the fixture's empty
        // tokenizer directories, read last, can still fail.
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from an empty tokenizer directory, got {other:?}"),
    }
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::EmbeddingPipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// The real, confirmed gap `EmbeddingPipelineBuilder::load`'s own doc names:
/// unlike the test above (which uses [`mark_locally_present`] purely to
/// satisfy `Store::local`'s own generic check cheaply, with no bearing on
/// whether the real bug is present), THIS fixture places the checkpoint at
/// EXACTLY the repo path `model_id` itself names, with no separate manifest
/// anywhere else in the store - the same shape a real `hf download
/// stabilityai/stable-diffusion-xl-base-1.0 --local-dir
/// $BRAIN_MODELS_DIR/stabilityai/stable-diffusion-xl-base-1.0` produces.
/// Before the resolve-first fix, this reached `Error::ModelNotFound("...: no
/// config.json in repo")` - `plan()`'s `TransformersRecipe` catch-all looks
/// for a top-level `config.json`, which SDXL's own `model_index.json`-keyed
/// layout does not carry - confirmed empirically while building this fix.
/// Now it resolves and reaches `Session::load`, failing cleanly on the
/// fixture's own empty tokenizer directories instead.
#[test]
fn from_pretrained_resolves_a_real_fixture_at_its_own_named_path_with_no_store_local_shortcut() {
    let root = scratch_root("no-shortcut");
    write_sdxl_tower_root(&root.join("stabilityai").join("stable-diffusion-xl-base-1.0"));

    let err = with_models_dir(&root, || brain::EmbeddingPipeline::from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").unwrap_err());
    match &err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from an empty tokenizer directory, got {other:?}"),
    }
}
