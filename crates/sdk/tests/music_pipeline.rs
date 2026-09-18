// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "audio")]

//! End-to-end coverage of `MusicPipeline::from_pretrained`'s resolution
//! path, fully offline.
//!
//! ## Why this stops short of a real resolved fixture
//!
//! `minimaxmusic3::spec::MinimaxMusic3Spec` classifies six roles across five
//! genuinely different on-disk shapes: `language_model` is an ordinary real
//! `Qwen3ForCausalLM` HF directory, while `depth_decoder`/`condition_encoder`/
//! `transformer`/`vocoder` are brain-native components with no `config.json`
//! of their own, classified by their own safetensors tensor NAMES read
//! directly (`crates/minimaxmusic3/src/spec.rs`'s own module doc). Building
//! a synthetic fixture that reaches a real `Resolved` outcome for all six
//! would mean reverse-engineering four separate tensor-name manifests this
//! milestone does not take on - the same class of ceiling
//! `tests/vlm_pipeline.rs`'s own module doc documents for qwen3vl's
//! construction path, one step earlier here (resolution itself, not
//! construction past it). What IS proven here is the ceiling every other
//! resolver-backed pipeline in this crate proves at minimum: an unparseable
//! model id is refused cleanly, and a store with nothing minimaxmusic3-shaped
//! in it comes back a clean, named `Error::Missing` naming every one of the
//! six roles - never a panic, and never a network attempt.

use std::path::{Path, PathBuf};

fn scratch_root(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-sdk-music-pipeline-{tag}-{}-{n}", std::process::id()));
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

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::MusicPipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// A locally-present repo whose OWN `brain.manifest.json` declares neither
/// of the six real roles (the generic single-role `{"weights": ...}` shape
/// every other spec's own "missing role" test fixture uses - the same
/// `mark_locally_present`-style trick `tests/vlm_pipeline.rs`'s own
/// `from_pretrained_names_the_missing_role_when_there_is_no_tokenizer` uses)
/// satisfies `Store::local`'s generic (family-agnostic) compound-manifest
/// check with no network access - without it,
/// `MusicPipelineBuilder::load`'s resolve-first ordering (see that
/// function's own doc for why it tries resolution before `Store::local`/
/// `plan`) would still fall through to a real hub network attempt once
/// resolution reports `Missing`, since nothing on disk here satisfies
/// `MinimaxMusic3Spec::classify` either.
#[test]
fn from_pretrained_names_every_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    let dir = root.join("MiniMaxAI").join("MiniMax-Music3-empty-test");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": "MiniMaxAI/MiniMax-Music3-empty-test", "family": "irrelevant", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();

    let err = with_models_dir(&root, || brain::MusicPipeline::from_pretrained("MiniMaxAI/MiniMax-Music3-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "minimaxmusic3");
            let roles: Vec<&str> = m.roles.iter().map(|r| r.role.as_str()).collect();
            for want in ["language_model", "depth_decoder", "condition_encoder", "transformer", "vocoder", "tokenizer"] {
                assert!(roles.contains(&want), "{roles:?} missing {want}");
            }
        }
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}
