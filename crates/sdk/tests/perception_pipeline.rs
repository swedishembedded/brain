// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "vision")]

//! `PerceptionPipeline::from_pretrained` against local, offline models-directory
//! fixtures: a Mirage checkpoint directory resolves by its own `config.json`
//! (`model_type: "mirage"`) and fails cleanly, never with a panic, when its
//! content is a stand-in; an empty store names the missing role; the offline
//! policy never touches the network. A decoded result on the released weights
//! is covered by the CLI acceptance test (`crates/cli/tests/it/mirage_cli.rs`),
//! which compares the command's JSON with this pipeline's.

use std::path::{Path, PathBuf};

fn scratch_root(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-sdk-perception-{tag}-{}-{n}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A sibling vendor directory so the resolver infers the models root rather than the fixture's repository.
fn write_unrelated_sibling(root: &Path) {
    let dir = root.join("other-vendor");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("unrelated.gguf"), b"not a real gguf").unwrap();
}

fn write_stand_in(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"model_type": "mirage"})).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("mirage.safetensors").to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
    std::fs::write(dir.join("mirage.cact"), b"stand-in").unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::PerceptionPipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

#[test]
fn a_stand_in_checkpoint_resolves_and_then_fails_cleanly_at_load() {
    let root = scratch_root("stand-in");
    write_stand_in(&root.join("Cactus-Compute").join("mirage"));
    write_unrelated_sibling(&root);
    let err = brain::PerceptionPipeline::builder("Cactus-Compute/mirage").models_dir(&root).load().unwrap_err();
    // Resolution found the directory (it is not `Missing`); the import then refused the stand-in's content.
    assert!(matches!(err, brain::Error::Backend(_)), "{err:?}");
    assert!(err.to_string().contains("mirage"), "{err}");
}

#[test]
fn an_empty_store_names_the_missing_role() {
    let root = scratch_root("missing");
    write_unrelated_sibling(&root);
    let err = brain::PerceptionPipeline::builder("Cactus-Compute/mirage").models_dir(&root).download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();
    match &err {
        brain::Error::Missing(m) => assert_eq!(m.arch, "mirage"),
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}
