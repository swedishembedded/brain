// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `text` surface, so it compiles only with it -
// the same reason `tests/image_pipeline.rs` gates itself on `image`.
#![cfg(feature = "text")]

//! End-to-end coverage of `TextGenerationPipeline::builder(...).load()`'s
//! construction sequence, against a real (if minimal) local checkpoint, with
//! no network access anywhere - what's under test here is the
//! checkpoint-open -> config-parse -> tokenizer-precedence sequence itself,
//! plus the local-path-vs-hub-id split `resolve_hub_weights` now does (see
//! `crates/sdk/src/text.rs`'s own module doc). The hub-id-resolves-to-a-real-
//! model case is NOT covered here: unlike `Qwen35Spec`'s own tests (which
//! stop at `resolve()`, never a full model construction), reaching
//! `Qwen::load_inference` with fake GGUF tensor content is unproven
//! territory this file does not take on either - what IS proven is that a
//! hub-shaped id reaches the resolver at all, and comes back a clean,
//! named error rather than silently falling back to "not found" the way a
//! genuinely-missing local path does.
//!
//! ## Why this stops short of a successful `.generate(...)`
//!
//! `QwenConfig::from_json`'s own defaults happen to equal `QwenConfig::tiny()`
//! exactly (both default to `vocab: 23, n_layers: 2, d_model: 16, ...`), so
//! an empty `{}` config header is enough to prove the checkpoint opens and
//! parses - genuinely cheaper than `ImagePipeline`'s fixtures, which need
//! real classifiable tensor shapes. What stays out of reach is the
//! TOKENIZER: `data::qwen_tokenizer::QwenBpe` has no synthetic/in-memory
//! constructor, only `from_file`/`from_dir`/`from_gguf`/`from_json_bytes`
//! reading a real HF `tokenizer.json` schema (vocab map, merges, chat
//! template) - reproducing a valid minimal one is a real, separate fixture
//! project this milestone does not take on. So these tests prove
//! construction through checkpoint-open and config-parse, then a clean,
//! named `Error::MissingArgument` at the tokenizer-resolution step - never a
//! panic, and never reaching the (much heavier) `Qwen::load_inference` call.

use std::path::{Path, PathBuf};

/// A fixture checkpoint file that deletes itself when the test ends.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

/// A minimal, real, valid safetensors checkpoint: one dummy tensor (this
/// pipeline never reaches a real weight read - see this file's module doc)
/// and an EMPTY `{}` config, which `QwenConfig::from_json` resolves to
/// exactly `QwenConfig::tiny()` via its own defaults.
fn tiny_checkpoint(tag: &str) -> Scratch {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("brain-sdk-text-pipeline-{tag}-{}-{n}.safetensors", std::process::id()));
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
    Scratch(path)
}

/// A string that is neither an existing local file NOR a valid
/// `<vendor>/<repo>` hub reference (a leading `/` makes `ModelRef::parse`
/// see an empty vendor segment) is `Error::ModelNotFound`, naming both
/// things it isn't - see `resolve_hub_weights`'s own doc for why a
/// non-local-file string is tried as a hub id at all.
#[test]
fn builder_with_a_nonexistent_weights_path_fails_cleanly() {
    let err = brain::TextGenerationPipeline::from_pretrained("/does/not/exist.safetensors").unwrap_err();
    match &err {
        brain::Error::ModelNotFound(msg) => assert!(msg.contains("does/not/exist"), "{msg}"),
        other => panic!("expected a clean Error::ModelNotFound naming the bad path, got {other:?}"),
    }
}

/// A RELATIVE nonexistent path with exactly one `/` DOES parse as a
/// syntactically valid hub reference (`ModelRef::parse`'s own grammar) - so
/// this one instead reaches the resolver, through `resolve_hub_weights`,
/// rather than being silently treated as "local file not found." Proves the
/// `Path::is_file()` local/hub split really does check the filesystem
/// rather than guessing from shape alone.
///
/// `mark_locally_present` satisfies `Store::local`'s OWN generic
/// (family-agnostic) compound-manifest check with a role name
/// (`Qwen3Spec` does not recognize) - the SAME trick
/// `tests/depth_pipeline.rs`'s own `mark_locally_present` uses - so
/// `resolve_hub_weights` never reaches `plan()`/the hub network at all: it
/// tries `qwen3::spec::Qwen3Spec::classify` first (finds nothing, since
/// this fixture has no real GGUF/safetensors content), gets `Missing`, then
/// finds `Store::local` already satisfied and returns that `Missing`
/// directly rather than attempting a fetch.
#[test]
fn a_relative_nonexistent_path_that_looks_like_a_hub_ref_is_tried_as_one() {
    let _serial = brain_testutil::env_lock();
    let root = std::env::temp_dir().join(format!("brain-sdk-text-pipeline-hub-missing-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let dir = root.join("does-not-exist").join("qwen3-toy.safetensors-repo");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": "does-not-exist/qwen3-toy.safetensors-repo", "family": "qwen3", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();

    std::env::set_var("BRAIN_MODELS_DIR", &root);
    let err = brain::TextGenerationPipeline::from_pretrained("does-not-exist/qwen3-toy.safetensors-repo").unwrap_err();
    std::env::remove_var("BRAIN_MODELS_DIR");
    std::fs::remove_dir_all(&root).ok();

    match &err {
        brain::Error::Missing(m) => assert_eq!(m.arch, "qwen3"),
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}

/// Reaches checkpoint-open and config-parse (a real, if minimal, safetensors
/// file) with no tokenizer given and none embedded (a safetensors file
/// carries no GGUF tokenizer KV) - a caller-programming error, knowable
/// before any real weight read, so it's `Error::MissingArgument` and not an
/// indistinguishable `Error::Backend`.
#[test]
fn load_without_a_tokenizer_names_the_missing_argument_before_touching_real_weights() {
    let ckpt = tiny_checkpoint("no-tokenizer");
    let err = brain::TextGenerationPipeline::from_pretrained(ckpt.to_str().unwrap()).unwrap_err();
    match &err {
        brain::Error::MissingArgument(msg) => assert!(msg.contains(".tokenizer"), "{msg}"),
        other => panic!("expected Error::MissingArgument naming the fix, got {other:?}"),
    }
}

/// An explicit `.tokenizer(path)` that itself does not exist fails cleanly
/// too, proving the builder's tokenizer knob is actually wired to
/// `QwenBpe::from_file` and not silently ignored.
#[test]
fn an_explicit_but_nonexistent_tokenizer_path_fails_cleanly() {
    let ckpt = tiny_checkpoint("bad-tokenizer");
    let err = brain::TextGenerationPipeline::builder(ckpt.to_str().unwrap()).tokenizer("/does/not/exist/tokenizer.json").load().unwrap_err();
    match &err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from the bad tokenizer path, got {other:?}"),
    }
}
