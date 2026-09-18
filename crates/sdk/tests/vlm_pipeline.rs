// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "multimodal")]

//! End-to-end coverage of `VisionLanguagePipeline::from_pretrained`'s
//! resolution path, fully offline.
//!
//! ## Why this stops short of a successful `.ask(...)`, or even a clean
//! `Error::Backend` from a resolved-but-fake checkpoint
//!
//! Every OTHER resolver-backed pipeline in this crate (`tests/
//! tts_pipeline.rs`, `tests/video_pipeline.rs`, ...) proves resolution
//! reaches a real, CONSTRUCTED pipeline against a fixture with fake tensor
//! CONTENT but real tensor NAMES/shapes, then shows the first real call
//! fails cleanly (`Error::Backend`) on that fake content. `qwen3vl`'s own
//! construction path does not offer that ceiling: `Qwen3Vl::from_hf`'s
//! weight upload (`brain_paramstore`) panics - `missing init weight
//! tok.weight: ... not present in this source` - rather than returning a
//! clean error, the moment a declared tensor is absent from the source
//! file, confirmed empirically while building this milestone's own fixture
//! (a full, real tensor manifest for both the vision tower AND the text
//! decoder is out of scope for proving SDK-level resolution). This mirrors
//! `TextGenerationPipelineBuilder::load`'s own documented reason for
//! pre-checking `WeightReader::open` before ever calling the equally
//! panicking `Qwen::load_inference` - the difference is `text.rs` can stop
//! BEFORE that call from the public API (checkpoint-open, then a clean
//! tokenizer-precedence error); `VisionLanguagePipeline::from_pretrained`
//! has no such earlier public stopping point, since resolution and
//! construction are one call. So what IS proven here is resolution itself:
//! a real content-classifiable checkpoint directory resolves, and a
//! checkpoint that fails `Qwen3VlSpec::validate` (no sibling
//! `tokenizer.json`) comes back a clean, named `Error::Missing` - never a
//! panic - exactly the same ceiling `crates/qwen3vl/src/spec.rs`'s own
//! `a_checkpoint_with_no_tokenizer_fails_validate_not_a_silent_resolve`
//! test already established at the spec level.

use std::path::{Path, PathBuf};

fn scratch_root(tag: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-sdk-vlm-pipeline-{tag}-{}-{n}", std::process::id()));
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

/// `crates/qwen3vl/src/spec.rs::HF_ARCHITECTURE`'s own required
/// `architectures[0]` value - a real, content-classifiable checkpoint
/// directory needs no more than this to satisfy `Qwen3VlSpec::classify`
/// (header/config-only, per that module's own doc).
const HF_ARCHITECTURE: &str = "Qwen3VLForConditionalGeneration";

fn write_checkpoint_dir(dir: &Path, with_tokenizer: bool) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": [HF_ARCHITECTURE]})).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![1], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
    if with_tokenizer {
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    }
}

/// `resolve()`'s own root inference needs a second, unrelated vendor - a
/// SIBLING of the fixture's own vendor directory, under `root` - to land on
/// `root` rather than collapsing onto the fixture's single repo directory,
/// the same requirement `crates/qwen3vl/src/spec.rs`'s own fixture (and
/// every other spec fixture in this workspace) documents.
fn write_unrelated_sibling(root: &Path) {
    let dir = root.join("other-vendor");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("unrelated.gguf"), b"not a real gguf").unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::VisionLanguagePipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// A checkpoint with real, classifiable content but no sibling
/// `tokenizer.json` fails `Qwen3VlSpec::validate` and comes back
/// `Error::Missing` - never a panic, and never a network attempt.
///
/// `Store::local`'s own generic (family-agnostic) compound-manifest check
/// is satisfied by a SEPARATE, harmless `brain.manifest.json` colocated in
/// the same directory (the same "any valid manifest, any role name, just
/// needs its declared file to exist" trick `tests/text_pipeline.rs`'s own
/// `a_relative_nonexistent_path_that_looks_like_a_hub_ref_is_tried_as_one`
/// test uses) - without it, `VisionLanguagePipelineBuilder::load`'s
/// resolve-first fix (see that function's own doc for why it tries
/// resolution before `Store::local`/`plan`) would still fall through to a
/// real hub network attempt once resolution reports `Missing`, since a raw
/// HF checkpoint directory alone satisfies neither of `Store::local`'s own
/// recognized shapes. Whether the manifest's presence also collapses this
/// directory's own per-file scan (`crates/modelstore/src/inventory.rs`'s
/// own `compound_records` doc) and hides it from `Qwen3VlSpec::classify`
/// entirely does not change the observed outcome either way - resolution
/// is `Missing` either because `classify` fails to find a fresh `weights`
/// candidate at all, or because `validate` rejects the one it does not
/// see a tokenizer for.
#[test]
fn from_pretrained_names_the_missing_role_when_there_is_no_tokenizer() {
    let root = scratch_root("missing");
    let dir = root.join("Qwen").join("Qwen3-VL-notok-test");
    write_checkpoint_dir(&dir, false);
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": "Qwen/Qwen3-VL-notok-test", "family": "irrelevant", "roles": {"weights": "model.safetensors"}})).unwrap(),
    )
    .unwrap();
    write_unrelated_sibling(&root);

    let err = with_models_dir(&root, || brain::VisionLanguagePipeline::from_pretrained("Qwen/Qwen3-VL-notok-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => assert_eq!(m.arch, "qwen3vl"),
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}

/// `DownloadPolicy::Offline` never reaches the network -- same proof shape
/// as `crates/sdk/tests/image_pipeline.rs`'s own
/// `download_policy_offline_never_touches_the_network`: point `HfHub` at a
/// loopback port nothing listens on, and show a reference resolving neither
/// locally nor from any real hub still comes back the resolver's own clean
/// `Error::Missing`, never a connection-error-flavored `Error::Download`.
#[test]
fn download_policy_offline_never_touches_the_network() {
    let root = scratch_root("offline");
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", &root);
    std::env::set_var("BRAIN_HUB_ENDPOINT", "http://127.0.0.1:1");

    let err = brain::VisionLanguagePipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}
