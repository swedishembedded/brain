// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `audio` surface, so it compiles only with it -
// the same reason `tests/transcribe_pipeline.rs` gates itself on `audio`.
#![cfg(feature = "audio")]

//! End-to-end coverage of `TtsPipeline::from_pretrained`'s resolution path
//! against a real local, synthetic, fully offline fixture reproducing
//! `crates/qwen3tts/src/spec.rs`'s own (private) `write_qwen3tts_checkpoint`
//! test fixture shape - mirroring `tests/transcribe_pipeline.rs`'s
//! established pattern for this exact class of ceiling.
//!
//! ## Why this stops short of a successful `.speak(...)`
//!
//! `qwen3tts::pipeline::synth` needs a real tokenizer
//! (`data::qwen_tokenizer::QwenBpe::from_dir`, via `prompt::load_tokenizer`)
//! and real Talker/MTP/codec checkpoints - the same real-content ceiling
//! `TranscribePipeline`'s and `TextGenerationPipeline`'s/`EmbeddingPipeline`'s
//! own tests document. So this test proves resolution through to
//! `TtsPipelineBuilder::load`'s own existence check (or, with all three
//! files present but fake, through to `qwen3tts::pipeline::synth` itself)
//! succeeding or failing cleanly - never a panic.

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
    let dir = std::env::temp_dir().join(format!("brain-sdk-tts-pipeline-{tag}-{}-{n}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
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

fn tiny_safetensors(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
}

/// The exact on-disk shape `brain tts import` writes and
/// `qwen3tts::spec::Qwen3TtsSpec::classify` reads: an HF checkpoint dir
/// (`config.json`) carrying a `brain_tts/` subdirectory of brain-format
/// weights, plus the two-role `brain.manifest.json` the conversion step
/// records. Not a real tokenizer/real checkpoint - see this file's own doc.
fn write_qwen3tts_checkpoint(repo: &Path) {
    std::fs::create_dir_all(repo).unwrap();
    std::fs::write(repo.join("config.json"), serde_json::to_vec(&serde_json::json!({"architectures": ["Qwen3TTSForConditionalGeneration"]})).unwrap()).unwrap();
    tiny_safetensors(&repo.join("brain_tts").join("talker.safetensors"));
    tiny_safetensors(&repo.join("brain_tts").join("mtp.safetensors"));
    tiny_safetensors(&repo.join("brain_tts").join("codec.safetensors"));
    tiny_safetensors(&repo.join("brain_tts").join("speaker.safetensors"));
    let manifest = serde_json::json!({
        "id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "family": "qwen3tts",
        "roles": {"ckpt": ".", "weights_dir": "brain_tts"},
    });
    std::fs::write(repo.join("brain.manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
}

/// `resolve()`'s own root inference needs a second, unrelated vendor -
/// a SIBLING of the fixture's own vendor directory, under `root` - to land
/// on `root` rather than collapsing onto the fixture's single repo
/// directory, the same requirement `qwen3tts::spec::tests`' own fixture
/// documents.
fn write_unrelated_sibling(root: &Path) {
    let dir = root.join("other-vendor");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("unrelated.gguf"), b"not a real gguf").unwrap();
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::TtsPipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
}

/// The full facade path against a real, converted-shape local fixture, with
/// no network access at any point: reference parses, `Store::local` resolves
/// it, `loader::resolve_structured` finds the `weights_dir`/`ckpt` pair via
/// the SAME `brain.manifest.json` the real conversion step writes, and
/// `TtsPipelineBuilder::load`'s own existence check passes (all three files
/// are present, even though they are not real Talker/MTP/codec shapes).
/// `.speak(...)` then reaches `qwen3tts::pipeline::synth`, which fails
/// cleanly on the fake tokenizer/checkpoint content rather than resolution
/// itself failing.
#[test]
fn from_pretrained_resolves_qwen3tts_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("resolve");
    write_qwen3tts_checkpoint(&root.join("Qwen").join("Qwen3-TTS-12Hz-0.6B-Base"));
    write_unrelated_sibling(&root);

    let pipe = with_models_dir(&root, || brain::TtsPipeline::from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base"));
    let pipe = pipe.expect("a converted-shape checkpoint (fake tensor content, real manifest) must resolve and pass the existence check");

    let err = pipe.speak("hello from a fixture").unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from the fake tokenizer/checkpoint content, got {other:?}"),
    }
}

/// A locally-present repo whose OWN `brain.manifest.json` declares neither
/// `weights_dir` nor `ckpt` (the generic single-role `{"weights": ...}`
/// shape every other spec's own "missing role" test fixture uses) surfaces
/// as `Error::Missing`, naming BOTH of `Qwen3TtsSpec`'s roles - proven the
/// same way `tests/depth_pipeline.rs`/`tests/detection_pipeline.rs` prove it
/// for their own single-role specs. `family` still says `qwen3tts` so
/// `classify_compound_manifest` does not skip the manifest outright; it is
/// the ROLE NAMES that miss, not the family.
#[test]
fn from_pretrained_names_the_missing_role_when_the_store_is_empty() {
    let root = scratch_root("missing");
    let dir = root.join("Qwen").join("Qwen3-TTS-empty-test");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("weights.stub"), b"stub").unwrap();
    std::fs::write(
        dir.join("brain.manifest.json"),
        serde_json::to_vec(&serde_json::json!({"id": "Qwen/Qwen3-TTS-empty-test", "family": "qwen3tts", "roles": {"weights": "weights.stub"}})).unwrap(),
    )
    .unwrap();
    write_unrelated_sibling(&root);

    let err = with_models_dir(&root, || brain::TtsPipeline::from_pretrained("Qwen/Qwen3-TTS-empty-test").unwrap_err());
    match &err {
        brain::Error::Missing(m) => {
            assert_eq!(m.arch, "qwen3tts");
            let roles: Vec<&str> = m.roles.iter().map(|r| r.role.as_str()).collect();
            assert!(roles.contains(&"weights_dir") && roles.contains(&"ckpt"), "{roles:?}");
        }
        other => panic!("expected Error::Missing, got {other:?}"),
    }
}
