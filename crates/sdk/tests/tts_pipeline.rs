// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `audio` surface, so it compiles only with it -
// the same reason `tests/transcribe_pipeline.rs` gates itself on `audio`.
#![cfg(feature = "audio")]

//! End-to-end coverage of `TtsPipeline::from_pretrained`'s resolution path,
//! against a real local, synthetic, fully offline fixture for EACH backend:
//! `crates/qwen3tts/src/spec.rs`'s own (private) `write_qwen3tts_checkpoint`
//! shape for Qwen3-TTS, and `crates/cosyvoice/src/spec.rs`'s own (private)
//! `fixture`/`write_{llm,flow,hift}_pt` shape for CosyVoice - mirroring
//! `tests/transcribe_pipeline.rs`'s established pattern for this exact class
//! of ceiling.
//!
//! ## Why this stops short of a successful `.speak(...)`/`.clone_voice(...)`
//!
//! `qwen3tts::pipeline::synth`/`clone` need a real tokenizer
//! (`data::qwen_tokenizer::QwenBpe::from_dir`, via `prompt::load_tokenizer`)
//! and real Talker/MTP/codec checkpoints; `cosyvoice::pipeline::generate`
//! needs real CAM++/S3Tokenizer checkpoints (`BRAIN_CAMPPLUS_DIR`/
//! `BRAIN_S3TOKENIZER_V2` - not yet resolver-migrated, so this test points
//! them at empty scratch directories) on top of its own real llm/flow/hift
//! weights - the same real-content ceiling `TranscribePipeline`'s and
//! `TextGenerationPipeline`'s/`EmbeddingPipeline`'s own tests document. So
//! this test proves resolution through to `TtsPipelineBuilder::load`'s own
//! existence check (or, with every file present but fake, through to the
//! backend's own generation entry point) succeeding or failing cleanly -
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

fn t(name: &str, shape: Vec<usize>) -> checkpoint::torchpt_write::TensorOut {
    let n: usize = shape.iter().product::<usize>().max(1);
    checkpoint::torchpt_write::TensorOut { name: name.to_string(), shape, data: vec![0.0; n] }
}

const COSYVOICE_TOY_TEXT_VOCAB: usize = 200;

/// Only the tensors `cosyvoice::spec::CosyVoiceSpec`'s own `llm_variant`/
/// `llm_text_vocab_size` actually read, at CosyVoice 2's own shape
/// (`llm_embedding.weight` present) - mirrors that module's own private
/// `write_llm_pt` test helper exactly, since it is not exported for reuse.
fn write_cosyvoice_llm_pt(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            t("llm.model.model.embed_tokens.weight", vec![COSYVOICE_TOY_TEXT_VOCAB, 32]),
            t("speech_embedding.weight", vec![6561, 32]),
            t("llm_decoder.weight", vec![6561, 32]),
            t("llm_embedding.weight", vec![2, 32]),
        ],
    )
    .unwrap();
}

/// CosyVoice 2's flow.pt shape (`decoder.estimator.down_blocks.*`, the UNet
/// CFM estimator) - mirrors `cosyvoice::spec::tests`' own `write_flow_pt`.
fn write_cosyvoice_flow_pt(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            t("input_embedding.weight", vec![6561, 80]),
            t("spk_embed_affine_layer.weight", vec![80, 192]),
            t("decoder.estimator.down_blocks.0.0.mlp.1.weight", vec![256, 256]),
        ],
    )
    .unwrap();
}

/// CosyVoice 2's hift.pt shape (`conv_pre`'s weight-normed kernel width 7) -
/// mirrors `cosyvoice::spec::tests`' own `write_hift_pt`.
fn write_cosyvoice_hift_pt(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            t("f0_predictor.classifier.weight", vec![1, 512]),
            t("m_source.l_linear.weight", vec![1, 8]),
            t("conv_pre.parametrizations.weight.original0", vec![512, 1, 1]),
            t("conv_pre.parametrizations.weight.original1", vec![512, 80, 7]),
        ],
    )
    .unwrap();
}

fn write_cosyvoice_tokenizer_json(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let vocab: serde_json::Map<String, serde_json::Value> = (0..COSYVOICE_TOY_TEXT_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
    std::fs::write(path, serde_json::to_vec(&serde_json::json!({"model": {"vocab": vocab}, "added_tokens": []})).unwrap()).unwrap();
}

/// A full, unambiguous CosyVoice 2 fixture: `llm`/`flow`/`hift` under
/// `FunAudioLLM/CosyVoice2-0.5B/`, a compatible tokenizer under the sibling
/// real-world repo `FunAudioLLM/CosyVoice-BlankEN/`, plus
/// [`write_unrelated_sibling`] for `resolve()`'s own root inference - the
/// same real-world layout `cosyvoice::spec::tests`' own `fixture` helper
/// documents.
fn write_cosyvoice_checkpoint(root: &Path) {
    let vendor = root.join("FunAudioLLM").join("CosyVoice2-0.5B");
    write_cosyvoice_llm_pt(&vendor.join("llm.pt"));
    write_cosyvoice_flow_pt(&vendor.join("flow.pt"));
    write_cosyvoice_hift_pt(&vendor.join("hift.pt"));
    write_cosyvoice_tokenizer_json(&root.join("FunAudioLLM").join("CosyVoice-BlankEN").join("tokenizer.json"));
}

/// CosyVoice 3's llm.pt shape: no `llm_embedding.weight` (the
/// `Qwen2LM`-vs-`CosyVoice3LM` split `cosyvoice::spec::llm_variant` tells
/// apart by that tensor's ABSENCE, not a name of its own) - mirrors
/// `cosyvoice::spec::tests`' own `write_llm_pt` at `Variant::CosyVoice3`.
fn write_cosyvoice3_llm_pt(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            t("llm.model.model.embed_tokens.weight", vec![COSYVOICE_TOY_TEXT_VOCAB, 32]),
            t("speech_embedding.weight", vec![6561, 32]),
            t("llm_decoder.weight", vec![6561, 32]),
        ],
    )
    .unwrap();
}

/// CosyVoice 3's flow.pt shape (`decoder.estimator.transformer_blocks.*`,
/// the DiT estimator rather than CosyVoice 2's UNet) - mirrors
/// `cosyvoice::spec::tests`' own `write_flow_pt` at `Variant::CosyVoice3`.
fn write_cosyvoice3_flow_pt(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            t("input_embedding.weight", vec![6561, 80]),
            t("spk_embed_affine_layer.weight", vec![80, 192]),
            t("decoder.estimator.transformer_blocks.0.attn.to_q.weight", vec![1024, 1024]),
        ],
    )
    .unwrap();
}

/// CosyVoice 3's hift.pt shape (`conv_pre`'s weight-normed kernel width 5,
/// vs CosyVoice 2's 7) - mirrors `cosyvoice::spec::tests`' own
/// `write_hift_pt` at `Variant::CosyVoice3`.
fn write_cosyvoice3_hift_pt(path: &Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    checkpoint::torchpt_write::write(
        path.to_str().unwrap(),
        &[
            t("f0_predictor.classifier.weight", vec![1, 512]),
            t("m_source.l_linear.weight", vec![1, 8]),
            t("conv_pre.parametrizations.weight.original0", vec![512, 1, 1]),
            t("conv_pre.parametrizations.weight.original1", vec![512, 80, 5]),
        ],
    )
    .unwrap();
}

/// A full, unambiguous CosyVoice 3 fixture - the same real-world layout
/// [`write_cosyvoice_checkpoint`] documents for CosyVoice 2, at CosyVoice
/// 3's own tensor shapes.
fn write_cosyvoice3_checkpoint(root: &Path) {
    let vendor = root.join("FunAudioLLM").join("CosyVoice3-0.5B");
    write_cosyvoice3_llm_pt(&vendor.join("llm.pt"));
    write_cosyvoice3_flow_pt(&vendor.join("flow.pt"));
    write_cosyvoice3_hift_pt(&vendor.join("hift.pt"));
    write_cosyvoice_tokenizer_json(&root.join("FunAudioLLM").join("CosyVoice-BlankEN").join("tokenizer.json"));
}

/// `CosyVoicePaths::from_assembly` reads `BRAIN_S3TOKENIZER_V2`/
/// `BRAIN_CAMPPLUS_DIR` directly (see that function's own doc: both roles
/// are not yet resolver-migrated) - this only needs to exist, not hold real
/// weights, since resolution/construction never reads its contents (only
/// `cosyvoice::pipeline::generate`'s own import step would, past this test's
/// documented ceiling). Must run INSIDE `with_models_dir`'s closure: both
/// helpers share one process-wide env lock
/// (`brain_testutil::env_lock`, non-reentrant), so nesting a second
/// acquisition would deadlock.
fn with_cosyvoice_env<T>(dir: &Path, f: impl FnOnce() -> T) -> T {
    std::env::set_var("BRAIN_S3TOKENIZER_V2", dir);
    std::env::set_var("BRAIN_CAMPPLUS_DIR", dir);
    let out = f();
    std::env::remove_var("BRAIN_S3TOKENIZER_V2");
    std::env::remove_var("BRAIN_CAMPPLUS_DIR");
    out
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

fn write_wav(path: &Path, seconds: f32, sample_rate: u32) {
    let n = ((seconds * sample_rate as f32) as usize).max(1);
    audio::wav::write(path, &vec![0.0f32; n], sample_rate).unwrap();
}

/// The full facade path against a real, content-classified local CosyVoice 2
/// fixture, with no network access at any point: reference parses,
/// `Store::local` resolves it, `resolve_arch` tries qwen3tts first (this
/// fixture has nothing qwen3tts-shaped, so that try returns `Missing` and is
/// silently discarded - the same first-architecture fallback
/// `ForecastPipeline`'s own `resolve_arch` documents), then
/// `cosyvoice::spec::CosyVoiceSpec` resolves it and
/// `TtsPipelineBuilder::load`'s own existence check passes.
/// `.clone_voice(...)` then reaches `cosyvoice::pipeline::generate`, which
/// fails cleanly on the fake CAM++/S3Tokenizer weight content (both env vars
/// point at an empty scratch directory) rather than resolution itself
/// failing.
#[test]
fn from_pretrained_resolves_cosyvoice_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("resolve-cosyvoice");
    write_cosyvoice_checkpoint(&root);
    write_unrelated_sibling(&root);
    let voice = root.join("reference.wav");
    write_wav(&voice, 1.0, 16000);

    let pipe = with_models_dir(&root, || with_cosyvoice_env(&root, || brain::TtsPipeline::from_pretrained("FunAudioLLM/CosyVoice2-0.5B")));
    let pipe = pipe.expect("a content-classifiable CosyVoice 2 checkpoint (fake tensor content, real shapes) must resolve and pass the existence check");

    let err = pipe.clone_voice("hello from a fixture", &voice, Some("the reference transcript")).unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from the fake CAM++/S3Tokenizer content, got {other:?}"),
    }
}

/// `speak`/`design` are Qwen3-TTS-only call shapes (see `crates/sdk/src/
/// tts.rs`'s own module doc) - a CosyVoice-resolved pipeline returns
/// `Error::MissingArgument` for both, knowable before any backend call.
#[test]
fn speak_and_design_are_missing_argument_errors_on_a_cosyvoice_resolved_pipeline() {
    let root = scratch_root("cosyvoice-wrong-actions");
    write_cosyvoice_checkpoint(&root);
    write_unrelated_sibling(&root);

    let pipe = with_models_dir(&root, || with_cosyvoice_env(&root, || brain::TtsPipeline::from_pretrained("FunAudioLLM/CosyVoice2-0.5B")))
        .expect("a content-classifiable CosyVoice 2 checkpoint must resolve and pass the existence check");

    assert!(matches!(pipe.speak("hello").unwrap_err(), brain::Error::MissingArgument(_)));
    assert!(matches!(pipe.design("hello", "cheerful", None).unwrap_err(), brain::Error::MissingArgument(_)));
}

/// CosyVoice's `clone_voice` has no x-vector-only mode - omitting `ref_text`
/// is a caller-programming error, knowable before any backend call, the same
/// class `speak`/`design` already return on this backend.
#[test]
fn clone_voice_without_ref_text_is_a_missing_argument_error_on_a_cosyvoice_resolved_pipeline() {
    let root = scratch_root("cosyvoice-no-ref-text");
    write_cosyvoice_checkpoint(&root);
    write_unrelated_sibling(&root);
    let voice = root.join("reference.wav");
    write_wav(&voice, 1.0, 16000);

    let pipe = with_models_dir(&root, || with_cosyvoice_env(&root, || brain::TtsPipeline::from_pretrained("FunAudioLLM/CosyVoice2-0.5B")))
        .expect("a content-classifiable CosyVoice 2 checkpoint must resolve and pass the existence check");

    assert!(matches!(pipe.clone_voice("hello", &voice, None).unwrap_err(), brain::Error::MissingArgument(_)));
}

/// The roadmap's own tracked gap: `CosyVoiceSpec::classify` has told
/// CosyVoice 2 and 3 apart by tensor content since Phase 4.2 landed (see
/// `cosyvoice::spec::tests::classify_tells_cosyvoice2_and_cosyvoice3_*_apart_by_*`),
/// but until now nothing proved `TtsPipeline::from_pretrained` itself
/// resolves a CosyVoice 3-shaped checkpoint end to end - only CosyVoice 2
/// had SDK-level coverage. Same ceiling as the CosyVoice 2 fixture test
/// above: resolution and the existence check succeed against fake tensor
/// content, and `.clone_voice_with(..., TtsOptions::new().variant("cosyvoice3"))`
/// (required - see [`TtsOptions::variant`]'s own doc: the default is
/// CosyVoice 2, independent of which generation actually resolved) reaches
/// `cosyvoice::pipeline::generate`'s real CosyVoice 3 branch
/// (`CosyVoiceLm::load_cosyvoice3`), which fails cleanly on the fake
/// CAM++/S3Tokenizer content rather than resolution itself failing.
#[test]
fn from_pretrained_resolves_cosyvoice3_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("resolve-cosyvoice3");
    write_cosyvoice3_checkpoint(&root);
    write_unrelated_sibling(&root);
    let voice = root.join("reference.wav");
    write_wav(&voice, 1.0, 16000);

    let pipe = with_models_dir(&root, || with_cosyvoice_env(&root, || brain::TtsPipeline::from_pretrained("FunAudioLLM/CosyVoice3-0.5B")));
    let pipe = pipe.expect("a content-classifiable CosyVoice 3 checkpoint (fake tensor content, real shapes) must resolve and pass the existence check");

    let opts = brain::TtsOptions::new().variant("cosyvoice3");
    let err = pipe.clone_voice_with("hello from a fixture", &voice, Some("the reference transcript"), opts).unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from the fake CAM++/S3Tokenizer content, got {other:?}"),
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
    std::env::set_var("BRAIN_MODELS_DIR", &*root);
    std::env::set_var("BRAIN_HUB_ENDPOINT", "http://127.0.0.1:1");

    let err = brain::TtsPipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}
