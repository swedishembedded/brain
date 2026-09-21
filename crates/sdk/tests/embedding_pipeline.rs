// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The file covers both `EmbeddingPipeline` backends; each test group is
// gated on the feature its backend actually needs, so `--features text`
// alone and `--features vision` alone each still compile and run their half.
#![cfg(any(feature = "vision", feature = "text"))]

//! End-to-end coverage of `EmbeddingPipeline::from_pretrained`'s resolution
//! path against a real local, synthetic, fully offline fixture - mirroring
//! `tests/image_pipeline.rs`'s pattern and, for the exact classification
//! schema, `crates/clip/src/spec.rs`'s own (private) `write_pipeline` test
//! fixture (an SDXL-layout root: `model_index.json` naming
//! `StableDiffusionXLPipeline`, plus four component directories).
//!
//! ## Why the CLIP tests stop short of a successful `.embed(...)`
//!
//! `clip::caps::Session::load` needs a REAL tokenizer under `tokenizer/`
//! (`data::clip_bpe::ClipBpe::from_dir` reads a real vocab/merges schema,
//! the same class of fixture gap `TextGenerationPipeline`'s own tests
//! document for `QwenBpe`). So those tests prove resolution all the way to
//! `Session::load` being reached with the right directory - then fail
//! CLEANLY (a typed `brain::Error`, never a panic) on the fixture's
//! deliberately empty tokenizer directories.
//!
//! ## Why the Qwen3 tests DO reach a real `.embed(...)`
//!
//! `data::qwen_tokenizer::QwenBpe` (unlike `ClipBpe`) accepts a merge-free
//! byte-level `tokenizer.json` - the exact synthetic fixture
//! `crates/sdk/tests/document_study.rs::write_base_dir` builds, reused here
//! (via a local `write_qwen3_embed_base`) with no real checkpoint or network
//! access needed at all.

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

#[cfg(feature = "vision")]
fn with_models_dir<T>(root: &Path, f: impl FnOnce() -> T) -> T {
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", root);
    let out = f();
    std::env::remove_var("BRAIN_MODELS_DIR");
    out
}

#[cfg(feature = "vision")]
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
#[cfg(feature = "vision")]
fn write_sdxl_tower_root(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model_index.json"), serde_json::to_vec(&serde_json::json!({"_class_name": "StableDiffusionXLPipeline"})).unwrap()).unwrap();
    for c in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"] {
        std::fs::create_dir_all(dir.join(c)).unwrap();
    }
}

#[cfg(feature = "vision")]
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

#[cfg(feature = "vision")]
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
#[cfg(feature = "vision")]
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

/// `DownloadPolicy::Offline` never reaches the network -- same proof shape
/// as `crates/sdk/tests/image_pipeline.rs`'s own
/// `download_policy_offline_never_touches_the_network`: point `HfHub` at a
/// loopback port nothing listens on, and show a reference resolving neither
/// locally nor from any real hub still comes back the resolver's own clean
/// `Error::Missing`, never a connection-error-flavored `Error::Download`.
#[cfg(feature = "vision")]
#[test]
fn download_policy_offline_never_touches_the_network() {
    let root = scratch_root("offline");
    let _serial = brain_testutil::env_lock();
    std::env::set_var("BRAIN_MODELS_DIR", &*root);
    std::env::set_var("BRAIN_HUB_ENDPOINT", "http://127.0.0.1:1");

    let err = brain::EmbeddingPipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}

// ---- Qwen3 backend: a real local checkpoint, no network, no real weights ----

/// A tiny, randomly initialised Qwen3 checkpoint plus a merge-free byte-level
/// `tokenizer.json` (every input byte is its own token - the same fixture
/// shape `crates/sdk/tests/document_study.rs::write_base_dir` builds), so the
/// Qwen3 backend can be driven end to end with no real checkpoint and no
/// network access. `vocab` only needs to cover the 256-entry byte alphabet -
/// unlike `document_study.rs`'s fixture, nothing here runs through
/// `data::chat`, so there is no `ENDOFTEXT` floor to also cover.
#[cfg(feature = "text")]
fn write_qwen3_embed_base(dir: &Path, block: u32) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();

    let cfg = qwen3::QwenConfig { vocab: 256, block_size: block, max_position_embeddings: block, ..qwen3::QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 11);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
        .collect();
    let weights = dir.join("qwen-embed.safetensors");
    checkpoint::save(weights.to_str().unwrap(), cfg.to_json(), &tensors);

    let mut vocab = serde_json::Map::new();
    for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::json!(i));
    }
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();

    weights
}

/// The headline contract: a real (if synthetic) forward pass, pooled and
/// L2-normalized - unit norm is not a tautology here, it is the one number
/// that proves `l2_normalize` actually ran over whatever `Qwen::prefill`
/// returned.
#[cfg(feature = "text")]
#[test]
fn qwen3_backend_embeds_a_real_checkpoint_and_returns_a_unit_norm_vector() {
    let root = scratch_root("qwen3-embed");
    let weights = write_qwen3_embed_base(&root, 32);

    let pipe = brain::EmbeddingPipeline::builder(weights.to_str().unwrap()).capacity(32).tokenizer(root.join("tokenizer.json").to_str().unwrap()).load().unwrap();
    let v = pipe.embed("hello world").unwrap();

    assert_eq!(v.dim(), 16, "QwenConfig::tiny d_model");
    let norm: f64 = v.as_slice().iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
}

#[cfg(feature = "text")]
#[test]
fn cosine_similarity_of_a_vector_with_itself_is_one() {
    let root = scratch_root("qwen3-embed-cosine");
    let weights = write_qwen3_embed_base(&root, 32);
    let pipe = brain::EmbeddingPipeline::builder(weights.to_str().unwrap()).capacity(32).tokenizer(root.join("tokenizer.json").to_str().unwrap()).load().unwrap();

    let v = pipe.embed("the report on Q3 revenue").unwrap();
    assert!((v.cosine_similarity(&v) - 1.0).abs() < 1e-4);
}

/// `EmbeddingOptions::dimensions` truncates AND renormalizes by default - the
/// deliberate difference from the HTTP `/v1/embeddings` endpoint this
/// module's own doc names.
#[cfg(feature = "text")]
#[test]
fn dimensions_option_truncates_and_renormalizes() {
    let root = scratch_root("qwen3-embed-dims");
    let weights = write_qwen3_embed_base(&root, 32);
    let pipe = brain::EmbeddingPipeline::builder(weights.to_str().unwrap()).capacity(32).tokenizer(root.join("tokenizer.json").to_str().unwrap()).load().unwrap();

    let full = pipe.embed("hello world").unwrap();
    let half = pipe.embed_with("hello world", brain::EmbeddingOptions::new().dimensions(full.dim() / 2)).unwrap();

    assert_eq!(half.dim(), full.dim() / 2);
    let norm: f64 = half.as_slice().iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "truncated vector must be renormalized by default, got norm {norm}");
}

/// A request past the pipeline's BUILT capacity is refused with a message
/// naming the capacity, not silently truncated or rebuilt - the same
/// contract `TextGenerationPipeline::generate_with` carries for `max_new`.
#[cfg(feature = "text")]
#[test]
fn a_request_past_the_built_capacity_is_refused() {
    let root = scratch_root("qwen3-embed-capacity");
    let weights = write_qwen3_embed_base(&root, 32);
    let pipe = brain::EmbeddingPipeline::builder(weights.to_str().unwrap()).capacity(4).tokenizer(root.join("tokenizer.json").to_str().unwrap()).load().unwrap();

    let err = pipe.embed("this input is longer than four tokens for certain").unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(msg.contains("capacity"), "got: {msg}"),
        other => panic!("expected Error::Backend naming the capacity, got {other:?}"),
    }
}

// ---- LFM2 backend: a real local checkpoint, no network, no real weights ----

/// A tiny, randomly initialised LFM2.5 checkpoint plus the same merge-free
/// byte-level `tokenizer.json` the Qwen3 fixture above uses - LFM2 reads the
/// SAME `data::qwen_tokenizer::QwenBpe` format. Written with
/// `checkpoint::save_carded` (not the plain `checkpoint::save` every other
/// fixture in this file uses): `EmbeddingPipeline`'s LFM2-vs-Qwen3 routing
/// reads `ModelCard.family` (`brain::EmbeddingPipeline`'s own doc), and
/// plain `save` writes no card at all (`checkpoint::save`'s own doc:
/// delegates to `st::save_safetensors(..., None)`) - a real `brain import`
/// always calls the carded path (`lfm2::import::remap`), so this fixture
/// matches that, not the card-less convenience every other test here uses
/// for a checkpoint nothing needs to classify.
#[cfg(feature = "text")]
fn write_lfm2_embed_base(dir: &Path, block: u32) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();

    let cfg = lfm2::LfmConfig { vocab: 256, block_size: block, ..lfm2::LfmConfig::tiny() };
    let init = lfm2::init::init_weights(&cfg, 11);
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
        .param_list()
        .into_iter()
        .map(|(name, n)| (name.clone(), vec![n as u64], init.get(&name).unwrap_or_else(|| panic!("init missing {name}")).clone()))
        .collect();
    let weights = dir.join("lfm2-embed.safetensors");
    let card = checkpoint::st::ModelCard::new("brain/lfm2", lfm2::spec::CARD_FAMILY);
    checkpoint::save_carded(weights.to_str().unwrap(), cfg.to_json(), &tensors, &card);

    let mut vocab = serde_json::Map::new();
    for (i, c) in data::bpe::bytes_to_unicode().iter().enumerate() {
        vocab.insert(c.to_string(), serde_json::json!(i));
    }
    std::fs::write(dir.join("tokenizer.json"), serde_json::json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();

    weights
}

/// The card-family routing this milestone adds: `ModelCard.family == "lfm"`
/// (`checkpoint::save`'s config carries `"model": "lfm"`, which
/// `checkpoint::st::read_card` reads back as the family) must resolve to the
/// LFM2 backend, not the pre-existing Qwen3 default - proving
/// `resolve_text_backend` actually distinguishes the two rather than always
/// falling through to Qwen3 the way it did before this backend existed.
#[cfg(feature = "text")]
#[test]
fn lfm2_backend_embeds_a_real_checkpoint_and_returns_a_unit_norm_vector() {
    let root = scratch_root("lfm2-embed");
    let weights = write_lfm2_embed_base(&root, 32);

    let pipe = brain::EmbeddingPipeline::builder(weights.to_str().unwrap()).capacity(32).tokenizer(root.join("tokenizer.json").to_str().unwrap()).load().unwrap();
    let v = pipe.embed("hello world").unwrap();

    assert_eq!(v.dim(), 16, "LfmConfig::tiny d_model");
    let norm: f64 = v.as_slice().iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
}

/// The rebuild-on-length-change path (`Backend::Lfm2::hot`): two different
/// inputs on the SAME pipeline must both succeed and both come back unit
/// norm - proving the mutex-guarded resident cache correctly rebuilds rather
/// than reusing a graph sized for the wrong length.
#[cfg(feature = "text")]
#[test]
fn lfm2_backend_rebuilds_when_the_request_length_changes() {
    let root = scratch_root("lfm2-embed-rebuild");
    let weights = write_lfm2_embed_base(&root, 64);
    let pipe = brain::EmbeddingPipeline::builder(weights.to_str().unwrap()).capacity(64).tokenizer(root.join("tokenizer.json").to_str().unwrap()).load().unwrap();

    for text in ["hi", "a somewhat longer sentence than the first one"] {
        let v = pipe.embed(text).unwrap();
        let norm: f64 = v.as_slice().iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "{text:?}: expected unit norm, got {norm}");
    }
}

/// `instruction` is a Qwen3-Embedding-only option (see
/// `EmbeddingOptions::instruction`'s own doc) - an LFM2-backed pipeline must
/// refuse it, not silently ignore it.
#[cfg(feature = "text")]
#[test]
fn lfm2_backend_refuses_the_instruction_option() {
    let root = scratch_root("lfm2-embed-instruction");
    let weights = write_lfm2_embed_base(&root, 32);
    let pipe = brain::EmbeddingPipeline::builder(weights.to_str().unwrap()).capacity(32).tokenizer(root.join("tokenizer.json").to_str().unwrap()).load().unwrap();

    let err = pipe.embed_with("hello world", brain::EmbeddingOptions::new().instruction("Given a query, retrieve relevant passages")).unwrap_err();
    match err {
        brain::Error::Backend(msg) => assert!(msg.contains("instruction"), "got: {msg}"),
        other => panic!("expected Error::Backend naming 'instruction', got {other:?}"),
    }
}
