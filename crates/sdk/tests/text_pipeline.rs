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
//! ## Why this stops short of a successful `.generate(...)` -- most of it, not all of it
//!
//! `QwenConfig::from_json`'s own defaults happen to equal `QwenConfig::tiny()`
//! exactly (both default to `vocab: 23, n_layers: 2, d_model: 16, ...`), so
//! an empty `{}` config header is enough to prove the checkpoint opens and
//! parses - genuinely cheaper than `ImagePipeline`'s fixtures, which need
//! real classifiable tensor shapes. Most tests below stop there: a clean,
//! named `Error::MissingArgument` at the tokenizer-resolution step - never a
//! panic, and never reaching the (much heavier) `Qwen::load_inference` call.
//!
//! [`a_fully_synthetic_checkpoint_and_tokenizer_reach_a_real_generate`] goes
//! all the way, closing the gap this doc used to describe as out of reach:
//! `QwenBpe::from_json_bytes` already accepts ARBITRARY JSON bytes (it was
//! never missing a constructor, just a fixture nobody had written), and
//! `QwenConfig::tiny()`'s own `param_list()` (`(name, numel)` pairs, the same
//! discipline `Flux2Config::tensor_manifest()` uses) is a complete,
//! mechanical recipe for a real, forward-capable checkpoint at 23-token
//! vocab / 2-layer / 16-dim scale. See that test's own doc for the fixture
//! design (a curated small vocab, not a universal one - reproducing an HF
//! tokenizer.json byte-for-byte is still out of scope, this only needs to
//! cover the one prompt under test).

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

/// A LOCAL path whose checkpoint positively declares a DIFFERENT
/// architecture (`ModelCard.family == "qwen35"`, not qwen3's own `"qwen"`)
/// is refused BEFORE `Qwen::load_inference` -- the local-path counterpart
/// to what `Qwen3Spec::classify` already guarantees for free on the hub-id
/// path (see `check_local_weights_architecture`'s own doc in
/// `crates/sdk/src/text.rs`). Without this check, this exact fixture would
/// instead reach `Qwen::load_inference`'s panic-on-mismatched-tensor-names
/// deep inside model construction -- the same class of unchecked panic this
/// module's own `WeightReader::open`-before-`load_inference` ordering
/// already guards against for a bad PATH, now also guarded for a bad
/// ARCHITECTURE at a real, openable one.
#[test]
fn a_local_checkpoint_declaring_a_different_architecture_is_refused_before_load_inference() {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("brain-sdk-text-pipeline-wrong-arch-{}-{n}.safetensors", std::process::id()));
    let card = checkpoint::st::ModelCard::new("some/qwen35-checkpoint", "qwen35");
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), Some(&card)).unwrap();

    let err = brain::TextGenerationPipeline::from_pretrained(path.to_str().unwrap()).unwrap_err();
    std::fs::remove_file(&path).ok();
    match &err {
        brain::Error::Backend(msg) => {
            assert!(msg.contains("qwen35"), "{msg}");
            assert!(msg.contains("qwen"), "{msg}");
        }
        other => panic!("expected a clean Error::Backend naming the architecture mismatch, got {other:?}"),
    }
}

/// `DownloadPolicy::Offline` never reaches the network -- same proof shape
/// as `crates/sdk/tests/image_pipeline.rs`'s own
/// `download_policy_offline_never_touches_the_network`: point `HfHub` at a
/// loopback port nothing listens on, and show a hub-shaped reference
/// resolving neither locally nor from any real hub still comes back the
/// resolver's own clean `Error::Missing`, never a connection-error-flavored
/// `Error::Download`.
#[test]
fn download_policy_offline_never_touches_the_network() {
    let _serial = brain_testutil::env_lock();
    let root = std::env::temp_dir().join(format!("brain-sdk-text-pipeline-offline-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).unwrap();
    std::env::set_var("BRAIN_MODELS_DIR", &root);
    std::env::set_var("BRAIN_HUB_ENDPOINT", "http://127.0.0.1:1");

    let err = brain::TextGenerationPipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");
    std::fs::remove_dir_all(&root).ok();

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}

/// A minimal, COMPLETE (for this one prompt, not universal text) tokenizer:
/// every byte of `a_prompt_within_vocab` maps to a distinct id in
/// `0..QwenConfig::tiny().vocab` (23), so the decode side can also round-trip
/// every id the tiny model's 23-wide lm_head can ever sample - not just the
/// prompt's own characters. `data::bpe::bytes_to_unicode()` is the SAME
/// byte<->char table `QwenBpe::encode_piece` looks up through, so a vocab key
/// built any other way (e.g. the literal ASCII byte) would silently miss on
/// every lookup (see [`data::qwen_tokenizer::QwenBpe`]'s own "a miss ... drop
/// it" contract - the failure mode would be an empty encode, not a panic, so
/// this is worth getting right on purpose rather than debugging it by
/// symptom).
const VOCAB_LETTERS: std::ops::RangeInclusive<u8> = b'a'..=b'w'; // 23 letters
const PROMPT_WITHIN_VOCAB: &str = "cabbage"; // every char in 'a'..='w'

fn tiny_tokenizer_json() -> serde_json::Value {
    let byte_encoder = data::bpe::bytes_to_unicode();
    let vocab: serde_json::Map<String, serde_json::Value> = VOCAB_LETTERS
        .enumerate()
        .map(|(id, b)| (byte_encoder[b as usize].to_string(), serde_json::json!(id as u32)))
        .collect();
    assert_eq!(vocab.len(), qwen3::QwenConfig::tiny().vocab as usize, "the vocab must exactly cover the tiny config's lm_head width");
    serde_json::json!({ "model": { "vocab": vocab, "merges": [] } })
}

/// A real, forward-capable `QwenConfig::tiny()` checkpoint: every tensor
/// `param_list()` names, filled with small deterministic non-constant values
/// (the same fill `crates/flux2/tests/model_smoke.rs` uses for its own
/// from-scratch synthetic DiT) - not all-zero, so a degenerate all-equal
/// softmax cannot mask a real indexing bug. The `{}` header is deliberate,
/// not a placeholder: `QwenConfig::from_json`'s own defaults already equal
/// `QwenConfig::tiny()` exactly (this file's module doc), so `param_list()`
/// called on that SAME `tiny()` is guaranteed to match what `load_inference`
/// reads back, with no risk of the header and the tensor set drifting apart.
fn tiny_qwen3_checkpoint(tag: &str) -> Scratch {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("brain-sdk-text-pipeline-real-{tag}-{}-{n}.safetensors", std::process::id()));
    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = qwen3::QwenConfig::tiny()
        .param_list()
        .into_iter()
        .map(|(name, numel)| {
            let data: Vec<f32> = (0..numel).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
            (name, vec![numel as u64], data)
        })
        .collect();
    checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::json!({}), None).unwrap();
    Scratch(path)
}

/// The gap this file's module doc used to call out of reach: resolve ->
/// open -> parse -> tokenize -> build -> a REAL `generate_kv_stream` forward
/// pass -> detokenize, entirely on a from-scratch synthetic fixture, no
/// network and no real checkpoint anywhere. `.chat(false)` sends
/// `PROMPT_WITHIN_VOCAB` to the model VERBATIM (no ChatML template, so the
/// fixture tokenizer needs no `<|im_start|>`/`<|im_end|>` specials) - the
/// output is not meaningful text (every weight is a small synthetic filler
/// value, not a trained parameter), but every stage between the public SDK
/// entry point and a real per-token sample runs for real, which is the
/// thing this test exists to prove reachable at all.
#[test]
fn a_fully_synthetic_checkpoint_and_tokenizer_reach_a_real_generate() {
    let ckpt = tiny_qwen3_checkpoint("real-generate");
    let tok_path = std::env::temp_dir().join(format!("brain-sdk-text-pipeline-real-generate-tokenizer-{}.json", std::process::id()));
    std::fs::write(&tok_path, serde_json::to_vec(&tiny_tokenizer_json()).unwrap()).unwrap();
    let tok_path = Scratch(tok_path);

    let pipe = brain::TextGenerationPipeline::builder(ckpt.to_str().unwrap()).tokenizer(tok_path.to_str().unwrap()).load().expect("a fully-synthetic but complete checkpoint + tokenizer must build");

    let out = pipe
        .generate_with(PROMPT_WITHIN_VOCAB, brain::TextGenerationOptions::new().chat(false).max_new_tokens(4))
        .expect("a real forward pass over a synthetic checkpoint must still produce a completion");

    assert_eq!(out.prompt_tokens as usize, PROMPT_WITHIN_VOCAB.len(), "every char of the prompt is a single known vocab entry, so encoding must not drop or merge any of them");
    assert_eq!(out.completion_tokens, 4, "no eos/stop token exists in this fixture's vocab, so generation must run the full requested budget");
    assert_eq!(out.finish_reason, "length");
    assert!(out.text.chars().all(|c| VOCAB_LETTERS.clone().any(|b| b as char == c)), "every decoded char must come from the fixture's own vocab, got {:?}", out.text);
}
