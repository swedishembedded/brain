// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `audio` surface, so it compiles only with it -
// the same reason `tests/image_pipeline.rs` gates itself on `image`.
#![cfg(feature = "audio")]

//! End-to-end coverage of `TranscribePipeline::from_pretrained`'s resolution
//! path against a real local, synthetic, fully offline fixture reproducing
//! `crates/qwen3asr/src/spec.rs`'s own (private) classification schema (an
//! `HfDir` whose `config.json` declares `architectures:
//! ["Qwen3ASRForConditionalGeneration"]`) - mirroring `tests/
//! image_pipeline.rs`'s established pattern.
//!
//! ## Why this stops short of a successful `.transcribe(...)`
//!
//! `qwen3asr::caps::QwenAsrProvider::load` needs both a real checkpoint
//! (`Qwen3Asr::from_hf_windowed`) and a real tokenizer
//! (`data::qwen_tokenizer::QwenBpe::from_dir`) - the same two real-content
//! ceilings `TextGenerationPipeline`'s and `EmbeddingPipeline`'s own tests
//! document. So this test proves resolution through to `QwenAsrProvider::load`
//! being reached with the right directory, then a clean `Error::Backend`,
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
    let dir = std::env::temp_dir().join(format!("brain-sdk-transcribe-pipeline-{tag}-{}-{n}", std::process::id()));
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

/// The exact declared-architecture schema `Qwen3AsrSpec::classify` reads,
/// plus a loose safetensors shard so a real `inventory::scan` collapses the
/// directory into an `HfDir` record at all (the same requirement `tests/
/// forecast_pipeline.rs`'s kronos fixture documents).
fn write_qwen3asr_hfdir(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let config = serde_json::json!({"architectures": ["Qwen3ASRForConditionalGeneration"]});
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
}

#[test]
fn from_pretrained_resolves_qwen3asr_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("qwen3asr");
    write_qwen3asr_hfdir(&root.join("Qwen").join("Qwen3-ASR-1.7B"));
    mark_locally_present(&root, "local", "asr-sdk-test", "qwen3asr");

    let err = with_models_dir(&root, || brain::TranscribePipeline::from_pretrained("local/asr-sdk-test").unwrap_err());
    match &err {
        brain::Error::Backend(msg) => assert!(!msg.is_empty(), "must name what went wrong"),
        other => panic!("expected a clean Error::Backend from incomplete checkpoint/tokenizer construction, got {other:?}"),
    }
}

#[test]
fn from_pretrained_rejects_an_unparseable_model_id() {
    let err = brain::TranscribePipeline::from_pretrained("../not/a/valid/ref").unwrap_err();
    assert!(matches!(err, brain::Error::ModelNotFound(_)), "{err:?}");
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

    let err = brain::TranscribePipeline::builder("nonexistent-vendor/nonexistent-repo").download_policy(brain::DownloadPolicy::Offline).load().unwrap_err();

    std::env::remove_var("BRAIN_MODELS_DIR");
    std::env::remove_var("BRAIN_HUB_ENDPOINT");

    assert!(matches!(err, brain::Error::Missing(_)), "Offline must never attempt a fetch, got {err:?}");
}

fn write_nemotronasr_hfdir(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let config = serde_json::json!({"architectures": ["Nemotron3_5AsrForRNNT"]});
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0f32, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
}

/// A Nemotron checkpoint resolves through the same entry point as a Qwen3-ASR
/// one: the failure is the Nemotron loader's own (the fixture holds no real
/// tensors), not "no such model" and not the Qwen3-ASR loader's.
#[test]
fn from_pretrained_resolves_nemotronasr_from_a_real_local_fixture_with_no_network_access() {
    let root = scratch_root("nemotronasr");
    write_nemotronasr_hfdir(&root.join("nvidia").join("nemotron-3.5-asr-streaming-0.6b"));
    mark_locally_present(&root, "local", "nemotron-sdk-test", "nemotronasr");

    let err = with_models_dir(&root, || brain::TranscribePipeline::from_pretrained("local/nemotron-sdk-test").unwrap_err());
    match &err {
        brain::Error::Backend(msg) => assert!(msg.to_lowercase().contains("nemotron"), "must come from the Nemotron loader: {msg}"),
        other => panic!("expected the Nemotron loader's Error::Backend, got {other:?}"),
    }
}

/// A real Nemotron checkpoint transcribes speech one-shot, through the
/// pipeline that also serves Qwen3-ASR. The speech is a Qwen3-TTS rendering of
/// a known sentence, so the transcript can be scored against it. Skips when
/// either checkpoint is absent from the model store.
#[test]
fn nemotron_transcribes_synthesized_speech_one_shot() {
    let _serial = brain_testutil::env_lock();
    let (Ok(tts), Ok(asr)) = (brain::TtsPipeline::from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base"), brain::TranscribePipeline::from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")) else {
        brain_testutil::skip("Qwen3-TTS and Nemotron ASR checkpoints are not both in the model store");
        return;
    };
    let text = "The quick brown fox jumps over the lazy dog.";
    // Seeded: unseeded synthesis of one sentence ranges from under four to
    // over eight seconds, and a short rendering can drop words before
    // recognition ever sees them.
    let clip = tts.speak_with(text, brain::TtsOptions::new().seed(1)).expect("synthesize");
    let out = asr.transcribe_audio(&clip).expect("transcribe");
    assert!(out.truncated.is_none(), "Nemotron has no fixed window");
    let wer = eval::asr::corpus_wer([(text, out.text.as_str())]).expect("reference has words");
    assert!(wer < 0.5, "round-trip WER {wer:.3}: {:?}", out.text);
}

/// Qwen3-ASR hears a whole utterance, not its first words: the encoder is told
/// how many mel frames are valid, and once read the mel-bin count (128, about
/// 1.3 s of audio) for it. Skips when either checkpoint is absent.
#[test]
fn qwen3_asr_transcribes_the_whole_of_synthesized_speech() {
    let _serial = brain_testutil::env_lock();
    let (Ok(tts), Ok(asr)) = (brain::TtsPipeline::from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base"), brain::TranscribePipeline::from_pretrained("Qwen/Qwen3-ASR-1.7B")) else {
        brain_testutil::skip("Qwen3-TTS and Qwen3-ASR checkpoints are not both in the model store");
        return;
    };
    let text = "The quick brown fox jumps over the lazy dog.";
    let clip = tts.speak_with(text, brain::TtsOptions::new().seed(1)).expect("synthesize");
    let out = asr.transcribe_audio(&clip).expect("transcribe");
    let wer = eval::asr::corpus_wer([(text, out.text.as_str())]).expect("reference has words");
    assert!(wer < 0.2, "round-trip WER {wer:.3}: {:?}", out.text);
}

/// The model asked for decides the architecture, not the order the
/// architectures are tried in: with both a Qwen3-ASR and a Nemotron checkpoint
/// in the store, asking for the Nemotron one must reach the Nemotron loader.
#[test]
fn the_requested_checkpoint_selects_the_backend_when_both_are_in_the_store() {
    let root = scratch_root("both");
    write_qwen3asr_hfdir(&root.join("Qwen").join("Qwen3-ASR-1.7B"));
    write_nemotronasr_hfdir(&root.join("nvidia").join("nemotron-3.5-asr-streaming-0.6b"));

    let err = with_models_dir(&root, || brain::TranscribePipeline::from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b").unwrap_err());
    match &err {
        brain::Error::Backend(msg) => assert!(msg.to_lowercase().contains("nemotron"), "must come from the Nemotron loader: {msg}"),
        other => panic!("expected the Nemotron loader's Error::Backend, got {other:?}"),
    }
}

fn tone(seconds: f32, rate: u32) -> brain::Audio {
    let n = (seconds * rate as f32) as usize;
    brain::Audio::new((0..n).map(|i| ((i as f32) * 0.07).sin() * 0.3 + ((i as f32) * 0.011).sin() * 0.2).collect(), rate)
}

/// Qwen3-ASR's encoder features of a clip of any length and rate: thirteen
/// rows per second, no decode window, the same for the same audio. Skips
/// without the checkpoint.
#[test]
fn qwen3_asr_exposes_features_of_speech() {
    let _serial = brain_testutil::env_lock();
    let Ok(asr) = brain::TranscribePipeline::from_pretrained("Qwen/Qwen3-ASR-1.7B") else {
        brain_testutil::skip("Qwen3-ASR is not in the model store");
        return;
    };
    let three = asr.features(&tone(3.0, 16_000)).expect("features");
    assert_eq!((three.encoder_dim, three.embed_dim), (1024, 2048));
    assert_eq!(three.encoder_out.len(), three.rows * three.encoder_dim);
    assert_eq!(three.embeds.len(), three.rows * three.embed_dim);
    assert_eq!(three.rows, 39, "three seconds are three 100-frame chunks of 13 rows");
    assert!(three.embeds.iter().chain(&three.encoder_out).all(|x| x.is_finite()));
    assert_eq!(asr.features(&tone(3.0, 16_000)).unwrap(), three, "deterministic");
    assert_eq!(asr.features(&tone(6.0, 16_000)).unwrap().rows, 78);
    assert_eq!(asr.features(&tone(3.0, 24_000)).unwrap().rows, 39, "resampled to 16 kHz first");
    assert_eq!(asr.features(&tone(35.0, 16_000)).unwrap().rows, 455, "longer than the 30 s decode window is not truncated");
}

#[test]
fn nemotron_has_no_spliceable_features() {
    let _serial = brain_testutil::env_lock();
    let Ok(asr) = brain::TranscribePipeline::from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b") else {
        brain_testutil::skip("Nemotron ASR is not in the model store");
        return;
    };
    assert!(matches!(asr.features(&tone(1.0, 16_000)), Err(brain::Error::MissingArgument(_))));
}
