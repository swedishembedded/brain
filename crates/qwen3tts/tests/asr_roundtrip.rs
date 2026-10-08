// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ASR round-trip quality gate: synthesize a known sentence, transcribe it
//! back with an independent ASR model (Nemotron-3.5-ASR), and assert the
//! transcript's word error rate against the input text is low. Catches gross
//! synthesis regressions (garbled audio, wrong language, near-silence) that a
//! per-tensor logit diff can miss and nobody is listening for on every CI
//! run, without needing a PyTorch reference at all - both models are already
//! in-tree, gradcheck-verified against their own oracles independently.
//!
//! Gated on `BRAIN_QWEN3TTS_WEIGHTS`/`BRAIN_QWEN3TTS_CKPT` (the TTS
//! checkpoint) and `BRAIN_NEMOTRONASR` (the ASR checkpoint dir, matching
//! `crates/arch`'s own env var for this architecture) all being set and
//! present; skips cleanly otherwise, the same convention as every other
//! real-checkpoint test in this crate.

use eval::asr::{normalize, word_error_rate};
use qwen3tts::{GenOpts, TtsPaths};

#[test]
fn synth_then_transcribe_recovers_the_text() {
    let (Ok(weights_dir), Ok(ckpt), Ok(asr_ckpt)) =
        (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT"), std::env::var("BRAIN_NEMOTRONASR"))
    else {
        brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT/BRAIN_NEMOTRONASR not all set");
        return;
    };
    let paths = TtsPaths::new(&weights_dir, ckpt);
    if paths.require(false).is_err() {
        brain_testutil::skip("TTS weights not found at BRAIN_QWEN3TTS_WEIGHTS");
        return;
    }
    if !std::path::Path::new(&format!("{asr_ckpt}/model.safetensors")).exists() {
        brain_testutil::skip("ASR weights not found at BRAIN_NEMOTRONASR");
        return;
    }

    let text = "The quick brown fox jumps over the lazy dog.";
    let opts = GenOpts { max_frames: 200, ..GenOpts::default() };
    let wav24 = qwen3tts::pipeline::synth(&paths, &opts, text, "english", &capability::CancelToken::default()).expect("synth");
    assert!(wav24.iter().all(|x| x.is_finite()), "synth produced a non-finite sample");
    let rms = (wav24.iter().map(|x| x * x).sum::<f32>() / wav24.len().max(1) as f32).sqrt();
    assert!(rms > 0.01, "synth produced near-silence (rms={rms:.4}) - the greedy-collapse failure mode this repo has hit before");

    let wav16 = audio::resample_linear(&wav24, 24000, 16000);

    let cfg = nemotronasr::NemotronConfig::nemotron_3_5_asr_0_6b();
    let asr = nemotronasr::model::NemotronAsr::from_hf(&asr_ckpt, cfg).expect("load ASR model");
    let detok = nemotronasr::tokenizer::Detokenizer::from_hf(&asr_ckpt).expect("load ASR tokenizer");
    let ids = asr.transcribe(&wav16, 0); // prompt 0 = english
    let nonblank: Vec<u32> = ids.into_iter().filter(|&x| x != cfg.blank_token_id).collect();
    let transcript = detok.decode(&nonblank);

    let wer = word_error_rate(&normalize(text), &normalize(&transcript));
    eprintln!("ASR round-trip: input={text:?} transcript={transcript:?} WER={wer:.3}");
    // A loose bound: this is a coarse regression net (garbled/silent/wrong-language
    // output), not a transcription-accuracy benchmark - two independently-trained
    // models each carry their own error on top of each other, so this number is a
    // smoke signal, not a claim about either model's real accuracy.
    assert!(wer < 0.5, "round-trip WER too high ({wer:.3}): input={text:?} got={transcript:?}");
}
