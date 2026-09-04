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

use qwen3tts::{GenOpts, TtsPaths};

/// Word-level Levenshtein distance / reference length - the standard ASR
/// quality metric. Case-insensitive (word tokens are compared with
/// `eq_ignore_ascii_case`); punctuation should already be stripped by
/// [`normalize`] before this is called.
fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hypothesis.split_whitespace().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let (n, m) = (r.len(), h.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        row[0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if r[i - 1].eq_ignore_ascii_case(h[j - 1]) {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1])
            };
        }
    }
    dp[n][m] as f32 / n as f32
}

/// Strip punctuation and case, matching what a WER comparison should ignore
/// (the ASR's own text normalization convention, not the TTS's).
fn normalize(s: &str) -> String {
    let cleaned: String = s.chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

#[test]
fn word_error_rate_matches_known_cases() {
    assert_eq!(word_error_rate("the cat sat", "the cat sat"), 0.0);
    assert_eq!(word_error_rate("the cat sat", "the cat"), 1.0 / 3.0);
    assert_eq!(word_error_rate("the cat sat", "the dog sat"), 1.0 / 3.0);
    assert_eq!(word_error_rate("THE Cat Sat", "the cat sat"), 0.0, "case-insensitive");
}

#[test]
fn synth_then_transcribe_recovers_the_text() {
    let (Ok(weights_dir), Ok(ckpt), Ok(asr_ckpt)) =
        (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT"), std::env::var("BRAIN_NEMOTRONASR"))
    else {
        brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT/BRAIN_NEMOTRONASR not all set");
        return;
    };
    if !std::path::Path::new(&format!("{weights_dir}/talker.safetensors")).exists() {
        brain_testutil::skip("TTS weights not found at BRAIN_QWEN3TTS_WEIGHTS");
        return;
    }
    if !std::path::Path::new(&format!("{asr_ckpt}/model.safetensors")).exists() {
        brain_testutil::skip("ASR weights not found at BRAIN_NEMOTRONASR");
        return;
    }

    let text = "The quick brown fox jumps over the lazy dog.";
    let paths = TtsPaths {
        talker: format!("{weights_dir}/talker.safetensors"),
        mtp: format!("{weights_dir}/mtp.safetensors"),
        codec: format!("{weights_dir}/codec.safetensors"),
        speaker: format!("{weights_dir}/speaker.safetensors"),
        ckpt_dir: ckpt,
    };
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
