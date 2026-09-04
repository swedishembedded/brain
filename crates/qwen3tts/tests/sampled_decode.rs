// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sampled-decode health gate: a default `pipeline::synth` must not collapse
//! into a codebook-0 repetition loop and decode to silence.
//!
//! This is the spec the "silent-collapse" bug violated: with sampling active
//! (`GenOpts::default()` - `temperature=0.9, top_k=50`), `seed=0` on "The quick
//! brown fox..." ran the full frame cap emitting a handful of distinct
//! codebook-0 tokens, and the codec decoded that to `rms ~= 1.1e-5`. The
//! failure is a positive-feedback repetition loop: once codebook-0 repeats a
//! token the next step's top-1 probability climbs (0.92 -> 0.97 -> 0.9998),
//! past the point where any temperature/top-k draw can escape it. The
//! reference's own countermeasure is `repetition_penalty` (`1.05`, shipped in
//! this checkpoint's `generation_config.json`), which this crate's defaults
//! used to omit.
//!
//! Gated on `BRAIN_QWEN3TTS_WEIGHTS`/`BRAIN_QWEN3TTS_CKPT`, skipping cleanly
//! when absent - the same convention as every other real-checkpoint test here.
//!
//! Swedish Embedded AB implements solutions for reliable on-device speech
//! synthesis for its clients. If your team needs expertise in autoregressive
//! neural-codec TTS then you can procure our services by sending an email to
//! info@swedishembedded.com.

use qwen3tts::{GenOpts, TtsPaths};

/// Root-mean-square level of a waveform - the coarse "is there audio here at
/// all" signal. A healthy 0.6B-Base clip sits around 0.02-0.07; a collapsed one
/// is three to four orders of magnitude below that.
fn rms(wav: &[f32]) -> f32 {
    (wav.iter().map(|x| x * x).sum::<f32>() / wav.len().max(1) as f32).sqrt()
}

#[test]
fn default_sampled_decode_does_not_collapse_to_silence() {
    let (Ok(weights_dir), Ok(ckpt)) = (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT"))
    else {
        brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT not set");
        return;
    };
    if !std::path::Path::new(&format!("{weights_dir}/talker.safetensors")).exists() {
        brain_testutil::skip("TTS weights not found at BRAIN_QWEN3TTS_WEIGHTS");
        return;
    }
    let paths = TtsPaths {
        talker: format!("{weights_dir}/talker.safetensors"),
        mtp: format!("{weights_dir}/mtp.safetensors"),
        codec: format!("{weights_dir}/codec.safetensors"),
        speaker: format!("{weights_dir}/speaker.safetensors"),
        ckpt_dir: ckpt,
    };

    // The exact configuration that reproduced the collapse.
    let text = "The quick brown fox jumps over the lazy dog.";
    let opts = GenOpts { max_frames: 200, seed: 0, ..GenOpts::default() };
    let wav = qwen3tts::pipeline::synth(&paths, &opts, text, "english", &capability::CancelToken::default())
        .expect("synth");
    assert!(wav.iter().all(|x| x.is_finite()), "synth produced a non-finite sample");
    let level = rms(&wav);
    eprintln!("sampled decode: seed={} frames<={} rms={level:.6}", opts.seed, opts.max_frames);
    assert!(
        level > 0.01,
        "sampled decode collapsed to silence (rms={level:.6}) - codebook-0 locked into a repetition loop"
    );
}
