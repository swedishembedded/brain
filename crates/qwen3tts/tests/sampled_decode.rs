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
//! **The gate asserts PROPERTIES, never exact PCM.** Sampled decode runs
//! through wgsl kernels on whatever device is ambient, so the last bits of a
//! sample - and, through a near-tied draw, occasionally a whole token - are
//! backend- and numerics-dependent. A bit-exact expectation would fail on a
//! machine where nothing is wrong, which is the fastest way to get a real gate
//! deleted. What is NOT numerics-dependent is the shape of the failure: a
//! collapsed clip runs to its cap, never reaches EOS, repeats one codebook-0
//! token for tens of consecutive frames, and decodes to something three to four
//! orders of magnitude quieter than speech. Each of those is asserted directly,
//! and each has a threshold read off the measured collapse rather than guessed.
//!
//! Gated on `BRAIN_QWEN3TTS_WEIGHTS`/`BRAIN_QWEN3TTS_CKPT`, skipping cleanly
//! when absent - the same convention as every other real-checkpoint test here.
//!
//! Swedish Embedded AB implements solutions for reliable on-device speech
//! synthesis for its clients. If your team needs expertise in autoregressive
//! neural-codec TTS then you can procure our services by sending an email to
//! info@swedishembedded.com.

use qwen3tts::sampling::DEGENERATE_RUN;
use qwen3tts::{GenOpts, TtsPaths};

/// Codebooks per codec frame (`[T,16]`), so `codes[f * CODEBOOKS]` is frame
/// `f`'s codebook-0 token.
const CODEBOOKS: usize = 16;

/// The frame cap for this run. Large enough that a healthy clip of this
/// sentence (measured: EOS at 38 frames) never approaches it, so "ran the cap"
/// is unambiguous evidence of the collapse rather than of a long sentence.
const MAX_FRAMES: usize = 200;

/// Root-mean-square level of a waveform - the coarse "is there audio here at
/// all" signal. A healthy 0.6B-Base clip sits around 0.02-0.07; a collapsed one
/// is three to four orders of magnitude below that. `0.01` is the bar every
/// real-weight speech gate in this repo already uses (`asr_roundtrip.rs`,
/// `runtime::tts`), kept identical so "not collapsed" means one thing here.
const MIN_RMS: f32 = 0.01;

/// Shortest believable clip for this nine-word sentence at 12.5 Hz frames
/// (~2.5 s of speech is ~31 frames; the measured healthy run is 38). An EOS
/// before this is the OTHER way to produce silence - a clip that stops
/// immediately - and would otherwise slip past every assertion below.
const MIN_FRAMES: usize = 12;

fn rms(wav: &[f32]) -> f32 {
    (wav.iter().map(|x| x * x).sum::<f32>() / wav.len().max(1) as f32).sqrt()
}

/// Longest run of one repeated codebook-0 token, and the token it repeated.
fn longest_cb0_run(cb0: &[u32]) -> (usize, u32) {
    let mut best = (0usize, 0u32);
    let mut run = 0usize;
    for (i, &t) in cb0.iter().enumerate() {
        run = if i > 0 && cb0[i - 1] == t { run + 1 } else { 1 };
        if run > best.0 {
            best = (run, t);
        }
    }
    best
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

    // The exact configuration that reproduced the collapse. `GenOpts::default()`
    // pins NO sampling knob, so this also gates the resolution chain: whatever
    // the checkpoint's `generation_config.json` says is what runs.
    let text = "The quick brown fox jumps over the lazy dog.";
    let opts = GenOpts { max_frames: MAX_FRAMES, seed: 0, ..GenOpts::default() };
    let cancel = capability::CancelToken::default();

    // Stop one stage short of the waveform first: the codes carry the evidence
    // (frame count, EOS, run lengths) that RMS alone cannot distinguish.
    let codes = qwen3tts::pipeline::synth_codes(&paths, &opts, text, "english", &cancel).expect("synth_codes");
    assert_eq!(codes.len() % CODEBOOKS, 0, "codes are not a whole number of [T,16] frames");
    let frames = codes.len() / CODEBOOKS;
    let cb0: Vec<u32> = (0..frames).map(|f| codes[f * CODEBOOKS]).collect();
    let distinct = cb0.iter().collect::<std::collections::HashSet<_>>().len();
    let (run, run_token) = longest_cb0_run(&cb0);
    eprintln!("sampled decode: seed={} frames={frames}/{MAX_FRAMES} distinct_cb0={distinct} longest_run={run}x{run_token}", opts.seed);

    // (1) A natural EOS. The decode loop leaves only on the codec EOS or on the
    // frame cap, and never emits the EOS frame itself - so finishing under the
    // cap IS "the model chose to stop". The collapsed run finished at exactly
    // 200/200; the healthy one at 38. Half the cap keeps a 2.6x margin over the
    // measurement while staying far from "never terminated".
    assert!(
        frames < MAX_FRAMES / 2,
        "sampled decode did not reach a natural EOS: {frames}/{MAX_FRAMES} frames, {distinct} distinct codebook-0 tokens \
         (a collapsed clip runs the whole cap)"
    );
    // (2) ...and stopped because it finished speaking, not immediately.
    assert!(frames >= MIN_FRAMES, "sampled decode stopped after only {frames} frames - too short to be this sentence");

    // (3) No pathological codebook-0 repetition run. Calibrated on the real
    // collapse: the run that still escaped was 10 long, the ones that locked
    // the clip were 20, 41, 52 and 80. `DEGENERATE_RUN` is the same 20 the
    // in-loop diagnostic trips on, shared so the gate and the diagnostic can
    // never drift apart.
    assert!(
        run <= DEGENERATE_RUN,
        "codebook-0 token {run_token} repeated {run} times in a row (limit {DEGENERATE_RUN}) - this is the repetition loop \
         that decodes to silence, even though the clip terminated"
    );

    // (4) Only now the waveform, through the same decode `synth` performs.
    let wav = qwen3tts::pipeline::decode_codes(&paths.codec, &codes).expect("decode_codes");
    assert!(wav.iter().all(|x| x.is_finite()), "synth produced a non-finite sample");
    let level = rms(&wav);
    eprintln!("sampled decode: samples={} rms={level:.6}", wav.len());
    assert!(
        level > MIN_RMS,
        "sampled decode collapsed to silence (rms={level:.6}, floor {MIN_RMS}) - codebook-0 locked into a repetition loop"
    );
}
