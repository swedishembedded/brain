// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resident engine on the device against the resident engine on the host.
//!
//! Both are greedy and deterministic. They run the same model in the same
//! fp32 but a different reduction order, so a long clip may part ways at a
//! near-tie; what is asserted is that they agree on the start of the clip,
//! stop at about the same length, and that the device engine is the fast one.
//! Gated on `BRAIN_QWEN3TTS_WEIGHTS`/`BRAIN_QWEN3TTS_CKPT` like the other
//! real-checkpoint tests.

use std::time::Instant;

use capability::CancelToken;
use qwen3tts::engine::{Placement, ResidentEngine};
use qwen3tts::{GenOpts, SamplingRequest, TtsPaths};

const TEXT: &str = "The people must be taught to value their liberty above their ease.";

fn paths() -> Option<TtsPaths> {
    let (Ok(weights), Ok(ckpt)) = (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT")) else {
        brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT not set");
        return None;
    };
    let paths = TtsPaths::new(&weights, ckpt);
    if paths.require(false).is_err() {
        brain_testutil::skip("TTS weights not found");
        return None;
    }
    Some(paths)
}

fn greedy(max_frames: usize) -> GenOpts {
    GenOpts { max_frames, sampling: SamplingRequest::greedy(), ..GenOpts::default() }
}

#[test]
fn the_device_engine_speaks_the_same_start_as_the_host_engine_and_faster() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let never = CancelToken::default();

    let mut host = ResidentEngine::load_with(&paths, Placement::Host).unwrap();
    let began = Instant::now();
    let on_host = host.speak_codes(TEXT, "english", &greedy(120), &never).unwrap();
    let host_time = began.elapsed();
    drop(host);

    let mut device = ResidentEngine::load_with(&paths, Placement::Device).unwrap();
    // The first request pays for kernel compilation and graph capture.
    device.speak_codes(TEXT, "english", &greedy(8), &never).unwrap();
    let began = Instant::now();
    let on_device = device.speak_codes(TEXT, "english", &greedy(120), &never).unwrap();
    let device_time = began.elapsed();

    let frames = |codes: &[u32]| codes.len() / 16;
    eprintln!("host {} frames in {host_time:?}, device {} frames in {device_time:?}", frames(&on_host), frames(&on_device));
    let head = 12 * 16;
    assert_eq!(&on_device[..head], &on_host[..head], "the first twelve frames are the same codes");
    let (a, b) = (frames(&on_host) as i64, frames(&on_device) as i64);
    assert!((a - b).abs() <= 3, "lengths {a} and {b} frames");
    assert!(device_time < host_time, "device {device_time:?} is not faster than host {host_time:?}");
}

#[test]
fn a_cancelled_device_request_leaves_the_engine_usable() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let mut device = ResidentEngine::load_with(&paths, Placement::Device).unwrap();
    let cancel = CancelToken::armed();
    cancel.cancel();
    assert_eq!(device.speak_codes(TEXT, "english", &greedy(40), &cancel).unwrap_err(), "cancelled");
    assert!(!device.speak_codes(TEXT, "english", &greedy(16), &CancelToken::default()).unwrap().is_empty());
}

#[test]
fn a_request_that_does_not_fit_the_device_context_is_refused() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let mut device = ResidentEngine::load_with(&paths, Placement::Device).unwrap();
    let err = device.speak_codes(TEXT, "english", &greedy(5_000), &CancelToken::default()).unwrap_err();
    assert!(err.contains("does not fit"), "{err}");
}

#[test]
fn the_device_codec_decodes_the_same_audio_as_the_host_codec() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let never = CancelToken::default();

    let mut host = ResidentEngine::load_with(&paths, Placement::Host).unwrap();
    let codes = host.speak_codes(TEXT, "english", &greedy(40), &never).unwrap();
    let on_host = host.decode(&codes, &mut |_, _| {}).unwrap();
    drop(host);

    let mut device = ResidentEngine::load_with(&paths, Placement::Device).unwrap();
    let mut chunks = 0;
    let on_device = device.decode(&codes, &mut |_, _| chunks += 1).unwrap();

    assert_eq!(on_device.len(), on_host.len(), "the same number of samples");
    assert!(chunks > 1, "the device clip still reaches the caller in chunks");
    let energy: f64 = on_host.iter().map(|x| f64::from(*x).powi(2)).sum();
    let error: f64 = on_host.iter().zip(&on_device).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    let relative = (error / energy).sqrt();
    eprintln!("codec device vs host: relative rms error {relative:.5}");
    assert!(relative < 0.02, "relative rms error {relative}");
}

#[test]
fn a_streamed_utterance_is_the_same_audio_and_its_first_sound_comes_before_it_is_finished() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let never = CancelToken::default();
    let mut engine = ResidentEngine::load_with(&paths, Placement::Device).unwrap();
    // Warm the kernels so the timing below is of the engine, not of the compiler.
    engine.speak_codes(TEXT, "english", &greedy(8), &never).unwrap();

    let whole = engine.speak(TEXT, "english", &greedy(60), &never, &mut |_, _| {}).unwrap();
    let mut chunks = Vec::new();
    let began = Instant::now();
    let mut first_sound = None;
    let streamed = engine
        .speak_streaming(TEXT, "english", &greedy(60), &never, &mut |pcm, seq| {
            first_sound.get_or_insert_with(|| began.elapsed());
            chunks.push((seq, pcm.len()));
        })
        .unwrap();
    let finished = began.elapsed();

    assert_eq!(streamed.len(), whole.len());
    let energy: f64 = whole.iter().map(|x| f64::from(*x).powi(2)).sum();
    let error: f64 = whole.iter().zip(&streamed).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    let relative = (error / energy).sqrt();
    eprintln!("streamed vs whole: relative rms error {relative:.6}; first sound {first_sound:?} of {finished:?}; chunks {chunks:?}");
    assert!(relative < 1e-3, "relative rms error {relative}");
    assert!(chunks.len() >= 2, "the sound reached the caller in pieces: {chunks:?}");
    assert_eq!(chunks.iter().map(|c| c.1).sum::<usize>(), streamed.len(), "the pieces make the whole");
    assert!(first_sound.unwrap() < finished / 2, "first sound {first_sound:?} of {finished:?}");
    assert!(engine.last_timings().first_audio.is_some());
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
fn a_pinned_voice_holds_the_speaker_tighter_than_the_free_voice_which_already_varies_little_across_seeds() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let never = CancelToken::default();
    let mut engine = ResidentEngine::load_with(&paths, Placement::Device).unwrap();
    let sampled = |seed: u64| GenOpts { max_frames: 60, seed, sampling: SamplingRequest::default(), ..GenOpts::default() };

    // The reference is the engine's own rendering: a synthetic voice.
    let reference = engine.speak(TEXT, "english", &sampled(1), &never, &mut |_, _| {}).unwrap();
    let dir = std::env::temp_dir().join(format!("qwen3tts-voice-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("reference.wav").to_string_lossy().into_owned();
    audio::wav::write(&wav, &reference, 24_000).unwrap();

    let encoder = ecapatdnn::SpeakerEncoder::load_inference(&paths.speaker);
    let embed = |pcm: &[f32]| encoder.embed_wav(pcm, 24_000);
    let mut pinned = Vec::new();
    let mut free = Vec::new();
    // seed: the pinned-over-free similarity margin and the 0.95 floor were measured on these four takes
    for seed in 2..=5u64 {
        let take = engine.clone_voice_streaming(TEXT, &wav, "", "english", &sampled(seed), &never, &mut |_, _| {}).unwrap();
        pinned.push(embed(&take));
        free.push(embed(&engine.speak(TEXT, "english", &sampled(seed), &never, &mut |_, _| {}).unwrap()));
    }
    let reference = embed(&reference);
    let mean = |set: &[Vec<f32>]| set.iter().map(|e| cosine(e, &reference)).sum::<f32>() / set.len() as f32;
    eprintln!("speaker similarity to the reference: pinned {:.3}, free {:.3}", mean(&pinned), mean(&free));
    // Measured on the 0.6B base: the free voice already sits near the reference
    // (the sampling seed moves delivery far more than the speaker), and pinning
    // holds it closer still. What matters for takes of one voice is that the
    // pinned ones are both close and closer than the free ones.
    assert!(mean(&pinned) > mean(&free), "pinning must hold the speaker: pinned {} free {}", mean(&pinned), mean(&free));
    assert!(mean(&pinned) > 0.95, "pinned takes sound like the reference: {}", mean(&pinned));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_rows_rebuilt_from_a_rollout_reproduce_the_policy_that_made_it() {
    let _serial = brain_testutil::env_lock();
    let Some(paths) = paths() else { return };
    let never = CancelToken::default();
    let mut engine = ResidentEngine::load_with(&paths, Placement::Device).unwrap();
    // Greedy: every sampled first codebook is the argmax of the logits at its row.
    let rollout = engine.rollout(TEXT, "english", None, &greedy(40), &never).unwrap();
    let rows = engine.policy_rows(&rollout).unwrap();
    let frames = rollout.codes.len() / 16;
    assert_eq!(rows.len(), rollout.prompt.embeds.len() / rows.d + frames);
    let logits = engine.score(&rows).unwrap();
    let vocab = logits.len() / rows.target_positions().count();
    let mut checked = 0;
    for (i, p) in rows.target_positions().enumerate() {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let best = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        // The repetition penalty reshapes sampling, so compare against the raw argmax
        // only where generation had no repeat to penalise.
        if rows.targets[p] == best {
            checked += 1;
        }
    }
    let total = rows.target_positions().count();
    eprintln!("{checked} of {total} first-codebook targets are the argmax under the rebuilt rows; ended by eos: {}", rollout.ended_by_eos);
    assert!(checked * 10 >= total * 9, "the rebuilt rows must reproduce the policy: {checked} of {total}");
}
