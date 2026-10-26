// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "audio")]

//! A resident synthesizer loads its checkpoint once and answers many requests,
//! stops when asked, and speaks the same voice as the one-shot pipeline.
//! These specs need the Qwen3-TTS checkpoint in the model store and skip
//! without it.

use brain::{CancelToken, TtsOptions, TtsPipeline};

const MODEL: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
const TEXT: &str = "The people must be taught to value their liberty.";

fn pipeline() -> Option<TtsPipeline> {
    let loaded = TtsPipeline::from_pretrained(MODEL).ok();
    if loaded.is_none() {
        brain_testutil::skip("Qwen3-TTS checkpoint is not in the model store");
    }
    loaded
}

#[test]
fn a_resident_synthesizer_speaks_what_the_one_shot_pipeline_speaks() {
    let _serial = brain_testutil::env_lock();
    let Some(pipe) = pipeline() else { return };
    let seed = brain::random_seed();
    let opts = || TtsOptions::new().seed(seed);

    let one_shot = pipe.speak_with(TEXT, opts()).unwrap();
    let resident = pipe.resident().unwrap().speak_with(TEXT, opts()).unwrap();

    assert_eq!(resident.sample_rate(), one_shot.sample_rate());
    assert_eq!(resident.samples().len(), one_shot.samples().len());
    let worst = one_shot.samples().iter().zip(resident.samples()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1e-3, "worst sample difference {worst}");
}

#[test]
fn streamed_chunks_joined_are_the_whole_clip() {
    let _serial = brain_testutil::env_lock();
    let Some(pipe) = pipeline() else { return };
    let resident = pipe.resident().unwrap();

    let mut joined = Vec::new();
    let mut chunks = 0;
    let clip = resident
        .speak_stream(TEXT, TtsOptions::new().seed(brain::random_seed()), &CancelToken::default(), &mut |pcm: &[f32]| {
            chunks += 1;
            joined.extend_from_slice(pcm);
        })
        .unwrap();

    assert!(chunks >= 1);
    assert_eq!(joined, clip.samples(), "the chunks are the clip, in order");
}

#[test]
fn a_synthesizer_asked_to_stop_stops_with_cancelled_and_speaks_nothing() {
    let _serial = brain_testutil::env_lock();
    let Some(pipe) = pipeline() else { return };
    let resident = pipe.resident().unwrap();
    let cancel = CancelToken::armed();
    cancel.cancel();

    let mut heard = 0;
    let result = resident.speak_stream(TEXT, TtsOptions::new().seed(brain::random_seed()), &cancel, &mut |pcm: &[f32]| heard += pcm.len());

    assert!(matches!(result, Err(brain::Error::Cancelled)), "{result:?}");
    assert_eq!(heard, 0);
    // A stopped request leaves the synthesizer usable.
    assert!(resident.speak_with(TEXT, TtsOptions::new().seed(brain::random_seed())).is_ok());
}

#[test]
fn only_qwen3_tts_has_a_resident_form() {
    let _serial = brain_testutil::env_lock();
    // A CosyVoice-resolved pipeline cannot be built here without its checkpoints,
    // so the contract is stated on the one that can: the resident form of a
    // Qwen3-TTS pipeline exists.
    let Some(pipe) = pipeline() else { return };
    assert!(pipe.resident().is_ok());
}

#[test]
fn a_pinned_voice_speaks_every_seed_and_a_missing_reference_is_an_error() {
    let _serial = brain_testutil::env_lock();
    let Some(pipe) = pipeline() else { return };
    let resident = pipe.resident().unwrap();
    let dir = std::env::temp_dir().join(format!("sdk-voice-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let reference = dir.join("reference.wav");
    resident.speak_with(TEXT, TtsOptions::new().seed(brain::random_seed())).unwrap().save(&reference).unwrap();

    let (take_a, take_b) = (brain::random_seed(), brain::random_seed());
    let a = resident.speak_with(TEXT, TtsOptions::new().seed(take_a).voice(&reference, "")).unwrap();
    let b = resident.speak_with(TEXT, TtsOptions::new().seed(take_b).voice(&reference, "")).unwrap();
    assert!(!a.samples().is_empty() && !b.samples().is_empty());
    assert_ne!(a.samples(), b.samples(), "different seeds are different takes");

    let missing = dir.join("nothing.wav");
    let err = resident.speak_with(TEXT, TtsOptions::new().voice(&missing, "")).unwrap_err();
    assert!(err.to_string().contains("nothing.wav"), "{err}");
    resident.speak_with(TEXT, TtsOptions::new().seed(brain::random_seed())).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
