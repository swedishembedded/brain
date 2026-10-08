// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#![cfg(feature = "audio")]

//! `Audio` is a clip a caller can build, save and read back, not only
//! something a synthesizer hands out.

use brain::Audio;

fn tone(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.05).sin() * 0.5).collect()
}

#[test]
fn a_clip_built_from_samples_keeps_them_and_its_rate() {
    let clip = Audio::new(tone(8_000), 16_000);
    assert_eq!(clip.samples().len(), 8_000);
    assert_eq!(clip.sample_rate(), 16_000);
    assert!((clip.seconds() - 0.5).abs() < 1e-9);
}

#[test]
fn a_saved_clip_reads_back_at_its_own_rate_within_16_bit_quantisation() {
    let dir = std::env::temp_dir().join(format!("brain-audio-clip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tone.wav");
    let clip = Audio::new(tone(2_400), 24_000);
    clip.save(&path).unwrap();

    let back = Audio::from_wav(&std::fs::read(&path).unwrap()).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(back.sample_rate(), 24_000, "the rate is the file's, not 16 kHz");
    assert_eq!(back.samples().len(), clip.samples().len());
    let worst = clip.samples().iter().zip(back.samples()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 1.0 / 16_000.0, "worst difference {worst}");
}

#[test]
fn bytes_that_are_not_a_wav_file_are_refused() {
    assert!(Audio::from_wav(b"not a wav file").is_err());
}

#[test]
fn the_word_error_rate_that_judges_a_round_trip_is_reachable_from_the_sdk() {
    assert_eq!(brain::wer::word_error_rate("the cat sat", "the cat"), 1.0 / 3.0);
    assert_eq!(brain::wer::corpus_wer([("a b", "a"), ("a b c d e f g h", "a b c d e f g x")]), Some(0.2));
}
