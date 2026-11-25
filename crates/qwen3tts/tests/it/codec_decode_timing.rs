// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements neural audio codec inference on GPUs for its clients. If your team needs
// expertise in low-latency speech synthesis then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Wall time of `mimi::Codec::decode` and `encode` on the device for a realistic clip, warm.
//!
//! Ignored by default: it measures, it does not gate. Run it alone, on a card
//! nothing else is using, to compare a change to the decode pass:
//! `cargo test -p brain-qwen3tts --test it codec_decode_timing -- --ignored --nocapture`.
//! Wrap it in `nsys profile -t cuda` for the per-kernel and driver-call split.

use std::time::Instant;

use data::rng::Lcg;

fn random_codes(frames: usize) -> Vec<u32> {
    let mut rng = Lcg::new(0xC0DE);
    (0..frames * 16).map(|_| rng.next_u64() as u32 % 2048).collect()
}

#[test]
#[ignore = "timing measurement, not a gate"]
fn codec_decode_timing_on_the_device() {
    let Some(paths) = crate::store::store_paths() else { return };
    let codec = mimi::Codec::load_inference_on(gpu_core::Gpu::new(mimi::PIPELINES), &paths.codec);
    for frames in [100usize, 600] {
        let codes = random_codes(frames);
        let cold = Instant::now();
        let first = codec.decode(&codes);
        let cold = cold.elapsed();
        let warm = Instant::now();
        let second = codec.decode(&codes);
        let warm = warm.elapsed();
        assert_eq!(first.len(), second.len());
        eprintln!("codec decode {frames} frames ({:.1} s of audio): cold {cold:?}, warm {warm:?}", first.len() as f64 / 24_000.0);
    }
}

/// Wall time of `mimi::Codec::encode` on the device for the audio a decode makes, warm: what building a training set
/// of codes from recordings costs per second of speech.
#[test]
#[ignore = "timing measurement, not a gate"]
fn codec_encode_timing_on_the_device() {
    let Some(paths) = crate::store::store_paths() else { return };
    let codec = mimi::Codec::load_inference_on(gpu_core::Gpu::new(mimi::PIPELINES), &paths.codec);
    for frames in [100usize, 600] {
        let wav = codec.decode(&random_codes(frames));
        let cold = Instant::now();
        let first = codec.encode(&wav);
        let cold = cold.elapsed();
        let warm = Instant::now();
        let second = codec.encode(&wav);
        let warm = warm.elapsed();
        assert_eq!(first, second, "a repeated encode of the same audio gives other codes");
        // A checksum of the codes, so two builds can be compared on the same audio.
        let sum = first.iter().enumerate().fold(0u64, |h, (i, &c)| h.wrapping_mul(0x100_0000_01b3).wrapping_add(u64::from(c) ^ i as u64));
        eprintln!("codec encode {:.1} s of audio: cold {cold:?}, warm {warm:?} ({:.1}x real time warm), codes checksum {sum:016x}", wav.len() as f64 / 24_000.0, wav.len() as f64 / 24_000.0 / warm.as_secs_f64());
    }
}
