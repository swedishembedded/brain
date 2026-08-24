// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vocoder at a REAL chunk length, on a real device.
//!
//! `vocoder_parity.rs` decodes 6 latent frames. A real denoise chunk is
//! `CHUNK_FRAMES = 200` AR frames, which the condition encoder resamples to
//! ~689 latent frames and the vocoder upsamples 512x to ~352 k samples per
//! channel - four orders of magnitude more, and the point at which this
//! stage's device footprint actually matters.
//!
//! That gap is not academic: a 10 s two-chunk generation denoised both
//! chunks and then died with `wgpu error: Out of Memory` inside the
//! vocoder, while every existing vocoder test passed. `vocoder::forward`
//! records ONE deferred tape and submits at the end, so every intermediate
//! in the whole four-stage upsample stack is alive simultaneously rather
//! than being freed as the stack unwinds.
//!
//! Gated on `BRAIN_MINIMAXMUSIC3_VOCODER`; skips cleanly without it.

use std::path::Path;

use minimaxmusic3::config::VocoderConfig;
use minimaxmusic3::vocoder::{forward, import, PIPELINES};

/// Latent frames in one full `denoise::CHUNK_FRAMES` chunk - what
/// `condition_encoder::latent_length` yields for 200 AR frames.
const REAL_CHUNK_LATENTS: usize = 689;

#[test]
fn decodes_a_full_denoise_chunk_without_exhausting_the_device() {
    let Ok(dir) = std::env::var("BRAIN_MINIMAXMUSIC3_VOCODER") else {
        brain_testutil::skip("BRAIN_MINIMAXMUSIC3_VOCODER unset");
        return;
    };
    if !Path::new(&dir).exists() {
        brain_testutil::skip(&format!("BRAIN_MINIMAXMUSIC3_VOCODER={dir} not found"));
        return;
    }

    let cfg = VocoderConfig::real();
    let w = import(&dir, &cfg).expect("import vocoder");
    let gpu = gpu_core::testgpu::dev(PIPELINES);

    let n = cfg.latent_channels as usize * REAL_CHUNK_LATENTS;
    let latents: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 0.1).collect();

    let upsample: usize = cfg.upsampling_ratios.iter().product::<u32>() as usize;
    let expect = 2 * REAL_CHUNK_LATENTS * upsample;

    // TWO chunks back to back on ONE device, which is what a real song does
    // (`stitch::Stitcher::push_chunk` per chunk, all on the vocoder stage's
    // single `Gpu`). A single chunk fits comfortably; the question this gate
    // exists to answer is whether chunk 1's intermediates are actually
    // released before chunk 2 allocates its own, or whether the peak is
    // additive - which on a 24 GB card is the difference between working and
    // an out-of-memory, and is exactly how a two-chunk generation died while
    // every single-chunk test passed.
    let mut first: Option<Vec<f32>> = None;
    for chunk in 0..2 {
        let t0 = std::time::Instant::now();
        let out = forward(&gpu, &cfg, &w, &latents, 1, REAL_CHUNK_LATENTS);
        let secs = t0.elapsed().as_secs_f32();
        eprintln!("vocoder real chunk {}: {REAL_CHUNK_LATENTS} latents -> {} samples in {secs:.2}s", chunk + 1, out.len());
        assert_eq!(out.len(), expect, "expected {expect} samples (2 channels x {REAL_CHUNK_LATENTS} x {upsample})");
        assert!(out.iter().all(|v| v.is_finite()), "vocoder produced a non-finite sample at real chunk length");
        // The same latents twice must give the same samples twice, to the bit.
        // The forward is deterministic, so anything else means state survived
        // between calls - which is a live risk now that the lowered
        // convolutions share ONE reusable scratch buffer across the whole
        // recorded tape (`audio::conv::ConvScratch`). A stale-scratch bug
        // would show up here and nowhere else in this crate's suite, because
        // every other vocoder test decodes 6 frames, well under one GEMM
        // chunk.
        match &first {
            None => first = Some(out),
            Some(prev) => assert_eq!(&out, prev, "the vocoder is not deterministic across two chunks on one device"),
        }
    }
}
