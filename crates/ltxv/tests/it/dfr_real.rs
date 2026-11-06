// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running large video diffusion
// pipelines on commodity accelerators for its clients. If your team needs
// expertise in multi-stage video generation then you can procure our services
// by sending an email to info@swedishembedded.com.

//! DFR end to end on the real checkpoint: canvas, half-resolution stage 1
//! with generated keyframe slots, the spatial upscale and the detailing pass
//! under the Refine-Details IC-LoRA, decode - every piece of which is pinned
//! against the reference separately (`dfr_stage_conditioning_matches_the_
//! reference` for both stages' conditioned states, the real-weight DiT and
//! connector goldens for the forward). This runs them together and checks
//! what a caller gets back.

/// A short DFR clip with sound comes back at the requested length, size
/// and rate, with frames that are neither blank nor saturated noise and a
/// soundtrack covering the clip.
#[test]
#[ignore = "needs a GPU and the real LTX-2.5 checkpoint, text encoder, spatial upscaler, audio VAE and Refine-Details IC-LoRA in the model store"]
fn a_dfr_clip_on_real_weights_comes_back_whole() {
    let Some(paths) = ltxv::spec::default_store_paths() else {
        brain_testutil::skip("the model store does not resolve LTX-2.5");
        return;
    };
    if paths.dit.is_none() || paths.detailing_lora.is_none() || paths.spatial_upsampler.is_none() || paths.audio_vae.is_none() || paths.text_encoder.is_none() {
        brain_testutil::skip("the model store lacks a role DFR needs (dit, text_encoder, spatial_upsampler, audio_vae, detailing_lora)");
        return;
    }
    let base = ltxv::pipeline::GenOpts { dit_config: "ltx25_22b".into(), frames: 33, width: 256, height: 128, fps: 24, audio: true, seed: 11, ..Default::default() };
    let o = ltxv::pipeline::DfrOpts { base, temporal_upsample_rounds: 0 };
    let t0 = std::time::Instant::now();
    let (video, timings) = ltxv::pipeline::generate_dfr(&paths, "a red kite climbs over a windy beach", &o, &Default::default(), |_, _, _| {}).unwrap_or_else(|e| panic!("generate_dfr: {e}"));
    eprintln!("DFR 33 frames at 256x128 with sound: {:.1} s ({timings:?})", t0.elapsed().as_secs_f64());

    assert_eq!((video.width, video.height, video.fps, video.frames.len()), (256, 128, 24, 33), "the request's own geometry, the canvas trimmed back");
    for (i, f) in video.frames.iter().enumerate() {
        let mean = f.iter().map(|&p| p as f64).sum::<f64>() / f.len() as f64;
        let var = f.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / f.len() as f64;
        assert!(var.sqrt() > 4.0, "frame {i} is flat (std {:.2})", var.sqrt());
    }
    let audio = video.audio.as_ref().expect("an audio request returns sound");
    let seconds = audio.samples_per_channel() as f64 / audio.sample_rate as f64;
    assert!((seconds - 33.0 / 24.0).abs() < 0.05, "the soundtrack covers {seconds:.3} s of a {:.3} s clip", 33.0 / 24.0);
    assert!(audio.channels.iter().flatten().all(|s| s.is_finite()), "non-finite audio samples");
}

/// One temporal round on a two-segment canvas: the round's two tiles - the
/// second resuming on its prefix plane - come back stitched into the doubled
/// clip at twice the frame rate, `(49 - 1) * 2 + 1` frames.
#[test]
#[ignore = "needs a GPU and the real LTX-2.5 checkpoint, text encoder, both latent upscalers, audio VAE and Refine-Details IC-LoRA in the model store"]
fn a_dfr_temporal_round_on_real_weights_doubles_the_clip() {
    let Some(paths) = ltxv::spec::default_store_paths() else {
        brain_testutil::skip("the model store does not resolve LTX-2.5");
        return;
    };
    if paths.dit.is_none() || paths.detailing_lora.is_none() || paths.spatial_upsampler.is_none() || paths.temporal_upsampler.is_none() || paths.audio_vae.is_none() {
        brain_testutil::skip("the model store lacks a role DFR's temporal round needs");
        return;
    }
    let base = ltxv::pipeline::GenOpts { dit_config: "ltx25_22b".into(), frames: 49, width: 128, height: 128, fps: 24, audio: true, seed: 12, ..Default::default() };
    let o = ltxv::pipeline::DfrOpts { base, temporal_upsample_rounds: 1 };
    let t0 = std::time::Instant::now();
    let (video, _) = ltxv::pipeline::generate_dfr(&paths, "waves roll onto a pebble beach", &o, &Default::default(), |_, _, _| {}).unwrap_or_else(|e| panic!("generate_dfr: {e}"));
    eprintln!("DFR 49 frames + one temporal round at 128x128: {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!((video.fps, video.frames.len()), (48, 97));
    for (i, f) in video.frames.iter().enumerate() {
        let mean = f.iter().map(|&p| p as f64).sum::<f64>() / f.len() as f64;
        let std = (f.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
        assert!(std > 4.0, "frame {i} is flat (std {std:.2})");
    }
    let audio = video.audio.as_ref().expect("an audio request returns sound");
    let seconds = audio.samples_per_channel() as f64 / audio.sample_rate as f64;
    assert!((seconds - 97.0 / 48.0).abs() < 0.05, "the soundtrack covers {seconds:.3} s of a {:.3} s clip", 97.0 / 48.0);
}
