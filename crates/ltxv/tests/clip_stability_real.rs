// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **A clip must not fall apart part-way through.** The real-weight gate on
//! the failure mode that every other gate in this crate is blind to: a
//! generation where each individual frame is a plausible image, every value
//! is finite, the statistics are in range, the tiled decode's seams are
//! clean - and the video still disintegrates before it ends.
//!
//! Swedish Embedded AB implements objective quality gates for generative
//! video pipelines for its clients. If your team needs a regression signal
//! that catches a clip degrading part-way through without a human watching
//! every frame, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What this catches that nothing else did
//!
//! The defect this file exists for was a real, user-reported 1080p
//! regression: the first ~0.7 s of a 1-second clip was correct and the last
//! several frames were visibly warped and smeared. It passed everything.
//! `vae_tiling`'s gates passed because the decoder was innocent.
//! `motion_real` passed because the clip DID move - peak excursion is a
//! floor, and a clip that runs away from frame 0 scores BETTER on it the
//! worse it disintegrates. `dit_parity` and friends passed because a single
//! forward against a golden tensor is correct at any token count. The
//! output statistics (min/max/std/nonfinite) were all normal.
//!
//! What separates the two cases is entirely TEMPORAL:
//! [`ltxv::clipmetric::blowup_ratio`], the largest frame-to-frame difference
//! over the median one. A clip with steady motion holds it near 1 whatever
//! its motion actually is, because the median tracks that clip's own pace; a
//! clip that comes apart at one point pushes it into double digits.
//!
//! Measured on the real 22B Q8_0 DiT + real Gemma-4 encoder + real conv VAE,
//! one prompt, one seed, one conditioning still, everything but the
//! resolution held fixed:
//!
//! | request | video tokens | blowup ratio |
//! |---|---:|---:|
//! | 512x512 | 1024 | 1.06 |
//! | 960x544 | 2040 | 1.03 |
//! | 1280x704 | 3520 | 1.04 |
//! | 1600x896 | 5600 | 1.04 |
//! | 1920x1088, ONE stage | 8160 | **14.66** |
//!
//! [`BOUND`] sits an order of magnitude clear of the defect and several
//! times clear of every healthy clip ever measured here.
//!
//! # Why it is `#[ignore]`d, and why it is still the right gate
//!
//! The defect only exists above `SINGLE_STAGE_MAX_TOKENS`, and a token count
//! that large IS a full real generation - there is no small shape that
//! reproduces it, because the token count is the variable. So this costs one
//! real 1080p clip (~20 minutes on a Tesla P40 pair) and is `#[ignore]`d for
//! cost, not for confidence. The cheap, always-run half of the same claim is
//! `ltxv::pipeline`'s own `the_stage_policy_matches_the_shapes_that_were_
//! measured`, which pins the routing decision this gate proves is the right
//! one, and `ltxv::clipmetric`'s unit tests, which pin the metric itself.
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! BRAIN_LTXV_UPSAMPLER_SPATIAL=<...latent-spatial-upscaler-x2-bf16-1.0.safetensors> \
//! cargo test --release -p brain-ltxv --test clip_stability_real -- --ignored --nocapture
//! ```
//!
//! Setting `BRAIN_LTXV_TWO_STAGE=0` runs the same shape the broken way and
//! is how the bound above was calibrated against a real failure rather than
//! against an assumption.

use ltxv::clipmetric::{blowup_ratio, frame_to_frame_diffs};
use ltxv::pipeline::{generate, GenOpts, Paths, SINGLE_STAGE_MAX_TOKENS};

/// The real bug's own shape: 25 frames at 1920x1088 is 8160 video tokens,
/// past [`SINGLE_STAGE_MAX_TOKENS`], and 4 latent frames - enough clip for a
/// last-latent-frame collapse to have somewhere to show.
const FRAMES: usize = 25;
const WIDTH: usize = 1920;
const HEIGHT: usize = 1088;
const FPS: usize = 24;
const SEED: u64 = 42;

/// A prompt with real motion in it: a static answer would be wrong, and a
/// clip that never moves would score a meaningless 1.0 here (see
/// [`blowup_ratio`]'s own doc on the zero-motion case, which this asserts
/// against separately below).
const PROMPT: &str = "a Belgian Malinois running at super speed alongside a flying NVIDIA P40 GPU with wings, camera tracking at the dog's speed, motion-blurred background";

/// Calibrated by running this exact shape BOTH ways (see this file's doc):
/// every healthy real clip measured on this port scores 1.02-1.06, the
/// single-stage 1080p defect scores 14.66. `4.0` is ~4x above anything
/// healthy and ~3.7x below the defect.
const BOUND: f32 = 4.0;

/// A clip has to have motion for the ratio to mean anything - a frozen clip
/// scores 1.0 trivially. Every real generation measured here has a median
/// frame-to-frame difference of at least 1.5 in 0-255 units.
const MIN_MEDIAN_MOTION: f32 = 0.5;

fn real_paths() -> Option<Paths> {
    let p = Paths::resolve(None, None, None).ok()?;
    p.dit.as_ref()?;
    Some(p)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

#[test]
#[ignore = "one full real 1080p generation, ~20 minutes on a P40 pair"]
fn a_real_1080p_clip_does_not_disintegrate_before_it_ends() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE to the real LTX-2.5 checkpoints");
    };
    let tokens = (FRAMES.div_ceil(8)) * (HEIGHT / 32) * (WIDTH / 32);
    assert!(
        tokens > SINGLE_STAGE_MAX_TOKENS,
        "this gate exists for the ABOVE-ceiling case; {WIDTH}x{HEIGHT} is {tokens} tokens against a ceiling of {SINGLE_STAGE_MAX_TOKENS}"
    );
    if paths.spatial_upsampler.is_none() && std::env::var("BRAIN_LTXV_TWO_STAGE").as_deref() != Ok("0") {
        return brain_testutil::skip("set BRAIN_LTXV_UPSAMPLER_SPATIAL to the real spatial x2 latent upscaler");
    }

    let o = GenOpts {
        frames: FRAMES,
        width: WIDTH,
        height: HEIGHT,
        fps: FPS,
        seed: SEED,
        guidance: 3.0,
        dit_config: "ltx25_22b".into(),
        ..GenOpts::default()
    };
    let cancel = capability::CancelToken::default();
    let (video, timings) = generate(&paths, PROMPT, &o, &cancel, |_, _, _| {}).unwrap_or_else(|e| panic!("generate failed: {e}"));
    assert_eq!(video.frames.len(), FRAMES, "frame count");

    // `Video::frames` is per-frame interleaved RGB; `frame_to_frame_diffs`
    // takes the decoder's own `[3, frames, h, w]` plane-major `[-1, 1]`
    // layout, so convert rather than re-deriving the metric here - one
    // definition of the observable, shared with the bench tool that measured
    // the table in this file's doc.
    let mut chw = vec![0f32; 3 * FRAMES * HEIGHT * WIDTH];
    for (f, frame) in video.frames.iter().enumerate() {
        for c in 0..3 {
            let base = (c * FRAMES + f) * HEIGHT * WIDTH;
            for i in 0..HEIGHT * WIDTH {
                chw[base + i] = frame[i * 3 + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    let diffs = frame_to_frame_diffs(&chw, FRAMES, HEIGHT, WIDTH);
    let ratio = blowup_ratio(&diffs);
    let med = median(diffs.clone());
    eprintln!("{:.1}s, {tokens} tokens, median frame-to-frame {med:.3}, blowup ratio {ratio:.2}", timings.total());
    eprintln!("per-frame: {}", diffs.iter().map(|d| format!("{d:.2}")).collect::<Vec<_>>().join(" "));

    assert!(med >= MIN_MEDIAN_MOTION, "the clip barely moves ({med:.3} < {MIN_MEDIAN_MOTION}), so its blowup ratio says nothing - check the prompt reached the model");
    assert!(
        ratio < BOUND,
        "the clip disintegrates part-way through: blowup ratio {ratio:.2} against a bound of {BOUND} (median frame-to-frame {med:.3}). \
         Per-frame differences: {}",
        diffs.iter().map(|d| format!("{d:.2}")).collect::<Vec<_>>().join(" ")
    );
}
