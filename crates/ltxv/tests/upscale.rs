// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ltxv::pipeline::upscale` - taking a clip that already finished rendering
//! back through VAE-encode -> official x2 latent spatial upscale ->
//! refinement denoise -> VAE-decode.
//!
//! Swedish Embedded AB implements post-hoc latent-space video upscaling for
//! its clients. If your team needs expertise in memory-bounded diffusion
//! video pipelines, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Two claims, and they are deliberately the only two:
//!
//! 1. The SEGMENT PLAN ([`a_clip_that_fits_is_one_segment_and_no_seam`] and
//!    its two siblings), which is what stands between a long clip and an hour
//!    of wasted device time. Pure arithmetic, no weights, always runs.
//! 2. The WIRING end to end
//!    ([`real_weights::an_upscaled_clip_is_twice_the_size_with_the_same_frames`])
//!    on the real VAE and the real spatial upscaler - tiny random-weight DiT,
//!    so it says nothing about quality, the same disclaimer `ltxv::pipeline`'s
//!    own module doc carries for every tiny-config path.
//!
//! What is deliberately NOT re-gated here: that the upscaler is un-normalized
//! around correctly. `upscale` reaches the upscaler through the same
//! `upscale_and_refine` the internal two-stage generation path uses, and that
//! sandwich already has an exact gate in `upsampler_parity.rs`
//! (`the_upscaler_is_un_normalized_around_exactly_as_the_reference_does_it`).
//! Asserting it again here would gate a second copy that does not exist.

use ltxv::pipeline::{refine_segments, Video, REFINE_MAX_TOKENS};

/// Reassemble a segment plan the way [`ltxv::pipeline::upscale`] does - every
/// segment after the first re-renders its predecessor's LAST frame, so that
/// duplicate is dropped - and say which pixel frames came out, in order.
fn reassembled(plan: &[(usize, usize)]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, &(start, len)) in plan.iter().enumerate() {
        out.extend(start + usize::from(i > 0)..start + len);
    }
    out
}

/// A clip whose refinement pass fits under [`REFINE_MAX_TOKENS`] must be ONE
/// segment covering the whole clip - so the common case takes exactly the
/// shape the internal two-stage path's refinement already takes, with no seam
/// anywhere.
#[test]
fn a_clip_that_fits_is_one_segment_and_no_seam() {
    // 25 frames at 1920x1088 out: 4 latent frames x 34 x 60 = 8160 tokens,
    // the largest refinement this crate has a recorded real run at.
    let plan = refine_segments(25, 34, 60).expect("8160 tokens is under the ceiling");
    assert_eq!(plan, vec![(0, 25)], "a clip that fits must not be split");
    assert_eq!(reassembled(&plan), (0..25).collect::<Vec<_>>());
}

/// The one that matters: a clip whose refinement would need more than
/// [`REFINE_MAX_TOKENS`] in a single pass is broken into consecutive segments
/// that each fit, each of which is a legal `1 + 8k` clip for the causal VAE,
/// and which reassemble to exactly the input frames in order.
#[test]
fn a_clip_too_long_to_refine_in_one_pass_is_segmented_rather_than_attempted() {
    // 105 frames upscaled from 1280x704 to 2560x1408: 14 latent frames x 44 x
    // 80 = 49280 tokens in one pass, four times the ceiling.
    let (frames, lh, lw) = (105, 44, 80);
    let plan = refine_segments(frames, lh, lw).expect("a long clip segments, it does not fail");
    assert!(plan.len() > 1, "49280 tokens cannot be one pass, got {plan:?}");
    for &(start, len) in &plan {
        assert_eq!((len - 1) % 8, 0, "segment ({start}, {len}) is not 1 + 8k, which the causal VAE requires");
        let lat_t = 1 + (len - 1) / 8;
        assert!(lat_t * lh * lw <= REFINE_MAX_TOKENS, "segment ({start}, {len}) is {} tokens, over the {REFINE_MAX_TOKENS} ceiling", lat_t * lh * lw);
        assert!(start + len <= frames, "segment ({start}, {len}) runs past the clip");
    }
    assert_eq!(reassembled(&plan), (0..frames).collect::<Vec<_>>(), "the segments do not reassemble to the input clip");
}

/// A frame count the causal VAE cannot represent is refused before any weight
/// is read, and so is an output grid so wide that not even the shortest legal
/// segment fits - the caller gets told, rather than getting an out-of-memory
/// abort an hour in.
#[test]
fn an_impossible_request_is_refused_up_front() {
    assert!(refine_segments(24, 4, 4).is_err(), "24 frames is not 1 + 8k");
    assert!(refine_segments(0, 4, 4).is_err(), "0 frames is not a clip");
    // One latent frame past the first already exceeds the ceiling here, so no
    // segmentation can rescue this shape.
    assert!(refine_segments(9, REFINE_MAX_TOKENS, 1).is_err(), "a grid this wide cannot fit even 2 latent frames");
}

mod real_weights {
    use super::*;
    use std::path::Path;

    use ltxv::pipeline::{upscale, GenOpts, Paths, UpscaleOpts, LTX2_STAGE2_STEPS};

    /// The named environment variable, else the repo-relative
    /// `resources/ltxv/weights/` the real files ship under - never a literal
    /// machine path, the same convention `upsampler_parity.rs`/`vae_parity.rs`
    /// already use, so this passes on any checkout that fetched the resource.
    fn weights_path(env: &str, rel: &str) -> Option<String> {
        if let Ok(p) = std::env::var(env) {
            return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
        }
        let p = format!("{}/../../resources/ltxv/weights/{rel}", env!("CARGO_MANIFEST_DIR"));
        Path::new(&p).exists().then_some(p)
    }

    /// A synthetic clip with real spatial structure (a moving bar over a
    /// two-axis gradient), so the VAE encode has something other than noise to
    /// compress and the decode has something recognisable to get wrong.
    fn synthetic_clip(frames: usize, w: usize, h: usize) -> Video {
        let px = (0..frames)
            .map(|f| {
                let mut buf = vec![0u8; w * h * 3];
                for y in 0..h {
                    for x in 0..w {
                        let bar = usize::from(x.abs_diff(f * w / frames) < w / 8);
                        let i = (y * w + x) * 3;
                        buf[i] = (255 * x / w) as u8;
                        buf[i + 1] = (255 * y / h) as u8;
                        buf[i + 2] = (bar * 255) as u8;
                    }
                }
                buf
            })
            .collect();
        Video { width: w as u32, height: h as u32, fps: 8, frames: px }
    }

    /// End to end on the real VAE and the real x2 spatial upscaler: the clip
    /// comes back at exactly twice the size, with exactly the frames it went
    /// in with, having gone through the official upscaler and the full
    /// refinement schedule.
    ///
    /// The DiT is the tiny random-weight config, so this is a WIRING claim and
    /// not a quality one. CPU device deliberately: what this gates is
    /// device-independent, and the box's cards are not this test's to reserve.
    #[test]
    fn an_upscaled_clip_is_twice_the_size_with_the_same_frames() {
        let (Some(vae), Some(ups)) = (
            weights_path("BRAIN_LTXV_VAE", "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            weights_path("BRAIN_LTXV_UPSAMPLER_SPATIAL", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        ) else {
            return brain_testutil::skip("set BRAIN_LTXV_VAE + BRAIN_LTXV_UPSAMPLER_SPATIAL to the real checkpoints");
        };
        let paths = Paths::resolve(Some(&vae), None, None, Some(&ups)).expect("the two configured paths resolve");
        let clip = synthetic_clip(9, 64, 64);
        let o = UpscaleOpts { base: GenOpts { device: Some("cpu".into()), seed: 3, ..GenOpts::default() }, ..UpscaleOpts::default() };

        let mut phases: Vec<String> = Vec::new();
        let (out, timings) = upscale(&paths, "a moving bar", &clip, &o, &capability::CancelToken::default(), |_, _, phase| phases.push(phase.to_string())).expect("upscale");

        assert_eq!((out.width, out.height), (128, 128), "the clip did not double");
        assert_eq!(out.frames.len(), clip.frames.len(), "the frame count changed");
        assert!(out.frames.iter().all(|f: &Vec<u8>| f.len() == 128 * 128 * 3), "a frame is the wrong size");
        assert!(out.frames.iter().any(|f: &Vec<u8>| f.iter().any(|&v| v != f[0])), "every frame is a flat colour - nothing was decoded");
        assert_eq!(out.fps, clip.fps, "fps must survive the round trip");
        assert!(phases.iter().any(|p| p == "spatial upscale"), "the official upscaler never ran: {phases:?}");
        assert_eq!(timings.steps, LTX2_STAGE2_STEPS, "the full refinement schedule did not run");
    }
}
