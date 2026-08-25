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
//! 1. The REFINEMENT PLAN ([`a_clip_that_fits_is_one_pass_and_no_seam`] and
//!    its three siblings), which is what stands between a long clip and an
//!    hour of wasted device time - and, since a clip that does not fit is
//!    refined in several passes, what decides whether those passes are one
//!    continuous clip or a pile of independently re-imagined ones. Pure
//!    arithmetic, no weights, always runs.
//! 2. The WIRING end to end
//!    ([`real_weights::a_multi_pass_upscale_is_one_clip_that_carries_its_own_latent_context`])
//!    on the real VAE and the real spatial upscaler - tiny random-weight DiT,
//!    so it says nothing about quality, the same disclaimer `ltxv::pipeline`'s
//!    own module doc carries for every tiny-config path.
//!
//! What is deliberately NOT re-gated here: that the upscaler is un-normalized
//! around correctly. `upscale` reaches the upscaler through the same
//! `upscale_and_refine` the internal two-stage generation path uses, and that
//! sandwich already has an exact gate in `upsampler_parity.rs`
//! (`the_upscaler_is_un_normalized_around_exactly_as_the_reference_does_it`).
//! Nor that the carried tail is a bit-exact slice of the previous pass's own
//! latent: `upscale` carries it with the same `longform::carry_tail`
//! `generate_long` does, and `longform.rs`'s
//! `the_carried_tail_is_the_previous_windows_own_last_latent_frames` gates
//! that function. Asserting either again here would gate a second copy that
//! does not exist.

use ltxv::longform::{Window, CONTEXT_LATENT_FRAMES};
use ltxv::pipeline::{refine_plan, Video, REFINE_MAX_TOKENS};

/// Reassemble a refinement plan the way [`ltxv::pipeline::upscale`] does -
/// every pass emits only the pixel frames its NEW latent frames cover, the
/// leading frames its carried context covers being decoded and dropped - and
/// say which output frames came out, in order.
fn reassembled(plan: &[Window]) -> Vec<usize> {
    let mut out = Vec::new();
    for w in plan {
        out.extend(w.first_frame..w.first_frame + w.emitted_frames());
    }
    out
}

/// A clip whose refinement pass fits under [`REFINE_MAX_TOKENS`] must be ONE
/// pass covering the whole clip - so the common case takes exactly the shape
/// the internal two-stage path's refinement already takes, with no carried
/// context to pay for and no seam anywhere.
#[test]
fn a_clip_that_fits_is_one_pass_and_no_seam() {
    // 25 frames at 1920x1088 out: 4 latent frames x 34 x 60 = 8160 tokens,
    // the largest refinement this crate has a recorded real run at.
    let plan = refine_plan(25, 34, 60, CONTEXT_LATENT_FRAMES, REFINE_MAX_TOKENS).expect("8160 tokens is under the ceiling");
    assert_eq!(plan.len(), 1, "a clip that fits must not be split: {plan:?}");
    assert_eq!(plan[0].context, 0, "a single pass has nothing to carry");
    assert_eq!(plan[0].emitted_frames(), 25);
    assert_eq!(reassembled(&plan), (0..25).collect::<Vec<_>>());
}

/// The one that matters, on the shape that exposed the defect: 217 frames
/// upscaled from 1280x704 to 2560x1408. A clip too long for one refinement
/// pass is refined in several, and every pass after the first must carry the
/// PREVIOUS pass's own refined latent frames as a frozen prefix. A plan whose
/// continuation passes carry nothing is a plan for N independently re-imagined
/// clips, which is what a 0.909-sigma refinement start produces when it is
/// given no history.
#[test]
fn a_clip_too_long_to_refine_in_one_pass_carries_real_latent_context_across_every_seam() {
    let (frames, lh, lw) = (217usize, 44usize, 80usize); // 1280x704 -> 2560x1408
    let plan = refine_plan(frames, lh, lw, CONTEXT_LATENT_FRAMES, REFINE_MAX_TOKENS).expect("a long clip plans, it does not fail");
    assert!(plan.len() > 1, "28 latent frames x 3520 tokens cannot be one pass, got {plan:?}");

    assert_eq!(plan[0].context, 0, "the first pass has no predecessor to carry from");
    let carried = plan[1].context;
    assert!(carried >= 1, "every pass after the first refines with NO history - this is the defect, not a plan: {plan:?}");
    for (i, w) in plan.iter().enumerate() {
        assert!(w.new >= 1, "pass {i} refines nothing: {w:?}");
        if i > 0 {
            assert_eq!(w.context, carried, "pass {i} carries a different amount than its siblings: {w:?}");
            assert!(plan[i - 1].latent_frames() >= w.context, "pass {} has only {} latent frames and cannot supply pass {i}'s {}-frame context", i - 1, plan[i - 1].latent_frames(), w.context);
        }
        assert!(w.tokens(lh, lw) <= REFINE_MAX_TOKENS, "pass {i} is {} tokens, over the {REFINE_MAX_TOKENS} ceiling", w.tokens(lh, lw));
        assert_eq!((w.decoded_frames() - 1) % 8, 0, "pass {i} decodes {} frames, which is not 1 + 8k", w.decoded_frames());
        // The source range this pass VAE-encodes: its whole decode, context
        // included, read out of the input clip. It has to be inside the clip,
        // and it has to end exactly where the pass's own output ends.
        assert!(w.source_first_frame() + w.decoded_frames() <= frames, "pass {i} reads past the end of the clip: {w:?}");
        assert_eq!(w.source_first_frame() + w.decoded_frames(), w.first_frame + w.emitted_frames(), "pass {i} reads a source range its own output does not end with: {w:?}");
    }
    assert_eq!(reassembled(&plan), (0..frames).collect::<Vec<_>>(), "the passes do not reassemble to the input clip");
}

/// A refinement grid carries FOUR times the tokens per latent frame that the
/// generation grid it came from does, so the full
/// [`CONTEXT_LATENT_FRAMES`]-frame context does not always fit under
/// [`REFINE_MAX_TOKENS`]. Where it fits it is taken whole; where it does not,
/// the plan carries the most the ceiling allows rather than refusing the clip
/// or silently carrying nothing.
#[test]
fn the_carried_context_shrinks_to_the_grid_rather_than_vanishing_or_refusing() {
    // 1280x704 out: 880 tokens per latent frame, so 13 latent frames fit and
    // the reference's own 8-frame context is affordable in full.
    let roomy = refine_plan(481, 22, 40, CONTEXT_LATENT_FRAMES, REFINE_MAX_TOKENS).expect("a 1280x704 refinement plans");
    assert!(roomy.len() > 1, "481 frames at 880 tokens per latent frame has to split: {roomy:?}");
    assert!(roomy[1..].iter().all(|w| w.context == CONTEXT_LATENT_FRAMES), "a grid with room for the full context must take it: {roomy:?}");

    // 2560x1408 out: 3520 tokens per latent frame, so ONE pass holds 3 latent
    // frames and 8 carried frames is arithmetically impossible.
    let tight = refine_plan(217, 44, 80, CONTEXT_LATENT_FRAMES, REFINE_MAX_TOKENS).expect("a 2560x1408 refinement plans rather than refusing");
    assert_eq!(tight[1].context, REFINE_MAX_TOKENS / (44 * 80) - 1, "the tight grid must carry every latent frame it can, leaving exactly one to refine: {tight:?}");
}

/// A frame count the causal VAE cannot represent is refused before any weight
/// is read, and so is an output grid so tight that not even one carried frame
/// plus one new one fits - the caller gets told, rather than getting an
/// out-of-memory abort an hour in.
#[test]
fn an_impossible_request_is_refused_up_front() {
    assert!(refine_plan(24, 4, 4, CONTEXT_LATENT_FRAMES, REFINE_MAX_TOKENS).is_err(), "24 frames is not 1 + 8k");
    assert!(refine_plan(0, 4, 4, CONTEXT_LATENT_FRAMES, REFINE_MAX_TOKENS).is_err(), "0 frames is not a clip");
    // One latent frame past the first already exceeds the ceiling here, so no
    // split can rescue this shape.
    assert!(refine_plan(9, REFINE_MAX_TOKENS, 1, CONTEXT_LATENT_FRAMES, REFINE_MAX_TOKENS).is_err(), "a grid this wide cannot fit even 2 latent frames");
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
        Video { width: w as u32, height: h as u32, fps: 8, frames: px, audio: None }
    }

    /// End to end on the real VAE and the real x2 spatial upscaler: a clip
    /// that needs SEVERAL refinement passes comes back at exactly twice the
    /// size, with exactly the frames it went in with, having gone through the
    /// official upscaler once per pass and carried a real latent context
    /// across the seam between them.
    ///
    /// The token ceiling is forced down to a value a 64x64 clip actually
    /// crosses, the way `longform.rs`'s own wiring test forces one, so the
    /// multi-pass path runs at a size CPU can finish. The DiT is the tiny
    /// random-weight config, so this is a WIRING claim - that the plan, the
    /// per-pass encode of the right source range, the latent carry and the
    /// reassembly compose - and not a quality one. CPU device deliberately:
    /// what this gates is device-independent, and the box's cards are not this
    /// test's to reserve.
    #[test]
    fn a_multi_pass_upscale_is_one_clip_that_carries_its_own_latent_context() {
        let (Some(vae), Some(ups)) = (
            weights_path("BRAIN_LTXV_VAE", "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            weights_path("BRAIN_LTXV_UPSAMPLER_SPATIAL", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        ) else {
            return brain_testutil::skip("set BRAIN_LTXV_VAE + BRAIN_LTXV_UPSAMPLER_SPATIAL to the real checkpoints");
        };
        let paths = Paths::resolve(Some(&vae), None, None, Some(&ups)).expect("the two configured paths resolve");
        // 128x128 out is a 4x4 latent grid, so one latent frame is 16 tokens:
        // the ceiling has to come down for a smoke-sized clip to need splitting.
        let (frames, max_tokens) = (25usize, 48usize);
        let clip = synthetic_clip(frames, 64, 64);
        let o = UpscaleOpts { max_refine_tokens: max_tokens, base: GenOpts { device: Some("cpu".into()), seed: 3, ..GenOpts::default() }, ..UpscaleOpts::default() };
        let plan = refine_plan(frames, 4, 4, o.context_latent_frames, max_tokens).expect("the plan is legal");
        assert!(plan.len() > 1, "this shape has to need several passes for the test to mean anything: {plan:?}");
        assert!(plan[1].context >= 1, "the seam this test exists to exercise carries nothing: {plan:?}");

        let mut phases: Vec<String> = Vec::new();
        let (out, timings) = upscale(&paths, "a moving bar", &clip, &o, &capability::CancelToken::default(), |_, _, phase| phases.push(phase.to_string())).expect("upscale");

        assert_eq!((out.width, out.height), (128, 128), "the clip did not double");
        assert_eq!(out.frames.len(), clip.frames.len(), "the frame count changed");
        assert!(out.frames.iter().all(|f: &Vec<u8>| f.len() == 128 * 128 * 3), "a frame is the wrong size");
        assert!(out.frames.iter().any(|f: &Vec<u8>| f.iter().any(|&v| v != f[0])), "every frame is a flat colour - nothing was decoded");
        assert_eq!(out.fps, clip.fps, "fps must survive the round trip");
        assert_eq!(phases.iter().filter(|p| p.as_str() == "spatial upscale").count(), plan.len(), "the official upscaler must run once per pass: {phases:?}");
        assert_eq!(phases.iter().filter(|p| p.as_str() == "vae decode").count(), plan.len(), "one decode per pass: {phases:?}");
        assert_eq!(timings.steps, LTX2_STAGE2_STEPS * plan.len(), "the full refinement schedule did not run on every pass");
    }
}
