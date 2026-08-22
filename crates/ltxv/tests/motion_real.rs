// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The clip has to MOVE.** A real-weight gate on the one property no
//! shape check, no cosine-similarity parity gate and no weight-free
//! geometry test in this crate can see: that `ltxv::pipeline::generate`'s
//! decoded frames are a video and not the same picture nine times.
//!
//! Every other real-weight gate here compares one forward against a golden
//! tensor, and all of them stayed green through a conditioning defect that
//! made every generated frame a mesh texture. What finally caught THAT was
//! a person looking at the frames; what nearly missed the follow-up was a
//! metric - see [`peak_excursion`].
//!
//! ## What is gated, and what is measured-but-not-gated
//!
//! Gated (both cases below): unconditioned text-to-video, and keyframe
//! interpolation between two DIFFERENT stills. Both must move.
//!
//! Deliberately NOT gated, because it is the model's honest answer rather
//! than a defect: conditioning the SAME still at both ends (a seamless
//! loop). Measured on the real 22B Q8_0 checkpoint + real Gemma-4 encoder,
//! [`peak_excursion`], seed 42, same prompt throughout:
//!
//! | request | 384x192 9f | 384x192 25f | 640x320 25f | 640x320 49f |
//! |---|---|---|---|---|
//! | text-to-video, no stills | 26.6 | 18.1 | 20.5 | - |
//! | one still at frame 0 | 35.1 | - | - | - |
//! | two DIFFERENT stills | 35.2 | - | 40.0 | - |
//! | the SAME still at both ends | 4.6 | 32.4 | 7.3 | 7.2 |
//!
//! Every run at or above 18.1 visibly animates; every run at or below 8.8
//! reproduces the pinned still in every frame. The split is that clean.
//!
//! The loop row is the only static one, and it stays static under every
//! lever the reference exposes: `conditioning_strength` 1.0/0.8/0.5 (4.6 /
//! 4.8 / 5.0), the deterministic and the ancestral sampler (4.6 / 5.1, and
//! 7.3 / 8.8 at 640x320), the in-place and the appended frame-0
//! conditioning mechanism (5.1 / 4.6), a CRF-33 re-compressed conditioning
//! still (8.4 vs 7.3), and a clip twice as long (7.2). The decisive control
//! is the last column pair: at ONE shape, one seed, one prompt, changing
//! only the end still from "the same image" to "that image mirrored" moves
//! the score from 7.3 to 40.0. "Start at this image and end at this same
//! image" has a correct, trivial solution and the model returns it. Gating
//! the loop case would be gating a request, not a regression.
//!
//! `#[ignore]`d: each case is two full real 22B generations (~10 minutes on
//! a P40, weight-streaming bound). Run explicitly:
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! BRAIN_LTXV_TEXT_ENCODER=<...gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
//! cargo test -p brain-ltxv --test motion_real -- --ignored --nocapture
//! ```
//!
//! Weights are resolved exactly the way `brain ltxv t2v` resolves them
//! ([`ltxv::pipeline::Paths::resolve`]); a missing checkpoint SKIPs.

use ltxv::pipeline::{generate, GenOpts, Paths};

/// The shape every threshold below was calibrated at. Small on purpose: 9
/// frames is 2 latent frames, the least room a conditioned clip can have,
/// so a floor that holds here holds at longer shapes too.
const FRAMES: usize = 9;
const WIDTH: usize = 384;
const HEIGHT: usize = 192;
const SEED: u64 = 42;
const FPS: usize = 8;

/// A motion-heavy prompt. The subject matters less than the fact that a
/// static answer would be wrong for it.
const PROMPT: &str = "a belgian malinois dog running at full speed along a mountain road, a small winged P40 graphics card flying beside it, camera tracking";

/// Mean absolute per-channel difference between two decoded frames, in
/// 0-255 pixel units.
fn frame_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64
}

/// **The metric this gate is built on:** how far the clip ever gets from
/// its own first frame, `max_i mean|frame_i - frame_0|`.
///
/// Not the mean consecutive-frame delta, which is what an earlier
/// investigation gated on and which is close to useless here. A visually
/// FROZEN clip - every frame the same dog in the same fully-extended pose -
/// scores 2.2 on that metric, because VAE decode dither on static content
/// is not zero; a real one scores 5.4. A 2.4x gap invites the reading
/// "lower, but a nonzero delta, so it is moving". Peak excursion separates
/// those same two runs 26.6 vs 5.1, because a static clip cannot
/// accumulate: its per-frame wobble is uncorrelated noise around one image,
/// while real motion walks away from frame 0 and stays away.
///
/// A clip that returns to its start by design (a loop) would defeat a
/// *last*-frame probe, which is why this is the peak over the whole clip
/// and not `|frame_last - frame_0|`.
fn peak_excursion(frames: &[Vec<u8>]) -> f64 {
    frames.iter().map(|f| frame_delta(f, &frames[0])).fold(0.0f64, f64::max)
}

/// Mean consecutive-frame delta - reported, never gated on (see
/// [`peak_excursion`]).
fn mean_consecutive(frames: &[Vec<u8>]) -> f64 {
    if frames.len() < 2 {
        return 0.0;
    }
    frames.windows(2).map(|w| frame_delta(&w[0], &w[1])).sum::<f64>() / (frames.len() - 1) as f64
}

fn base_opts() -> GenOpts {
    GenOpts {
        frames: FRAMES,
        width: WIDTH,
        height: HEIGHT,
        seed: SEED,
        fps: FPS,
        // The distilled checkpoint's own fixed schedule ignores `steps`.
        // `eta = 0` is the deterministic sampler; it is used here (rather
        // than the distilled pipeline's own ancestral default) so both
        // cases are reproducible run to run, which is what a threshold
        // needs.
        eta: 0.0,
        guidance: 1.0,
        dit_config: "ltx25_22b".into(),
        device: Some("gpu".into()),
        ..GenOpts::default()
    }
}

/// The real checkpoints, or `None` (SKIP) - same resolution order the CLI
/// uses.
fn real_paths() -> Option<Paths> {
    let p = Paths::resolve(None, None, None, None).ok()?;
    p.dit.as_ref()?;
    Some(p)
}

fn run(o: &GenOpts, paths: &Paths, label: &str) -> Vec<Vec<u8>> {
    let cancel = capability::CancelToken::default();
    let (video, timings) = generate(paths, PROMPT, o, &cancel, |_, _, _| {}).unwrap_or_else(|e| panic!("{label}: generate failed: {e}"));
    assert_eq!(video.frames.len(), FRAMES, "{label}: frame count");
    eprintln!("{label}: {:.1}s, peak excursion {:.2}, mean consecutive {:.2}", timings.total(), peak_excursion(&video.frames), mean_consecutive(&video.frames));
    video.frames
}

/// Write one decoded frame out as a PNG a conditioning run can read back.
fn write_still(dir: &std::path::Path, name: &str, frame: &[u8]) -> String {
    std::fs::create_dir_all(dir).expect("temp dir");
    let path = dir.join(name);
    image::RgbImage::from_raw(WIDTH as u32, HEIGHT as u32, frame.to_vec()).expect("decoded frame as an RGB image").save(&path).expect("write the conditioning still");
    path.to_string_lossy().into_owned()
}

/// Calibrated on the real 22B Q8_0 DiT + real Gemma-4 encoder at the shape
/// above (see this file's table): everything that moves scores 18.1 or
/// better, everything frozen scores 8.8 or worse. `12.0` sits clear of
/// both, i.e. above anything a static clip's decode dither can reach and
/// far below what a moving one produces.
const PEAK_FLOOR: f64 = 12.0;

#[test]
#[ignore = "full real 22B generation, ~5 minutes"]
fn unconditioned_text_to_video_actually_moves() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE to the real LTX-2.5 checkpoints");
    };
    let frames = run(&base_opts(), &paths, "t2v");
    let peak = peak_excursion(&frames);
    assert!(peak >= PEAK_FLOOR, "text-to-video produced a static clip: peak excursion from frame 0 is {peak:.2}, floor {PEAK_FLOOR:.2} (mean consecutive {:.2})", mean_consecutive(&frames));
}

/// Keyframe interpolation must still GENERATE the frames in between, not
/// hold a conditioned one.
///
/// The two anchors are this pipeline's own first and last decoded frames
/// from the unconditioned run above - two genuinely different instants of
/// one coherent shot, in distribution, at exactly the target resolution, so
/// the gate needs no image fixture and cannot be blamed on a resize or an
/// out-of-distribution input. A correct run has to travel at least as far
/// as the clip those anchors came from did.
///
/// This is the case that regressed twice: once into mesh texture (frozen
/// tokens announced at the schedule's sigma instead of their own zero), and
/// once into a suspicion that conditioning suppresses motion outright. It
/// does not - see this file's table for the one shape that legitimately
/// stays static.
#[test]
#[ignore = "two full real 22B generations, ~10 minutes"]
fn keyframe_interpolation_generates_motion_between_two_stills() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE to the real LTX-2.5 checkpoints");
    };
    let seed_frames = run(&base_opts(), &paths, "t2v (source of the two anchors)");
    let reference_peak = peak_excursion(&seed_frames);

    let dir = std::env::temp_dir().join(format!("ltxv-motion-gate-{}", std::process::id()));
    let start = write_still(&dir, "start.png", &seed_frames[0]);
    let end = write_still(&dir, "end.png", seed_frames.last().expect("non-empty"));

    let o = GenOpts { start_frame: Some(start), end_frame: Some(end), ..base_opts() };
    let frames = run(&o, &paths, "keyframe interpolation");
    let _ = std::fs::remove_dir_all(&dir);

    let peak = peak_excursion(&frames);
    assert!(
        peak >= PEAK_FLOOR,
        "a clip conditioned on two different stills is frozen: peak excursion from frame 0 is {peak:.2}, floor {PEAK_FLOOR:.2} \
         (mean consecutive {:.2}; the unconditioned clip the anchors came from reached {reference_peak:.2}). \
         A nonzero consecutive delta is NOT evidence of motion - see this file's `peak_excursion` doc.",
        mean_consecutive(&frames)
    );
}
