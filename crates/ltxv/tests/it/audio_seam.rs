// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What has to be true for LTX-2.5's audio stream to cross a denoising-window
//! seam without drifting against the picture.
//!
//! Swedish Embedded AB implements time-exact multi-stream generative pipelines
//! for its clients. If your team needs audio and video that stay in sync
//! across a rolling generation boundary, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! The failure this file exists to catch is not a crash and not a shape
//! error. The two streams run at different time resolutions - a video latent
//! frame is `VAE_TEMPORAL_SCALE / fps` seconds and an audio token is
//! `1 / LATENT_RATE` seconds - so a seam carried with the wrong number of
//! audio tokens produces a clip that decodes, plays, and is progressively out
//! of sync. Every assertion here is integer arithmetic on the SAME functions
//! the pipeline calls, so a half-token error has nowhere to hide:
//!
//! 1. [`every_seam_shifts_both_streams_by_exactly_the_same_amount_of_time`] -
//!    the property the whole design rests on, checked as an integer identity
//!    over a frame-count and frame-rate sweep of multi-window shapes.
//! 2. [`a_carried_token_lands_on_the_moment_of_the_picture_it_came_from`] -
//!    the same claim read off the two streams' real RoPE position tables,
//!    token by token and latent frame by latent frame.
//! 3. The container arithmetic
//!    ([`a_multi_window_clip_carries_exactly_the_tokens_a_single_window_one_would`]):
//!    the windows' contributions sum to the clip's own token count, so the two
//!    stream durations do not move.
//! 4. The refusal
//!    ([`a_plan_whose_seams_miss_the_token_grid_is_refused_rather_than_rounded`]),
//!    which is what makes 1-3 a rule rather than a hope.
//! 5. [`a_silent_clips_window_plan_is_exactly_what_it_always_was`] - the
//!    quantum must cost a video-only request nothing.

use ltxv::audio::{self, LATENT_RATE, TOKEN_DIM};
use ltxv::longform::{window_plan, window_plan_aligned, Window, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS};
use ltxv::pipeline::{real_pixel_positions, VAE_TEMPORAL_SCALE};

/// 1280x704 - the resolution this crate has a recorded real audio-visual run
/// at, and the one whose 880-token latent frames make `LONGFORM_MAX_TOKENS`
/// bite at 113 frames. Tokens per forward is what decides which code path
/// runs, so a sweep at a convenient smaller grid would be a different test.
const LH: usize = 22;
const LW: usize = 40;

/// The frame rates the CLI offers, plus one whose quantum is 1 (25) and one
/// whose quantum is neither (16), so the sweep covers a divisible rate, an
/// indivisible one, and the two the reference's own configs use.
const RATES: [usize; 4] = [24, 25, 30, 16];

/// Multi-window lengths at [`LH`]x[`LW`]: 121 is the shortest clip that needs
/// two windows there, and the ladder reaches past the 10 s target shape.
const FRAMES: [usize; 8] = [121, 129, 161, 193, 241, 289, 361, 481];

/// Where in the WHOLE clip's pixel timeline a window's own time origin sits -
/// its local latent frame 0. A window emits from `first_frame`, but its
/// sequence starts `dropped_frames` earlier, at the head of the context it
/// carried.
fn origin(w: &Window) -> usize {
    w.first_frame - w.dropped_frames()
}

fn plan_for(frames: usize, fps: usize) -> Vec<Window> {
    window_plan_aligned(frames, LH, LW, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS, audio::window_latent_frame_quantum(fps))
        .unwrap_or_else(|e| panic!("{frames} frames at {fps} fps has to plan: {e}"))
}

/// **The alignment rule, as an integer identity.**
///
/// A seam re-bases both streams onto the new window's own time origin. The
/// video's shift is `(origin(w+1) - origin(w)) / fps` seconds and the audio's
/// is `(tokens(w) - carried) / LATENT_RATE` seconds, and the carried content
/// only lands where it belongs if those are the SAME number. Cross-multiplied
/// so no float ever decides it: an error of half a token is an error of two
/// in the integers below, not a rounding difference.
#[test]
fn every_seam_shifts_both_streams_by_exactly_the_same_amount_of_time() {
    let mut seams = 0usize;
    for fps in RATES {
        for frames in FRAMES {
            let plan = plan_for(frames, fps);
            assert!(plan.len() > 1, "{frames} frames at {LW}x{LH} latent has to need several windows, or this sweep proves nothing");
            let a = audio::audio_plan(&plan, CONTEXT_LATENT_FRAMES, frames, fps).unwrap_or_else(|e| panic!("{frames} frames at {fps} fps: {e}"));
            for wi in 0..plan.len() - 1 {
                let video_pixels = origin(&plan[wi + 1]) - origin(&plan[wi]);
                let audio_tokens = a.per_window[wi] - a.context;
                assert_eq!(
                    video_pixels * LATENT_RATE as usize,
                    audio_tokens * fps,
                    "{frames} frames at {fps} fps, seam {wi}: the picture advances {video_pixels} pixel frames ({} s) but the sound advances {audio_tokens} tokens ({} s)",
                    video_pixels as f64 / fps as f64,
                    audio_tokens as f64 / f64::from(LATENT_RATE)
                );
                seams += 1;
            }
        }
    }
    assert!(seams >= 40, "the sweep covered only {seams} seams");
}

/// The same claim, read off the two streams' REAL position tables rather than
/// re-derived: a token carried across a seam has to name the same instant of
/// the picture in its new window that it named in its old one.
///
/// Both tables are causal - a window's first video latent frame covers one
/// pixel frame rather than eight and its first audio token one mel frame
/// rather than four - so it is the tokens' END bounds that are comparable
/// across a re-basing, and every carried token past the first has comparable
/// start bounds too. This is the assertion a one-token error in the carried
/// count fails, because it compares position VALUES, not counts.
///
/// **Do not weaken this into a count comparison.** That is the whole reason
/// it exists, and it is mutation-proven: shifting `AudioPlan::context` by one
/// token while leaving the clip's total correct keeps every count in this
/// suite self-consistent, so the total check and the per-window count checks
/// all still pass - and this test fails with the size of the error in
/// seconds ("the sound is -0.04 s away from the picture it belongs to"). A
/// carried count can be wrong in a way that still adds up; a carried
/// POSITION cannot.
#[test]
fn a_carried_token_lands_on_the_moment_of_the_picture_it_came_from() {
    for fps in RATES {
        for frames in FRAMES {
            let plan = plan_for(frames, fps);
            let a = audio::audio_plan(&plan, CONTEXT_LATENT_FRAMES, frames, fps).expect("the plan is aligned");
            for wi in 0..plan.len() - 1 {
                let shift = (origin(&plan[wi + 1]) - origin(&plan[wi])) as f64 / fps as f64;
                let (prev, next) = (&plan[wi], &plan[wi + 1]);

                // The video half, one latent frame per carried frame. Its
                // first carried frame keeps only its end bound (the causal
                // re-basing narrows its start deliberately).
                let (pv, nv) = (real_pixel_positions(prev.latent_frames(), 1, 1, fps as f64), real_pixel_positions(next.latent_frames(), 1, 1, fps as f64));
                for f in 0..next.context {
                    let src = prev.latent_frames() - next.context + f;
                    assert!(
                        (f64::from(pv[src * 2 + 1]) - f64::from(nv[f * 2 + 1]) - shift).abs() < 1e-6,
                        "{frames} frames at {fps} fps, seam {wi}: carried latent frame {f} ends at {} in its new window and {} in its old one, a shift of {shift}",
                        nv[f * 2 + 1],
                        pv[src * 2 + 1]
                    );
                    if f > 0 {
                        assert!((f64::from(pv[src * 2]) - f64::from(nv[f * 2]) - shift).abs() < 1e-6, "{frames} frames at {fps} fps, seam {wi}: carried latent frame {f} starts at the wrong moment");
                    }
                }

                // The audio half, on its own grid, against the SAME shift.
                let (pa, na) = (audio::positions(a.per_window[wi]), audio::positions(a.per_window[wi + 1]));
                for i in 0..a.context {
                    let src = a.per_window[wi] - a.context + i;
                    assert!(
                        (f64::from(pa[src * 2 + 1]) - f64::from(na[i * 2 + 1]) - shift).abs() < 1e-6,
                        "{frames} frames at {fps} fps, seam {wi}: carried audio token {i} ends at {} in its new window and {} in its old one, a shift of {shift} - the sound is {} s away from the picture it belongs to",
                        na[i * 2 + 1],
                        pa[src * 2 + 1],
                        f64::from(pa[src * 2 + 1]) - f64::from(na[i * 2 + 1]) - shift
                    );
                    if i > 0 {
                        assert!((f64::from(pa[src * 2]) - f64::from(na[i * 2]) - shift).abs() < 1e-6, "{frames} frames at {fps} fps, seam {wi}: carried audio token {i} starts at the wrong moment");
                    }
                }
            }
        }
    }
}

/// A multi-window clip's sound is the same LENGTH as a single-window clip of
/// the same request would be - the windows contribute tokens to one latent,
/// and those contributions sum to the clip's own count.
///
/// This is what keeps the container's two stream durations bit-identical to
/// what they already were: the decode that follows is
/// `(4 * total - 3) * HOP_LENGTH` samples whatever the plan was.
#[test]
fn a_multi_window_clip_carries_exactly_the_tokens_a_single_window_one_would() {
    for fps in RATES {
        for frames in FRAMES {
            let plan = plan_for(frames, fps);
            let a = audio::audio_plan(&plan, CONTEXT_LATENT_FRAMES, frames, fps).expect("the plan is aligned");
            assert_eq!(a.total, audio::latent_frames(frames, fps), "{frames} frames at {fps} fps: the plan's windows do not sum to the clip's own token count");
            assert_eq!(a.context, audio::context_tokens(CONTEXT_LATENT_FRAMES, fps));
            let summed: usize = (0..plan.len()).map(|i| a.new_tokens(i)).sum();
            assert_eq!(summed, a.total, "the per-window contributions do not sum to the total");
            // And the bound the single-window frame-count/fps sweep enforces
            // (`ltxv::audio`'s own `the_audio_track_is_the_same_length_as_the
            // _clip_it_belongs_to`), now on the clip a MULTI-window plan
            // actually produces: short of the picture by the causal trim plus
            // at most half a token, and never long, because the pad step only
            // ever extends.
            let video_seconds = frames as f64 / fps as f64;
            let audio_seconds = f64::from(4 * a.total as u32 - 3) * f64::from(audio::HOP_LENGTH) / f64::from(audio::SAMPLE_RATE);
            let slack = 3.0 * f64::from(audio::HOP_LENGTH) / f64::from(audio::SAMPLE_RATE) + 0.5 / f64::from(LATENT_RATE);
            assert!(video_seconds - audio_seconds <= slack + 1e-9, "{frames} frames at {fps} fps: video {video_seconds:.4}s vs audio {audio_seconds:.4}s over {} windows", plan.len());
            assert!(audio_seconds <= video_seconds + 1e-9, "{frames} frames at {fps} fps: audio {audio_seconds:.4}s is longer than video {video_seconds:.4}s");
            for (i, w) in plan.iter().enumerate() {
                // Each window's own sequence is exactly the audio a clip of
                // its decoded length carries - the window is a clip, and the
                // rule that sizes one sizes the other.
                assert_eq!(a.per_window[i], audio::latent_frames(w.decoded_frames(), fps), "window {i} of a {frames}-frame clip at {fps} fps");
                assert!(a.new_tokens(i) > 0, "window {i} would generate no sound at all");
            }
        }
    }
}

/// The refusal. A plan whose seams do not land on the audio token grid is
/// rejected with a specific message, not carried with a rounded context - and
/// the UNALIGNED planner produces exactly such a plan at a 24-frame-a-second
/// clip, which is why
/// the aligned one exists.
#[test]
fn a_plan_whose_seams_miss_the_token_grid_is_refused_rather_than_rounded() {
    let (frames, fps) = (241usize, 24usize);
    assert_eq!(audio::window_latent_frame_quantum(fps), 3, "at 24 frames a second a video latent frame is 25/3 audio tokens, so three of them are the smallest whole number");
    let naive = window_plan(frames, LH, LW, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS).expect("the silent plan is legal");
    let e = audio::audio_plan(&naive, CONTEXT_LATENT_FRAMES, frames, fps).expect_err("a plan that advances by 6 then 5 latent frames cannot carry whole audio tokens at 24 frames a second");
    assert!(e.contains("whole number"), "the refusal has to say what is wrong: {e}");
    assert!(e.contains("multiples of 3"), "the refusal has to say what would work: {e}");

    // And the aligned planner at the same request does not need refusing.
    let aligned = plan_for(frames, fps);
    audio::audio_plan(&aligned, CONTEXT_LATENT_FRAMES, frames, fps).expect("the aligned plan carries whole tokens");

    // A quantum of 1 constrains nothing, which is what a divisible frame rate
    // gets: at 25 frames a second a video latent frame is exactly 8 tokens.
    assert_eq!(audio::window_latent_frame_quantum(25), 1);
    audio::audio_plan(&window_plan(frames, LH, LW, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS).expect("legal"), CONTEXT_LATENT_FRAMES, frames, 25).expect("every seam is whole at 25 frames a second");
}

/// The quantum must cost a video-only request NOTHING: `align == 1` is the
/// plan this crate already made, window for window, at every shape the
/// existing long-form sweep covers.
#[test]
fn a_silent_clips_window_plan_is_exactly_what_it_always_was() {
    for frames in FRAMES.iter().chain(&[113, 1 + 8 * 200]) {
        for (lh, lw) in [(LH, LW), (11, 20), (2, 2)] {
            let want = window_plan(*frames, lh, lw, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS);
            let got = window_plan_aligned(*frames, lh, lw, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS, 1);
            assert_eq!(want, got, "{frames} frames on a {lh}x{lw} grid");
        }
    }
}

/// A window's carried audio prefix is the previous window's own last tokens,
/// bit for bit - the audio counterpart of `longform::carry_tail`'s promise,
/// and the reason nothing is re-encoded at a seam.
#[test]
fn the_carried_tail_is_the_previous_windows_own_last_tokens() {
    let dim = TOKEN_DIM as usize;
    let ta = 7usize;
    let latent: Vec<f32> = (0..ta * dim).map(|i| i as f32).collect();
    let tail = audio::carry_tail(&latent, 3);
    assert_eq!(tail.len(), 3 * dim);
    assert_eq!(tail, latent[(ta - 3) * dim..], "the tail has to be a slice of the source, not a resample of it");
    assert_eq!(audio::carry_tail(&latent, 0).len(), 0);
    assert_eq!(audio::carry_tail(&latent, ta), latent, "carrying everything is the whole latent");
}

/// Across EVERY frame rate a caller can type, and lengths that need two
/// windows and many, a plan either carries an exact token layout or is
/// refused - never a layout that is off by a token.
///
/// The narrow sweeps above prove the rule at the rates this model is actually
/// run at. This one is the backstop for the rest of the range, including the
/// rates where `round(frames / fps * LATENT_RATE)` can land near a tie: the
/// exactness argument is arithmetic on real numbers, and this asserts that
/// nothing in the floating-point realisation of it can slip through as a
/// wrong answer rather than as a refusal.
#[test]
fn no_frame_rate_produces_a_plan_that_is_wrong_rather_than_refused() {
    let (mut planned, mut refused) = (0usize, 0usize);
    for fps in 1..=120usize {
        let q = audio::window_latent_frame_quantum(fps);
        for k in [15usize, 16, 20, 30, 45, 60] {
            let frames = 1 + 8 * k;
            let Ok(plan) = window_plan_aligned(frames, LH, LW, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS, q) else {
                refused += 1;
                continue;
            };
            match audio::audio_plan(&plan, CONTEXT_LATENT_FRAMES, frames, fps) {
                Ok(a) => {
                    assert_eq!(a.total, audio::latent_frames(frames, fps), "{frames} frames at {fps} frames a second: an accepted plan must be exact");
                    for wi in 0..plan.len() - 1 {
                        assert_eq!(
                            (origin(&plan[wi + 1]) - origin(&plan[wi])) * LATENT_RATE as usize,
                            (a.per_window[wi] - a.context) * fps,
                            "{frames} frames at {fps} frames a second, seam {wi}"
                        );
                    }
                    planned += 1;
                }
                Err(_) => refused += 1,
            }
        }
    }
    // Most of this range is legitimately unplannable at this grid: a window
    // has 15 latent frames to spend, so a frame rate whose quantum exceeds
    // `15 - CONTEXT_LATENT_FRAMES` has nowhere to put a seam. What the two
    // bounds assert is that BOTH arms were exercised - a sweep that only
    // refused, or only planned, would prove nothing about the other one.
    assert!(planned > 80, "only {planned} of the sweep planned; the backstop has to exercise the planning arm too");
    assert!(refused > 80, "only {refused} of the sweep was refused; the backstop has to exercise the refusal arm too");
}

/// **The LAST window must stay unconstrained, and this is what it buys.**
///
/// Only a window with a successor hands anything across a seam, so only those
/// have to advance by a whole quantum. That is not a relaxation for
/// convenience - it is what makes the rule usable at all, and the 10 s target
/// shape is the proof. If every window were constrained then `head - context`
/// and every `new` would be multiples of the quantum, which forces the clip's
/// own `k` into the single residue class `context - 1`; 241 frames is `k =
/// 30`, the default context is 8, and `30 % 3 != 7 % 3`, so the length a
/// caller actually asked for would be refused outright.
///
/// This test fails the moment someone "tidies up" by constraining the last
/// window too: it pins both that the target shape plans, and that its final
/// window genuinely uses the freedom.
#[test]
fn the_last_window_is_free_and_that_is_what_makes_a_ten_second_clip_plannable() {
    let (frames, fps, context) = (241usize, 24usize, CONTEXT_LATENT_FRAMES);
    let q = audio::window_latent_frame_quantum(fps);
    let plan = plan_for(frames, fps);
    audio::audio_plan(&plan, context, frames, fps).expect("the target shape has to plan");

    // Every window WITH a successor advances by a whole quantum...
    for (wi, w) in plan.iter().enumerate().take(plan.len() - 1) {
        assert!(
            (w.latent_frames() - context).is_multiple_of(q),
            "window {wi} advances by {} latent frames, which is not a multiple of the {q}-frame quantum",
            w.latent_frames() - context
        );
    }
    // ...and the last one does NOT, which is precisely the freedom in use.
    let last = plan.last().expect("a plan has windows");
    assert!(!last.new.is_multiple_of(q), "the last window advances by {} latent frames, a multiple of {q} - this shape no longer exercises the freedom, so pick one that does", last.new);

    // And the congruence that constraining it would impose excludes this
    // clip, which is why the freedom is load-bearing rather than cosmetic.
    let k_total = (frames - 1) / 8;
    assert_ne!(k_total % q, (context - 1) % q, "constraining every window would happen to admit this length, so it is the wrong witness for the rule");
}

/// The quantum is what its name says at every frame rate a caller can ask
/// for: the smallest number of video latent frames worth a whole number of
/// audio tokens, never zero, never larger than `fps`.
#[test]
fn the_quantum_is_the_smallest_advance_worth_whole_tokens() {
    for fps in 1..=120usize {
        let q = audio::window_latent_frame_quantum(fps);
        assert!(q >= 1 && q <= fps, "{fps} fps: quantum {q}");
        assert!(audio::tokens_for_video_latent_frames(q, fps).is_some(), "{fps} fps: {q} latent frames is not a whole number of tokens");
        for n in 1..q {
            assert!(audio::tokens_for_video_latent_frames(n, fps).is_none(), "{fps} fps: {n} latent frames is already whole, so {q} is not the smallest");
        }
        // And the derivation it comes from, spelled out once: a video latent
        // frame is VAE_TEMPORAL_SCALE pixel frames of clip time.
        assert_eq!(audio::TOKENS_PER_VIDEO_LATENT_FRAME_NUM, VAE_TEMPORAL_SCALE * LATENT_RATE as usize);
    }
}
