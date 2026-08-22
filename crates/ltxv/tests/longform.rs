// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ltxv::longform` / `ltxv::pipeline::generate_long` - generating a clip
//! longer than one denoising window can hold by carrying REAL LATENT frames
//! across every window boundary, rather than re-encoding one decoded pixel
//! frame.
//!
//! Swedish Embedded AB implements rolling-window latent video diffusion for
//! its clients. If your team needs a generative video pipeline whose motion
//! survives a window boundary, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Three always-run claims, and they are deliberately the only three:
//!
//! 1. The WINDOW PLAN ([`a_clip_that_fits_one_window_carries_no_context`] and
//!    its siblings) - pure arithmetic, no weights, always runs. This is what
//!    stands between a long request and hours of wasted device time.
//! 2. The CARRY is a slice, not an approximation
//!    ([`the_carried_tail_is_the_previous_windows_own_last_latent_frames`]):
//!    what feeds window `n+1` is bit-identical to what window `n` produced.
//!    Its other half - that a frozen prefix survives the whole sampler
//!    trajectory unchanged - is gated inside `ltxv::pipeline` itself
//!    (`a_frozen_prefix_of_latent_frames_survives_the_whole_trajectory`),
//!    because that is where the sampler lives. Together the two close the
//!    chain "window `n`'s last K latent frames == window `n+1`'s first K".
//! 3. The WIRING end to end ([`real_weights`]) on the real VAE with a tiny
//!    random-weight DiT - so it says nothing about quality, the same
//!    disclaimer `ltxv::pipeline`'s own module doc carries for every
//!    tiny-config path.
//!
//! And one more that is `#[ignore]`d because it costs four real 22B
//! generations: [`seam_real`], which measures the thing the other three only
//! construct - that MOTION really is more continuous across a seam than the
//! last-frame chaining this path replaces.

use ltxv::longform::{carry_tail, window_plan, Window, CONTEXT_FRAMES, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS};

/// Reassemble a window plan the way `ltxv::pipeline::generate_long` does -
/// every window emits only the pixel frames its NEW latent frames cover - and
/// say which output frames came out, in order.
fn reassembled(plan: &[Window]) -> Vec<usize> {
    let mut out = Vec::new();
    for w in plan {
        out.extend(w.first_frame..w.first_frame + w.emitted_frames());
    }
    out
}

/// The reference's own prefix size for temporal extension
/// (`packages/ltx-trainer/configs/video_extend_lora.yaml`,
/// `temporal_boundary: 8`, whose validation samples spell the same number as
/// `num_frames: 57`). A constant that drifted from that would silently make
/// this port's continuation conditioning a different thing from the one the
/// checkpoint family was trained for.
#[test]
fn the_carried_context_is_the_references_own_eight_latent_frames() {
    assert_eq!(CONTEXT_LATENT_FRAMES, 8, "the reference's video-extension prefix is 8 latent frames");
    assert_eq!(CONTEXT_FRAMES, 57, "8 latent frames is (8 - 1) * 8 + 1 = 57 pixel frames, the number the reference's own validation samples carry");
}

/// A request that fits one window must produce exactly one window, carrying
/// NOTHING - so every shape this crate already generated keeps taking the
/// path it already took, with no context tokens spent and no seam anywhere.
#[test]
fn a_clip_that_fits_one_window_carries_no_context() {
    // 113 frames at 1280x704: 15 latent frames x 22 x 40 = 13200 tokens, the
    // largest single-window generation this crate has a recorded real run at.
    let plan = window_plan(113, 22, 40, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS).expect("13200 tokens is one window");
    assert_eq!(plan.len(), 1, "a request that fits must not be split: {plan:?}");
    assert_eq!(plan[0].context, 0, "a single window has nothing to carry");
    assert_eq!(plan[0].new, 15);
    assert_eq!(plan[0].emitted_frames(), 113);
    assert_eq!(reassembled(&plan), (0..113).collect::<Vec<_>>());
}

/// The one that matters: a request longer than one window becomes consecutive
/// windows that each fit the token ceiling, each of which is a legal `1 + 8k`
/// clip for the causal VAE, each carrying the previous window's last
/// `CONTEXT_LATENT_FRAMES` latent frames, and which reassemble to exactly the
/// requested frame count in order with no duplicated and no missing frame.
#[test]
fn a_request_longer_than_one_window_rolls_a_latent_context_across_every_seam() {
    let (frames, lh, lw) = (481usize, 22usize, 40usize); // 20 seconds at 24 fps, 1280x704
    let plan = window_plan(frames, lh, lw, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS).expect("a long request plans, it does not fail");
    assert!(plan.len() > 1, "481 frames cannot be one window, got {plan:?}");

    assert_eq!(plan[0].context, 0, "the first window has no predecessor to carry from");
    for (i, w) in plan.iter().enumerate() {
        assert!(w.new >= 1, "window {i} generates nothing: {w:?}");
        if i > 0 {
            assert_eq!(w.context, CONTEXT_LATENT_FRAMES, "window {i} must carry the full context, not a truncated one: {w:?}");
            assert!(plan[i - 1].latent_frames() >= w.context, "window {} has only {} latent frames and cannot supply window {i}'s {}-frame context", i - 1, plan[i - 1].latent_frames(), w.context);
        }
        assert!(w.latent_frames() * lh * lw <= LONGFORM_MAX_TOKENS, "window {i} is {} tokens, over the {LONGFORM_MAX_TOKENS} ceiling", w.latent_frames() * lh * lw);
        assert_eq!((w.decoded_frames() - 1) % 8, 0, "window {i} decodes {} frames, which is not 1 + 8k", w.decoded_frames());
    }
    assert_eq!(reassembled(&plan), (0..frames).collect::<Vec<_>>(), "the windows do not reassemble to the requested clip");
}

/// A request the causal VAE cannot represent, and a geometry where the
/// context alone already fills the token ceiling, are both refused before any
/// weight is read - the caller gets told, rather than getting an
/// out-of-memory abort an hour in.
#[test]
fn an_impossible_request_is_refused_up_front() {
    assert!(window_plan(24, 4, 4, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS).is_err(), "24 frames is not 1 + 8k");
    assert!(window_plan(0, 4, 4, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS).is_err(), "0 frames is not a clip");
    assert!(window_plan(9, 4, 4, 0, LONGFORM_MAX_TOKENS).is_err(), "a zero-frame context is the naive chaining this replaces, not a plan");
    // Eight context latent frames plus one new one is already over the
    // ceiling at this grid, so no split can rescue it.
    assert!(window_plan(1201, 64, 64, CONTEXT_LATENT_FRAMES, LONGFORM_MAX_TOKENS).is_err(), "the context alone does not fit this grid");
}

/// What crosses a seam is a SLICE of the previous window's own final latent -
/// the last `k` latent frames, channel-major, bit for bit. Nothing is
/// decoded, re-encoded, interpolated or rescaled on the way.
#[test]
fn the_carried_tail_is_the_previous_windows_own_last_latent_frames() {
    let (c, lat_t, lh, lw, k) = (3usize, 5usize, 2usize, 2usize, 2usize);
    // A latent whose every value is distinguishable, so a transposed or
    // off-by-one slice cannot pass.
    let latent: Vec<f32> = (0..c * lat_t * lh * lw).map(|i| i as f32).collect();

    let tail = carry_tail(&latent, c, lat_t, lh, lw, k);

    assert_eq!(tail.len(), c * k * lh * lw);
    let plane = lh * lw;
    for ci in 0..c {
        for f in 0..k {
            for s in 0..plane {
                let want = latent[(ci * lat_t + (lat_t - k + f)) * plane + s];
                let got = tail[(ci * k + f) * plane + s];
                assert_eq!(got, want, "channel {ci} carried frame {f} cell {s} is not the source latent's own value");
            }
        }
    }
}

mod real_weights {
    use std::path::Path;

    use ltxv::longform::{window_plan, CONTEXT_LATENT_FRAMES};
    use ltxv::pipeline::{generate_long, GenOpts, LongOpts, Paths};

    /// The named environment variable, else the repo-relative
    /// `resources/ltxv/weights/` the real files ship under - never a literal
    /// machine path, the same convention `upscale.rs`/`vae_parity.rs` use.
    fn weights_path(env: &str, rel: &str) -> Option<String> {
        if let Ok(p) = std::env::var(env) {
            return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
        }
        let p = format!("{}/../../resources/ltxv/weights/{rel}", env!("CARGO_MANIFEST_DIR"));
        Path::new(&p).exists().then_some(p)
    }

    /// End to end on the real VAE: a request that needs several windows comes
    /// back as ONE clip of exactly the requested length, at the requested
    /// size, with every window's frames really decoded.
    ///
    /// The DiT is the tiny random-weight config and the token ceiling is
    /// forced down to a value a 64x64 clip actually crosses, so this is a
    /// WIRING claim - that the plan, the latent carry, the per-window
    /// denoise and the reassembly compose - and not a quality one. CPU
    /// device deliberately: what this gates is device-independent, and the
    /// box's cards are not this test's to reserve.
    #[test]
    fn a_multi_window_request_comes_back_as_one_clip_of_the_requested_length() {
        let Some(vae) = weights_path("BRAIN_LTXV_VAE", "vae/ltx-2.5-video-vae-conv-bf16.safetensors") else {
            return brain_testutil::skip("set BRAIN_LTXV_VAE to the real VAE checkpoint");
        };
        let paths = Paths::resolve(Some(&vae), None, None, None).expect("the configured path resolves");
        // 64x64 is a 2x2 latent grid, so one latent frame is 4 tokens: the
        // ceiling has to come down for a smoke-sized clip to need splitting.
        let (frames, context, max_tokens) = (41usize, 2usize, 20usize);
        let o = LongOpts {
            context_latent_frames: context,
            max_window_tokens: max_tokens,
            base: GenOpts { frames, width: 64, height: 64, steps: 2, fps: 8, device: Some("cpu".into()), seed: 5, ..GenOpts::default() },
        };
        let plan = window_plan(frames, 2, 2, context, max_tokens).expect("the plan is legal");
        assert!(plan.len() > 1, "this shape has to need several windows for the test to mean anything: {plan:?}");

        let mut phases: Vec<String> = Vec::new();
        let (video, timings) = generate_long(&paths, "a moving bar", &o, &capability::CancelToken::default(), |_, _, phase| phases.push(phase.to_string())).expect("generate_long");

        assert_eq!(video.frames.len(), frames, "the clip is not the requested length");
        assert_eq!((video.width, video.height), (64, 64));
        assert!(video.frames.iter().all(|f: &Vec<u8>| f.len() == 64 * 64 * 3), "a frame is the wrong size");
        assert!(video.frames.iter().any(|f: &Vec<u8>| f.iter().any(|&v| v != f[0])), "every frame is a flat colour - nothing was decoded");
        assert_eq!(timings.steps, o.base.steps * plan.len(), "not every window denoised");
        assert!(phases.iter().filter(|p| p.as_str() == "vae decode").count() == plan.len(), "one decode per window: {phases:?}");
    }

    /// The default context is the one this crate ships, and a default-shaped
    /// long request really does plan several windows against it - a guard
    /// against a future ceiling change silently turning long-form generation
    /// back into a single oversized window that cannot run.
    #[test]
    fn a_twenty_second_request_at_720p_plans_several_windows_at_the_default_context() {
        let plan = window_plan(481, 22, 40, CONTEXT_LATENT_FRAMES, ltxv::longform::LONGFORM_MAX_TOKENS).expect("plans");
        assert!(plan.len() >= 4, "20 seconds at 1280x704 should be several windows, got {}", plan.len());
    }
}

/// **The seam has to hold the motion.** The one claim in this file that
/// cannot be made by construction: that a rolling latent context really does
/// produce a more continuous seam than chaining on a re-encoded last frame.
///
/// Everything else here proves the carry is exact and the plan is legal. A
/// bit-identical latent prefix is necessary for continuous motion and does
/// not by itself demonstrate it - the model still has to USE the history it
/// is given. That is a real-weight question and this is the measurement.
///
/// ## The metric
///
/// `clipmetric::frame_to_frame_diffs` on the assembled clip, and the entry at
/// the seam divided by the clip's own median. A clip with steady motion sits
/// near 1 whatever the motion's speed (that is exactly what the metric was
/// built for in Phase 19); a boundary where the motion changes, stalls or
/// reverses is a spike at one known index. The claim is comparative - the
/// rolling-context clip's seam ratio must be BELOW the naively chained one's
/// at the same shape, seed and prompt - because no absolute bound has been
/// calibrated for this shape and inventing one would be a number nothing
/// measured.
///
/// ## Cost
///
/// Four real 22B generations (two per arm, ~6 minutes each on a Tesla P40,
/// weight-streaming bound). Run explicitly:
///
/// ```text
/// BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
/// BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
/// BRAIN_LTXV_TEXT_ENCODER=<...gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
/// cargo test --release -p brain-ltxv --test longform -- --ignored --nocapture
/// ```
mod seam_real {
    use ltxv::clipmetric::frame_to_frame_diffs;
    use ltxv::longform::{window_plan, CONTEXT_LATENT_FRAMES};
    use ltxv::pipeline::{generate, generate_long, GenOpts, LongOpts, Paths, Video};

    /// 384x192 is 72 tokens per latent frame, the shape `motion_real.rs` and
    /// `anchor_real.rs` both calibrate at. The window ceiling is forced to 14
    /// latent frames so a 121-frame request splits into exactly two windows -
    /// with the REAL default context of 8 latent frames, not a shrunken one -
    /// at a size four real generations can finish in.
    const FRAMES: usize = 121;
    const WIDTH: usize = 384;
    const HEIGHT: usize = 192;
    const FPS: usize = 24;
    const SEED: u64 = 42;
    const MAX_WINDOW_TOKENS: usize = 14 * (HEIGHT / 32) * (WIDTH / 32);
    const PROMPT: &str = "a Belgian Malinois running left to right across an open field at constant speed, camera tracking alongside it, motion-blurred background";

    fn median(mut v: Vec<f32>) -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    }

    /// `Video::frames` (per-frame interleaved RGB8) to the decoder's own
    /// `[3, frames, h, w]` plane-major `[-1, 1]` layout, which is what
    /// `frame_to_frame_diffs` takes - the same conversion
    /// `clip_stability_real.rs` does, so both files measure one observable.
    fn to_chw(v: &Video) -> Vec<f32> {
        let (h, w, n) = (v.height as usize, v.width as usize, v.frames.len());
        let mut chw = vec![0f32; 3 * n * h * w];
        for (f, frame) in v.frames.iter().enumerate() {
            for c in 0..3 {
                let base = (c * n + f) * h * w;
                for i in 0..h * w {
                    chw[base + i] = frame[i * 3 + c] as f32 / 127.5 - 1.0;
                }
            }
        }
        chw
    }

    /// `diffs[seam - 1]` is the difference between the last frame the earlier
    /// window contributed and the first frame the later one did.
    fn seam_ratio(v: &Video, seam: usize) -> (f32, f32, f32) {
        let diffs = frame_to_frame_diffs(&to_chw(v), v.frames.len(), v.height as usize, v.width as usize);
        let med = median(diffs.clone());
        let at = diffs[seam - 1];
        (at / med.max(1e-6), at, med)
    }

    fn real_paths() -> Option<Paths> {
        let p = Paths::resolve(None, None, None, None).ok()?;
        p.dit.as_ref()?;
        Some(p)
    }

    fn base_opts(frames: usize) -> GenOpts {
        GenOpts { frames, width: WIDTH, height: HEIGHT, fps: FPS, seed: SEED, guidance: 1.0, dit_config: "ltx25_22b".into(), ..GenOpts::default() }
    }

    #[test]
    #[ignore = "four real 22B generations, ~25 minutes on a Tesla P40"]
    fn a_rolling_latent_context_holds_the_motion_better_than_chaining_on_a_re_encoded_frame() {
        let Some(paths) = real_paths() else {
            return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE to the real LTX-2.5 checkpoints");
        };
        let (lh, lw) = (HEIGHT / 32, WIDTH / 32);
        let plan = window_plan(FRAMES, lh, lw, CONTEXT_LATENT_FRAMES, MAX_WINDOW_TOKENS).expect("the plan is legal");
        assert_eq!(plan.len(), 2, "this gate compares ONE seam; got {plan:?}");
        let seam = plan[1].first_frame;
        let cancel = capability::CancelToken::default();

        // Arm A: the rolling latent context.
        let o = LongOpts { context_latent_frames: CONTEXT_LATENT_FRAMES, max_window_tokens: MAX_WINDOW_TOKENS, base: base_opts(FRAMES) };
        let (rolling, _) = generate_long(&paths, PROMPT, &o, &cancel, |_, _, _| {}).unwrap_or_else(|e| panic!("generate_long failed: {e}"));
        assert_eq!(rolling.frames.len(), FRAMES);

        // Arm B: the naive chain this path replaces - generate window 0, write
        // its last decoded frame out, and condition window 1 on that ONE
        // picture. Same seed, same prompt, same window lengths, so the seam is
        // the only thing that differs.
        let (head, _) = generate(&paths, PROMPT, &base_opts(plan[0].decoded_frames()), &cancel, |_, _, _| {}).unwrap_or_else(|e| panic!("head generation failed: {e}"));
        let anchor = std::env::temp_dir().join(format!("brain-ltxv-longform-seam-{}.png", std::process::id()));
        let last = head.frames.last().expect("the head clip has frames").clone();
        image::RgbImage::from_raw(WIDTH as u32, HEIGHT as u32, last).expect("the frame is WIDTH*HEIGHT*3 bytes").save(&anchor).expect("write the anchor still");
        let tail_frames = 1 + 8 * plan[1].new;
        let tail_opts = GenOpts { start_frame: Some(anchor.to_string_lossy().into_owned()), ..base_opts(tail_frames) };
        let (tail, _) = generate(&paths, PROMPT, &tail_opts, &cancel, |_, _, _| {}).unwrap_or_else(|e| panic!("tail generation failed: {e}"));
        let _ = std::fs::remove_file(&anchor);
        // The tail's own frame 0 IS the anchor, which the head already
        // emitted - exactly the duplicate the hand-run chaining drops.
        let mut chained = head.clone();
        chained.frames.extend(tail.frames.into_iter().skip(1));
        assert_eq!(chained.frames.len(), FRAMES, "the chained arm has to be the same length as the rolling one");

        let (r_ratio, r_at, r_med) = seam_ratio(&rolling, seam);
        let (c_ratio, c_at, c_med) = seam_ratio(&chained, seam);
        eprintln!("seam at frame {seam}: rolling context {r_ratio:.2} (diff {r_at:.2}, median {r_med:.2}) vs last-frame chain {c_ratio:.2} (diff {c_at:.2}, median {c_med:.2})");

        assert!(r_med > 0.5 && c_med > 0.5, "one of the clips barely moves (medians {r_med:.2} / {c_med:.2}), so a seam ratio says nothing about motion continuity");
        // 1.0 is the target, not 0: it means the seam transitions exactly like
        // a typical frame in the clip. Below 1.0 is a freeze (the seam moves
        // LESS than usual, the "still octopus" artifact naive chaining makes),
        // above 1.0 is a jump - so distance from 1.0 is the defect, not
        // magnitude, and a naive `r_ratio < c_ratio` would falsely reward a
        // chain arm that stalls hard enough to undercut a barely-imperfect one.
        let (r_dist, c_dist) = ((r_ratio - 1.0).abs(), (c_ratio - 1.0).abs());
        assert!(
            r_dist < c_dist,
            "the rolling latent context did not land closer to a natural (ratio 1.0) seam: {r_ratio:.2} (|{r_dist:.2}|) against the last-frame chain's {c_ratio:.2} (|{c_dist:.2}|) at frame {seam}"
        );
    }
}
