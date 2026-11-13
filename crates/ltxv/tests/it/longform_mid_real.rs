// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **A keyframe must be REALISED at its instant across long-form windows,
//! and the clip must move through it smoothly.** The clip is refined across
//! several full-resolution windows and an interior keyframe's instant
//! belongs to exactly one of them. If the routing loses the still - drops
//! it, or conditions it at the wrong local instant - the trajectory glides
//! from start to end without ever showing it. If the mechanism holds the
//! still instead of guiding toward it, the clip freezes on it and jumps into
//! and out of the freeze. Both are gated.
//!
//! All stills come from an unconditioned reference run at the same shape
//! (in distribution, at the exact target resolution, the same reasoning
//! `anchor_with_mid_real.rs` uses). Four claims:
//!
//! 1. **Position**: the generated frame closest to the keyframe is the
//!    frame at its instant (within the VAE's own frame-grouping slack).
//! 2. **Depth**: the mean |delta| AT the instant is close to the still -
//!    closer than the unconditioned reference's own neighbouring frames sit
//!    from it, so the clip reaches the still rather than passing near it.
//! 3. **The end still too**: the last frame reaches it. A refinement window
//!    handed the end as a guide past its own span pulled its own last frames
//!    onto it instead, so the clip reached the end early and jumped away.
//! 4. **Smoothness**: no frame-to-frame step anywhere in the clip stands out
//!    from the steps beside it - no freeze, no jump. A still pinned into its
//!    latent slot measured a several-times step into an eight-frame freeze
//!    at 768x448; the unconditioned clip's worst is barely above its
//!    neighbours.
//!
//! Swedish Embedded AB implements long-form video diffusion pipelines and
//! their measurement gates for its clients. If your team needs expertise in
//! video generation then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! `#[ignore]`d: two full real 22B generations. Run explicitly:
//!
//! ```text
//! cargo test -p brain-ltxv --test it longform_mid_real -- --ignored --nocapture
//! ```

use ltxv::pipeline::{generate, generate_long, GenOpts, LongOpts, Paths};

const FRAMES: usize = 121;
const MID: usize = 60;
const WIDTH: usize = 384;
const HEIGHT: usize = 192;
const SEED: u64 = 42; // seed: the keyframe realisation margins were measured on the real 22B checkpoint at this seed (ignored, needs weights)
const FPS: usize = 24;

const PROMPT: &str = "a belgian malinois dog running at full speed along a mountain road, a small winged P40 graphics card flying beside it, camera tracking";

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64
}

fn real_paths() -> Option<Paths> {
    let p = ltxv::spec::default_store_paths()?;
    p.dit.as_ref()?;
    p.spatial_upsampler.as_ref()?;
    Some(p)
}

fn write_still(dir: &std::path::Path, name: &str, frame: &[u8]) -> String {
    std::fs::create_dir_all(dir).expect("temp dir");
    let path = dir.join(name);
    image::RgbImage::from_raw(WIDTH as u32, HEIGHT as u32, frame.to_vec()).expect("decoded frame as an RGB image").save(&path).expect("write the conditioning still");
    path.to_string_lossy().into_owned()
}

/// Three windows of the default loop at the reference's own 8-latent-frame
/// context: 16 latent frames at 72 tokens each under a 1008-token ceiling
/// (14 latent frames) split evenly into 8 new, then 8 carried + 4 new twice,
/// emitting pixel frames [0, 57), [57, 89) and [89, 121). The keyframe's
/// instant is the second window's, the end still the third's.
fn long_opts(base: GenOpts) -> LongOpts {
    LongOpts { max_window_tokens: 14 * (HEIGHT / 32) * (WIDTH / 32), base, ..LongOpts::default() }
}

#[test]
#[ignore = "two full real 22B generations"]
fn a_keyframe_is_reached_at_its_instant_and_passed_through_smoothly_across_windows() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("put the real LTX-2.5 checkpoints in the model store");
    };
    let cancel = capability::CancelToken::default();

    // The reference run is the source of all three stills, so an ignored
    // keyframe cannot be excused as an out-of-distribution anchor.
    let (reference, _) = generate(&paths, PROMPT, &GenOpts { frames: FRAMES, width: WIDTH, height: HEIGHT, seed: SEED, fps: FPS, eta: 1.0, guidance: ltxv::guidance::Guidance::cfg(1.0), dit_config: "ltx25_22b".into(), device: Some("gpu".into()), ..GenOpts::default() }, &cancel, |_, _, _| {}).expect("reference run"); // seed: the keyframe realisation margins were measured on the real 22B checkpoint at this seed (ignored, needs weights)
    let dir = std::env::temp_dir().join(format!("ltxv-longform-mid-{}", std::process::id()));
    let start_path = write_still(&dir, "start.png", &reference.frames[0]);
    let mid_path = write_still(&dir, "mid.png", &reference.frames[MID]);
    let end_path = write_still(&dir, "end.png", &reference.frames[FRAMES - 1]);

    let base = GenOpts {
        frames: FRAMES,
        width: WIDTH,
        height: HEIGHT,
        // Another draw than the reference's: on the same noise the clip would
        // reach the stills by reproducing the reference, not by being guided.
        seed: SEED + 1, // seed: the keyframe realisation margins were measured on the real 22B checkpoint at this seed (ignored, needs weights)
        fps: FPS,
        eta: 1.0,
        guidance: ltxv::guidance::Guidance::cfg(1.0),
        dit_config: "ltx25_22b".into(),
        device: Some("gpu".into()),
        start_frame: Some(start_path),
        keyframes: vec![(MID, mid_path)],
        end_frame: Some(end_path),
        ..GenOpts::default()
    };
    let (video, _) = generate_long(&paths, PROMPT, &long_opts(base), &cancel, |_, _, _| {}).expect("a three-window run with all three anchors");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(video.frames.len(), FRAMES);
    assert_eq!((video.width, video.height), (WIDTH as u32, HEIGHT as u32));

    let to_mid: Vec<f64> = video.frames.iter().map(|f| mean_abs_delta(f, &reference.frames[MID])).collect();
    let (best_i, best_d) = to_mid.iter().enumerate().fold((0usize, f64::INFINITY), |(bi, bd), (i, &d)| if d < bd { (i, d) } else { (bi, bd) });
    let d_start = mean_abs_delta(&video.frames[0], &reference.frames[0]);
    let d_mid = to_mid[MID];
    let d_end = mean_abs_delta(&video.frames[FRAMES - 1], &reference.frames[FRAMES - 1]);
    // How far the moving scene's own frames around the instant sit from the
    // still: the floor "reaching the still" has to beat.
    let ref_nearest = (MID - 4..=MID + 4).filter(|&i| i != MID).map(|i| mean_abs_delta(&reference.frames[i], &reference.frames[MID])).fold(f64::INFINITY, f64::min);
    let ref_nearest_end = mean_abs_delta(&reference.frames[FRAMES - 2], &reference.frames[FRAMES - 1]);
    // The worst frame-to-frame step anywhere in the clip over the median of
    // the steps beside it - the keyframe's own instant and every window seam.
    let steps: Vec<f64> = video.frames.windows(2).map(|w| mean_abs_delta(&w[0], &w[1])).collect();
    let worst = jump_ratio(&steps, 0..steps.len());
    let ref_steps: Vec<f64> = reference.frames.windows(2).map(|w| mean_abs_delta(&w[0], &w[1])).collect();
    let ref_worst = jump_ratio(&ref_steps, 0..ref_steps.len());

    println!("anchor deltas: start {d_start:.2}, keyframe(instant) {d_mid:.2}, end {d_end:.2}");
    println!("closest approach to the keyframe: frame {best_i} at {best_d:.2} (instant is frame {MID}); the reference's own nearest neighbour sits {ref_nearest:.2} from it");
    println!("the end still: {d_end:.2} at the last frame; the reference's own previous frame sits {ref_nearest_end:.2} from it");
    println!("worst step in the clip: {worst:.2}x its neighbours (reference {ref_worst:.2}x)");
    for (i, d) in to_mid.iter().enumerate().step_by(2) {
        println!("  frame {i:3}: {d:.2} from the keyframe, {:.2} from the end still", mean_abs_delta(&video.frames[i], &reference.frames[FRAMES - 1]));
    }

    assert!(
        (MID - 2..=MID + 2).contains(&best_i),
        "the clip's closest approach to the keyframe is frame {best_i}, not its instant {MID}: the keyframe is not realised where it is conditioned (start {d_start:.2}, keyframe {d_mid:.2}, end {d_end:.2})"
    );
    assert!(d_mid < ref_nearest, "the keyframe is only approached (delta {d_mid:.2} at its instant), not reached: the reference's own neighbouring frames sit {ref_nearest:.2} from it");
    assert!(d_end < ref_nearest_end, "the end still is not reached at the last frame (delta {d_end:.2}; the reference's own previous frame sits {ref_nearest_end:.2} from it): a window reached it early, or not at all");
    assert!(worst <= MAX_JUMP, "a step in the clip is {worst:.2}x its neighbours (reference {ref_worst:.2}x): the clip freezes on, jumps to or jumps away from an anchor instead of moving through it");
}

/// Above this, one frame-to-frame step reads as a jump against the steps
/// around it.
const MAX_JUMP: f64 = 1.6;

/// The largest step in `range` over the median of the six steps on either
/// side of it.
fn jump_ratio(steps: &[f64], range: std::ops::Range<usize>) -> f64 {
    range
        .map(|i| {
            let mut ctx: Vec<f64> = steps[i.saturating_sub(6)..i].iter().chain(&steps[i + 1..(i + 7).min(steps.len())]).copied().collect();
            ctx.sort_by(f64::total_cmp);
            steps[i] / ctx[ctx.len() / 2].max(0.25)
        })
        .fold(0.0, f64::max)
}
