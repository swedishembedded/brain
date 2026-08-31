// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **A mid-frame anchor must be realised at its instant across long-form
//! windows.** `anchor_with_mid_real.rs` gates the single-generation case.
//! Long-form is different in the one way that matters here: the clip is
//! refined across several full-resolution windows, and an interior anchor's
//! instant belongs to exactly one of them. If the routing loses the mid
//! still - drops it, or conditions it at the wrong local instant - the
//! trajectory interpolates start to end and glides past the middle without
//! ever showing it, which is what real usage reports.
//!
//! The claim measured here is positional, not cosmetic: the frame of the
//! generated clip that lies CLOSEST to the mid still must be the frame at
//! the mid instant (within the VAE's own frame-grouping slack). The stills
//! are decoded frames of an unconditioned reference run at the same shape -
//! in distribution, at the exact target resolution, the same reasoning
//! `anchor_with_mid_real.rs` uses - so an ignored mid leaves the closest
//! approach wherever the start-to-end wander happens to dip, not at the
//! named instant.
//!
//! Swedish Embedded AB implements long-form video diffusion pipelines and
//! their measurement gates for its clients. If your team needs expertise in
//! video generation then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! `#[ignore]`d: three full real 22B generations. Run explicitly:
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! BRAIN_LTXV_TEXT_ENCODER=<...gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
//! BRAIN_LTXV_UPSAMPLER_SPATIAL=<...latent-spatial-upscaler-x2-bf16-1.0.safetensors> \
//! cargo test -p brain-ltxv --test longform_mid_real -- --ignored --nocapture
//! ```

use ltxv::pipeline::{generate, generate_long, GenOpts, LongOpts, Paths};

const FRAMES: usize = 41;
const MID: usize = 20;
const WIDTH: usize = 512;
const HEIGHT: usize = 256;
const SEED: u64 = 42;
const FPS: usize = 8;

const PROMPT: &str = "a belgian malinois dog running at full speed along a mountain road, a small winged P40 graphics card flying beside it, camera tracking";

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64
}

fn real_paths() -> Option<Paths> {
    let p = Paths::resolve(None, None, None, None).ok()?;
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

/// The shape has to be a genuine stage-major one: the stage-2 ceiling forced
/// low enough that the full-resolution plan splits (41 px frames = 6 latent,
/// 128 tokens/latent frame at 512x256, ceiling 640 = 5 latent/window: a head
/// window emitting px [0, 33) and a continuation emitting [33, 41)), while
/// both the window-major dispatch plan (512 tokens = 4 latent/window at full
/// res) and the half-resolution stage-1 plan (32 tokens/latent frame) stay
/// coarse - stage 1 remains one global window that sees all three anchors
/// and decides the motion, the detail is refined across windows, which is
/// the shape real 720p runs take.
fn long_opts(base: GenOpts) -> LongOpts {
    LongOpts { context_latent_frames: 2, max_window_tokens: 512, max_refine_tokens: 640, base }
}

#[test]
#[ignore = "three full real 22B generations"]
fn the_mid_anchor_is_realised_at_its_instant_in_a_stage_major_run() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE + BRAIN_LTXV_UPSAMPLER_SPATIAL to the real LTX-2.5 checkpoints");
    };
    let cancel = capability::CancelToken::default();

    // The reference run is the source of all three stills, so an ignored mid
    // cannot be excused as an out-of-distribution anchor.
    let (reference, _) = generate(&paths, PROMPT, &GenOpts { frames: FRAMES, width: WIDTH, height: HEIGHT, seed: SEED, fps: FPS, eta: 1.0, guidance: 1.0, dit_config: "ltx25_22b".into(), device: Some("gpu".into()), ..GenOpts::default() }, &cancel, |_, _, _| {}).expect("reference run");
    let dir = std::env::temp_dir().join(format!("ltxv-longform-mid-{}", std::process::id()));
    let start_path = write_still(&dir, "start.png", &reference.frames[0]);
    let mid_path = write_still(&dir, "mid.png", &reference.frames[MID]);
    let end_path = write_still(&dir, "end.png", &reference.frames[FRAMES - 1]);

    let base = GenOpts {
        frames: FRAMES,
        width: WIDTH,
        height: HEIGHT,
        seed: SEED,
        fps: FPS,
        eta: 1.0,
        guidance: 1.0,
        dit_config: "ltx25_22b".into(),
        device: Some("gpu".into()),
        start_frame: Some(start_path),
        mid_frame: Some(mid_path),
        end_frame: Some(end_path),
        ..GenOpts::default()
    };
    let (video, _) = generate_long(&paths, PROMPT, &long_opts(base), &cancel, |_, _, _| {}).expect("stage-major run with all three anchors");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(video.frames.len(), FRAMES);
    assert_eq!((video.width, video.height), (WIDTH as u32, HEIGHT as u32));

    let to_mid: Vec<f64> = video.frames.iter().map(|f| mean_abs_delta(f, &reference.frames[MID])).collect();
    let (best_i, best_d) = to_mid.iter().enumerate().fold((0usize, f64::INFINITY), |(bi, bd), (i, &d)| if d < bd { (i, d) } else { (bi, bd) });
    let d_start = mean_abs_delta(&video.frames[0], &reference.frames[0]);
    let d_mid = to_mid[MID];
    let d_end = mean_abs_delta(&video.frames[FRAMES - 1], &reference.frames[FRAMES - 1]);

    println!("anchor deltas: start {d_start:.2}, mid(instant) {d_mid:.2}, end {d_end:.2}");
    println!("closest approach to the mid still: frame {best_i} at {best_d:.2} (mid instant is frame {MID})");
    for (i, d) in to_mid.iter().enumerate().step_by(2) {
        println!("  frame {i:3}: {d:.2}");
    }

    assert!(
        (18..=22).contains(&best_i),
        "the clip's closest approach to the mid still is frame {best_i}, not the mid instant {MID}: the mid anchor is not being realised where it is conditioned (start {d_start:.2}, mid {d_mid:.2}, end {d_end:.2})"
    );
}
