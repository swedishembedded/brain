// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Isolates the exact combination real usage reports as broken: a
//! mid-frame anchor AND the two-stage refinement path, together.**
//!
//! Two prior gates each pass alone:
//! `anchor_two_stage_real.rs` forces two-stage with `--start-frame` only
//! (no mid) and the anchor survives. `anchor_with_mid_real.rs` runs
//! `--start-frame` + `--mid-frame` + `--end-frame` at a shape small enough
//! to stay single-stage, and the anchor survives there too. Real usage
//! (1280x704, naturally two-stage) reports that `--start-frame` +
//! `--end-frame` alone interpolates correctly, but adding `--mid-frame`
//! loses the anchoring entirely. This test is the one combination neither
//! prior gate covers: three anchors AND two-stage, together, at the same
//! small/fast shape `anchor_two_stage_real.rs` uses.
//!
//! Swedish Embedded AB implements diffusion-sampler conditioning and its
//! numerical gates for its clients. If your team needs expertise in video
//! diffusion pipelines then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! `#[ignore]`d: two full real 22B generations plus one spatial-upscale
//! refinement pass, forced two-stage. Run explicitly:
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! BRAIN_LTXV_TEXT_ENCODER=<...gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
//! BRAIN_LTXV_UPSAMPLER=<...latent-spatial-upscaler-x2-bf16-1.0.safetensors> \
//! cargo test -p brain-ltxv --test anchor_mid_two_stage_real -- --ignored --nocapture
//! ```

use ltxv::pipeline::{generate, GenOpts, Paths};

const FRAMES: usize = 17;
const WIDTH: usize = 384;
const HEIGHT: usize = 192;
const SEED: u64 = 42;
const FPS: usize = 8;

const PROMPT: &str = "a belgian malinois dog running at full speed along a mountain road, a small winged P40 graphics card flying beside it, camera tracking";

fn mean_saturation(rgb: &[u8]) -> f64 {
    let n = rgb.len() / 3;
    let mut acc = 0.0f64;
    for p in rgb.chunks_exact(3) {
        let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
        let hi = r.max(g).max(b);
        let lo = r.min(g).min(b);
        if hi > 0.0 {
            acc += (hi - lo) / hi;
        }
    }
    acc / n as f64
}

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64
}

fn real_paths() -> Option<Paths> {
    let p = Paths::resolve(None, None, None, None).ok()?;
    p.dit.as_ref()?;
    Some(p)
}

fn write_still(dir: &std::path::Path, name: &str, frame: &[u8]) -> String {
    std::fs::create_dir_all(dir).expect("temp dir");
    let path = dir.join(name);
    image::RgbImage::from_raw(WIDTH as u32, HEIGHT as u32, frame.to_vec()).expect("decoded frame as an RGB image").save(&path).expect("write the conditioning still");
    path.to_string_lossy().into_owned()
}

fn base_opts() -> GenOpts {
    GenOpts { frames: FRAMES, width: WIDTH, height: HEIGHT, seed: SEED, fps: FPS, eta: 1.0, guidance: 1.0, dit_config: "ltx25_22b".into(), device: Some("gpu".into()), ..GenOpts::default() }
}

const MAX_SATURATION_RATIO: f64 = 1.08;
const MAX_FRAME0_DELTA: f64 = 7.0;

#[test]
#[ignore = "two full real 22B generations + one spatial upscale, forced two-stage"]
fn a_start_frame_anchor_survives_a_mid_frame_anchor_under_two_stage_refinement() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE to the real LTX-2.5 checkpoints");
    };
    if paths.spatial_upsampler.is_none() {
        return brain_testutil::skip("set BRAIN_LTXV_UPSAMPLER to the real spatial x2 latent upscaler - the two-stage path needs it");
    }
    // SAFETY (test-only): this is the only test in this binary that reads or
    // writes this variable.
    unsafe { std::env::set_var("BRAIN_LTXV_TWO_STAGE", "1") };
    let cancel = capability::CancelToken::default();

    let (seed_video, _) = generate(&paths, PROMPT, &base_opts(), &cancel, |_, _, _| {}).expect("unconditioned run (source of the anchors)");
    let start_anchor = seed_video.frames[0].clone();
    let end_anchor = seed_video.frames.last().expect("non-empty").clone();
    let mid_anchor = seed_video.frames[FRAMES / 2].clone();

    let dir = std::env::temp_dir().join(format!("ltxv-anchor-mid-2stage-gate-{}", std::process::id()));
    let start_path = write_still(&dir, "start.png", &start_anchor);
    let mid_path = write_still(&dir, "mid.png", &mid_anchor);
    let end_path = write_still(&dir, "end.png", &end_anchor);

    let o = GenOpts { start_frame: Some(start_path), mid_frame: Some(mid_path), end_frame: Some(end_path), ..base_opts() };
    let (video, timings) = generate(&paths, PROMPT, &o, &cancel, |_, _, _| {}).expect("keyframe-interpolation run with start+mid+end, forced two-stage");
    let _ = std::fs::remove_dir_all(&dir);
    unsafe { std::env::remove_var("BRAIN_LTXV_TWO_STAGE") };
    assert_eq!(video.frames.len(), FRAMES);

    let start_sat = mean_saturation(&start_anchor);
    let curve: Vec<f64> = video.frames.iter().map(|f| mean_saturation(f)).collect();
    let delta = mean_abs_delta(&video.frames[0], &start_anchor);
    let ratio = curve[0] / start_sat;
    eprintln!("start+mid+end (forced two-stage): {:.1}s, start-still saturation {start_sat:.4}, frame-0 saturation {:.4} (ratio {ratio:.3}), frame-0 delta {delta:.2}", timings.total(), curve[0]);
    eprintln!("start+mid+end (forced two-stage) saturation curve: {}", curve.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>().join(" "));

    assert!(
        (1.0 / MAX_SATURATION_RATIO..=MAX_SATURATION_RATIO).contains(&ratio),
        "frame 0 does not reproduce the start-frame still with a mid-frame anchor present AND two-stage refinement forced on: mean saturation {:.4} against the still's {start_sat:.4} (ratio {ratio:.3}, bound {MAX_SATURATION_RATIO:.2}). \
         anchor_with_mid_real.rs (same 3 anchors, single-stage) passes; anchor_two_stage_real.rs (two-stage, single anchor) passes. If this fails, the defect needs BOTH a mid-frame anchor AND the two-stage refinement path.",
        curve[0]
    );
    assert!(
        delta <= MAX_FRAME0_DELTA,
        "frame 0 does not reproduce the start-frame still with a mid-frame anchor present AND two-stage refinement forced on: mean absolute pixel delta {delta:.2} (bound {MAX_FRAME0_DELTA:.2})."
    );
}
