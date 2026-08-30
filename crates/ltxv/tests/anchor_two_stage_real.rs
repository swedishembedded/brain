// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The anchor has to survive stage 2, not just stage 1.** `anchor_real.rs`
//! proves `--start-frame` reproduces the clip's first decoded frame on the
//! reference's ordinary, single-stage path. This is the same gate forced
//! through the reference's OTHER shape - the two-stage distilled pipeline
//! (half-res generate, real spatial x2 upscale, 3-step full-res refinement) -
//! which real usage (`examples/videogen`-style scripts at 1280x704) hits by
//! default the moment the request crosses `SINGLE_STAGE_MAX_TOKENS`, and
//! which had NO real-weight coverage at all: every existing keyframe gate
//! (`anchor_real.rs`, `motion_real.rs`, `anchors.rs`) runs at shapes small
//! enough to stay single-stage, and `should_two_stage` can never fire on the
//! tiny random-weight DiT (it requires `real_distilled`), so the tiny-weight
//! wiring tests cannot exercise this combination either.
//!
//! Swedish Embedded AB implements diffusion-sampler conditioning and its
//! numerical gates for its clients. If your team needs expertise in video
//! diffusion pipelines then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! `#[ignore]`d: two full real 22B generations plus one spatial-upscale
//! refinement pass, forced two-stage at the smallest two-stage-eligible shape
//! (384x192, both axes multiples of 64). Run explicitly:
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! BRAIN_LTXV_TEXT_ENCODER=<...gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
//! BRAIN_LTXV_UPSAMPLER_SPATIAL=<...latent-spatial-upscaler-x2-bf16-1.0.safetensors> \
//! cargo test -p brain-ltxv --test anchor_two_stage_real -- --ignored --nocapture
//! ```

use ltxv::pipeline::{generate, GenOpts, Paths};

const FRAMES: usize = 9;
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

fn write_still(dir: &std::path::Path, frame: &[u8]) -> String {
    std::fs::create_dir_all(dir).expect("temp dir");
    let path = dir.join("anchor.png");
    image::RgbImage::from_raw(WIDTH as u32, HEIGHT as u32, frame.to_vec()).expect("decoded frame as an RGB image").save(&path).expect("write the conditioning still");
    path.to_string_lossy().into_owned()
}

fn base_opts() -> GenOpts {
    GenOpts { frames: FRAMES, width: WIDTH, height: HEIGHT, seed: SEED, fps: FPS, eta: 1.0, guidance: 1.0, dit_config: "ltx25_22b".into(), device: Some("gpu".into()), ..GenOpts::default() }
}

/// Same bounds `anchor_real.rs` calibrated for the single-stage path. Not
/// claimed to be the right bound for stage 2 too - the point of printing the
/// actual numbers is to find out whether they are anywhere close, or whether
/// stage 2 loses the anchor outright, which is what distinguishes "needs a
/// looser bound" from "needs a real fix".
const MAX_SATURATION_RATIO: f64 = 1.08;
const MAX_FRAME0_DELTA: f64 = 7.0;

#[test]
#[ignore = "two full real 22B generations + one spatial upscale, forced two-stage"]
fn a_start_frame_anchor_survives_the_two_stage_refinement() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE to the real LTX-2.5 checkpoints");
    };
    if paths.spatial_upsampler.is_none() {
        return brain_testutil::skip("set BRAIN_LTXV_UPSAMPLER_SPATIAL to the real spatial x2 latent upscaler - the two-stage path needs it");
    }
    // SAFETY (test-only): this is the only test in this binary that reads or
    // writes this variable.
    unsafe { std::env::set_var("BRAIN_LTXV_TWO_STAGE", "1") };
    let cancel = capability::CancelToken::default();

    let (seed_video, _) = generate(&paths, PROMPT, &base_opts(), &cancel, |_, _, _| {}).expect("unconditioned run (source of the anchor)");
    let anchor = seed_video.frames[0].clone();

    let dir = std::env::temp_dir().join(format!("ltxv-anchor-2stage-gate-{}", std::process::id()));
    let still = write_still(&dir, &anchor);
    let o = GenOpts { start_frame: Some(still), ..base_opts() };
    let (video, timings) = generate(&paths, PROMPT, &o, &cancel, |_, _, _| {}).expect("image-to-video run, forced two-stage");
    let _ = std::fs::remove_dir_all(&dir);
    unsafe { std::env::remove_var("BRAIN_LTXV_TWO_STAGE") };
    assert_eq!(video.frames.len(), FRAMES);

    let anchor_sat = mean_saturation(&anchor);
    let curve: Vec<f64> = video.frames.iter().map(|f| mean_saturation(f)).collect();
    let delta = mean_abs_delta(&video.frames[0], &anchor);
    let ratio = curve[0] / anchor_sat;
    eprintln!("i2v (forced two-stage): {:.1}s, anchor saturation {anchor_sat:.4}, frame-0 saturation {:.4} (ratio {ratio:.3}), frame-0 delta {delta:.2}", timings.total(), curve[0]);
    eprintln!("i2v saturation curve: {}", curve.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>().join(" "));

    assert!(
        (1.0 / MAX_SATURATION_RATIO..=MAX_SATURATION_RATIO).contains(&ratio),
        "frame 0 does not reproduce the conditioning still after two-stage refinement: mean saturation {:.4} against the still's {anchor_sat:.4} (ratio {ratio:.3}, bound {MAX_SATURATION_RATIO:.2}, same bound anchor_real.rs uses for the single-stage path).",
        curve[0]
    );
    assert!(
        delta <= MAX_FRAME0_DELTA,
        "frame 0 does not reproduce the conditioning still after two-stage refinement: mean absolute pixel delta {delta:.2} (bound {MAX_FRAME0_DELTA:.2})."
    );
}
