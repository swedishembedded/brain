// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The anchor has to survive the sampler.** A real-weight gate on
//! image-to-video (`--start-frame` with no `--end-frame`): the clip's first
//! decoded frame must BE the conditioning still, because that mechanism
//! (`VideoConditionByLatentIndex`, `latent_idx=0`, `strength=1.0`) pins the
//! whole of latent frame 0 to the still's own encoded latent for the entire
//! trajectory.
//!
//! Swedish Embedded AB implements diffusion-sampler conditioning and its
//! numerical gates for its clients. If your team needs expertise in video
//! diffusion pipelines then you can procure our services by sending an email
//! to info@swedishembedded.com.
//!
//! ## Why this gate exists
//!
//! Nothing else in this crate could see the defect it was written for. The
//! x0 conversion (`ltx_core.utils.to_denoised`) ran against the schedule's
//! SCALAR sigma instead of the per-token `Modality.timesteps` the reference's
//! `X0Model.forward` uses. For plain text-to-video the two are the same
//! number on every token, so every parity gate, every gradcheck and the
//! unconditioned motion gate stayed green. With an anchor frozen at timestep
//! 0 they are not: the sampler's terminal step short-circuits to the x0
//! estimate WITHOUT re-pinning, so the anchor came out multiplied by roughly
//! `1 + sigma_terminal` (1.421875 on the real distilled schedule).
//!
//! What that looked like, measured as mean HSV saturation per decoded frame
//! (64x64 downsample) on the real 22B Q8_0 checkpoint at 512x512 / 25f:
//! frame 0 at 0.555 against a conditioning still whose own saturation is
//! 0.461, then a trough down to 0.311 around frames 5-9 as the causal VAE
//! decoder's temporal receptive field smeared the over-driven latent frame
//! across its neighbours, then a recovery to 0.46-0.50 - the model's honest,
//! correct level - for the rest of the clip. The trough and the "unexplained
//! late-clip rise" were one artefact of one scale error on one latent frame.
//!
//! The fast, weight-free gate on the SAME defect is
//! `ltxv::pipeline`'s `a_frozen_token_survives_the_terminal_step_exactly`;
//! this file is the perceptual half, which is what proves the fast one is
//! testing the thing a viewer actually sees.
//!
//! `#[ignore]`d: two full real 22B generations (~6 minutes on a Tesla P40,
//! weight-streaming bound) - one unconditioned run to produce the anchor,
//! one conditioned on it. Run explicitly:
//!
//! ```text
//! BRAIN_LTXV_DIT=<...22b-distilled-transformer-Q8_0.gguf> \
//! BRAIN_LTXV_VAE=<...video-vae-conv-bf16.safetensors> \
//! BRAIN_LTXV_TEXT_ENCODER=<...gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf> \
//! cargo test -p brain-ltxv --test anchor_real -- --ignored --nocapture
//! ```

use ltxv::pipeline::{generate, GenOpts, Paths};

/// Same shape `motion_real.rs` calibrates at, for the same reason: 9 frames
/// is 2 latent frames, the least room a conditioned clip can have, so the
/// anchor is half the sequence and any leak onto it is at its most visible.
const FRAMES: usize = 9;
const WIDTH: usize = 384;
const HEIGHT: usize = 192;
const SEED: u64 = 42;
const FPS: usize = 8;

const PROMPT: &str = "a belgian malinois dog running at full speed along a mountain road, a small winged P40 graphics card flying beside it, camera tracking";

/// Mean HSV saturation over an RGB8 frame - the measurement the
/// investigation this gate came from used, reimplemented here so the gate
/// owns its own metric rather than depending on a script.
///
/// `rgb_to_hsv`'s saturation is `(max - min) / max`, `0` for pure black.
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

/// Mean absolute per-channel difference in 0-255 pixel units.
fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).sum::<f64>() / a.len() as f64
}

fn real_paths() -> Option<Paths> {
    let p = Paths::resolve(None, None, None, None).ok()?;
    p.dit.as_ref()?;
    Some(p)
}

/// The conditioning still, and the exact bytes `generate` will condition on.
///
/// Built from this pipeline's own unconditioned output rather than an image
/// fixture, exactly as `motion_real.rs` builds its anchors: in distribution,
/// already at the target resolution, so neither a resize nor an
/// out-of-distribution photo can be blamed for a failure. Written at
/// WIDTHxHEIGHT so `generate`'s own `resize_exact` is a no-op and the file
/// on disk IS what gets encoded.
fn write_still(dir: &std::path::Path, frame: &[u8]) -> String {
    std::fs::create_dir_all(dir).expect("temp dir");
    let path = dir.join("anchor.png");
    image::RgbImage::from_raw(WIDTH as u32, HEIGHT as u32, frame.to_vec()).expect("decoded frame as an RGB image").save(&path).expect("write the conditioning still");
    path.to_string_lossy().into_owned()
}

fn base_opts() -> GenOpts {
    GenOpts {
        frames: FRAMES,
        width: WIDTH,
        height: HEIGHT,
        seed: SEED,
        fps: FPS,
        // **The ancestral sampler, and it has to be.** `motion_real.rs` picks
        // the deterministic one (`eta = 0`) for reproducibility; copying that
        // here made this gate pass against the defect it was written for, at
        // bit-identical numbers.
        //
        // The two loops differ in WHERE `post_process_latent` runs.
        // Deterministic (`samplers._step_state`) re-pins the x0 ESTIMATE
        // before the step formula touches it, which overwrites a frozen
        // token's x0 with its clean content and so erases a bad x0
        // conversion completely. Ancestral
        // (`samplers._ancestral_euler_denoising_loop`) re-pins the STEPPED
        // latent instead and short-circuits the terminal `sigma_next == 0`
        // step to the raw x0 estimate - which never gets re-pinned. Only the
        // second one can see this class of defect, and the second one is what
        // LTX-2.5's own distilled stage 1 runs (`ANCESTRAL_SAMPLER_SINCE_
        // VERSION = (2, 5)`, `ANCESTRAL_ETA = 1.0`) and what
        // `GenOpts::default` therefore sets.
        //
        // Reproducibility is not given up: the renoise draw is
        // `data::rng::Rng` seeded from `GenOpts::seed`, so `eta = 1` is
        // exactly as run-to-run deterministic as `eta = 0`.
        eta: 1.0,
        guidance: 1.0,
        dit_config: "ltx25_22b".into(),
        device: Some("gpu".into()),
        ..GenOpts::default()
    }
}

/// A frozen anchor is only ever as faithful as one VAE encode/decode round
/// trip allows, so these are not bit-exactness bounds. Both were calibrated
/// by running THIS test at THIS shape and seed against the defective sampler
/// and against the fixed one, changing nothing else - the real 22B Q8_0 DiT
/// + real Gemma-4 encoder + real conv VAE on one Tesla P40:
///
/// | | frame-0 saturation | ratio vs the still (0.3037) | frame-0 delta |
/// |---|---:|---:|---:|
/// | scalar-sigma x0 conversion | 0.4522 | 1.489 | 12.84 |
/// | per-token x0 conversion | 0.3050 | 1.004 | 2.67 |
///
/// The saturation ratio is the discriminating measurement (1.004 vs 1.489
/// against a 1.08 bound); the pixel delta is the corroborating one, and its
/// bound sits 2.6x above what a correct run produced and 1.8x below what the
/// defect produced. A ratio near 1.42 is the defect's own signature - see
/// this file's module doc.
const MAX_SATURATION_RATIO: f64 = 1.08;
const MAX_FRAME0_DELTA: f64 = 7.0;

#[test]
#[ignore = "two full real 22B generations, ~6 minutes on a Tesla P40"]
fn a_start_frame_anchor_is_reproduced_by_the_clips_first_frame() {
    let Some(paths) = real_paths() else {
        return brain_testutil::skip("set BRAIN_LTXV_DIT + BRAIN_LTXV_VAE to the real LTX-2.5 checkpoints");
    };
    let cancel = capability::CancelToken::default();

    let (seed_video, _) = generate(&paths, PROMPT, &base_opts(), &cancel, |_, _, _| {}).expect("unconditioned run (source of the anchor)");
    let anchor = seed_video.frames[0].clone();

    let dir = std::env::temp_dir().join(format!("ltxv-anchor-gate-{}", std::process::id()));
    let still = write_still(&dir, &anchor);
    let o = GenOpts { start_frame: Some(still), ..base_opts() };
    let (video, timings) = generate(&paths, PROMPT, &o, &cancel, |_, _, _| {}).expect("image-to-video run");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(video.frames.len(), FRAMES);

    let anchor_sat = mean_saturation(&anchor);
    let curve: Vec<f64> = video.frames.iter().map(|f| mean_saturation(f)).collect();
    let delta = mean_abs_delta(&video.frames[0], &anchor);
    eprintln!("i2v: {:.1}s, anchor saturation {anchor_sat:.4}, frame-0 delta {delta:.2}", timings.total());
    eprintln!("i2v saturation curve: {}", curve.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>().join(" "));

    let ratio = curve[0] / anchor_sat;
    assert!(
        (1.0 / MAX_SATURATION_RATIO..=MAX_SATURATION_RATIO).contains(&ratio),
        "frame 0 does not reproduce the conditioning still: its mean saturation is {:.4} against the still's {anchor_sat:.4} (ratio {ratio:.3}, bound {MAX_SATURATION_RATIO:.2}). \
         A ratio near 1.42 is the signature of the anchor latent being converted to x0 at the schedule's scalar sigma instead of its own zero timestep - see this file's module doc.",
        curve[0]
    );
    assert!(
        delta <= MAX_FRAME0_DELTA,
        "frame 0 does not reproduce the conditioning still: mean absolute pixel delta {delta:.2} (bound {MAX_FRAME0_DELTA:.2}). \
         Latent frame 0 is pinned to the still's encoded latent for the whole trajectory, so this is a VAE round trip and nothing else."
    );
}
