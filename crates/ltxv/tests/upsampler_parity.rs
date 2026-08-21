// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5 latent upscaler parity (spatial x2, temporal x2) against
//! `tools/goldens/ltxv_upsampler_dump_reference.py`, climbed the same way
//! `vae_parity.rs` does:
//!
//! 1. [`ltxv_upsampler_spatial_matches_reference`] /
//!    [`ltxv_upsampler_temporal_matches_reference`] - every dumped block tap,
//!    real weights, on the golden's own synthetic `[128,2,4,4]` latent.
//! 2. [`ltxv_upsampler_import_covers_both_shipped_checkpoints`] - the
//!    importer against both REAL 72-tensor files, both directions.
//!
//! Needs the real upscaler weights and the golden fixture; skips loudly
//! without them (`BRAIN_REQUIRE_FIXTURES=1` upgrades a skip to a failure,
//! same convention as every other parity suite in this repo).

use std::path::Path;

use ltxv::import::import_upsampler;
use ltxv::upsampler::{LatentUpsampler, LatentUpsamplerConfig};

// ------------------------------------------------------------------ metrics

/// Same formula `vae_parity.rs`'s own `cosine` uses - reimplemented locally
/// per this repo's convention (every parity test file owns its tiny metric
/// helpers rather than sharing a crate for them).
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch ({} vs {})", a.len(), b.len());
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    let den = na.sqrt() * nb.sqrt();
    if den <= 0.0 {
        0.0
    } else {
        d / den
    }
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// The largest absolute deviation this gate tolerates against the golden.
///
/// **Asserted, not merely printed, and that distinction is the whole point.**
/// This function used to compute `max_abs`, print it, and gate on cosine
/// alone - and cosine is SCALE-INVARIANT, so a port that returned exactly
/// `k * golden` for any `k` passed at cosine 1.000000000. That is not
/// hypothetical: the missing per-channel un-normalize around
/// `LatentUpsampler` (see `ltxv::upsampler::upsample_video`) was a scale
/// error of very nearly that shape, it cost half the latent's variance in a
/// real generation, and nothing in this file could see it. Every tap
/// measures 6.7e-6..3.4e-5 against the golden, so this bound is ~30x the
/// worst real deviation and orders of magnitude below any scale error.
const MAX_ABS_BOUND: f32 = 1e-3;

fn report(label: &str, got: &[f32], want: &[f32], min_cos: f64) {
    assert_eq!(got.len(), want.len(), "{label}: {} values vs {}", got.len(), want.len());
    let (c, m) = (cosine(got, want), max_abs(got, want));
    eprintln!("{label}: cosine={c:.9}  max_abs={m:.3e}  n={}", got.len());
    assert!(c >= min_cos, "{label}: cosine {c:.9} < {min_cos}");
    assert!(m <= MAX_ABS_BOUND, "{label}: max_abs {m:.3e} > {MAX_ABS_BOUND:.0e} - cosine can be 1.0 while the MAGNITUDE is wrong, which is exactly what this bound is for");
}

// ---------------------------------------------------------- real fixtures

/// `BRAIN_LTXV_UPSAMPLER_SPATIAL`/`_TEMPORAL`, else the repo-relative
/// `resources/ltxv/weights/latent_upscale_models/` the real files ship
/// under - a variable rather than a literal machine path so this test passes
/// on any checkout that fetched the resource, not just the one it was
/// written on (same convention as `vae_parity.rs`'s `weights_path`).
/// The real conv video VAE, whose `per_channel_statistics` the upscaler
/// sandwich needs. Same resolution order as the upscaler paths above.
fn vae_weights_path() -> Option<String> {
    if let Ok(p) = std::env::var("BRAIN_LTXV_VAE") {
        return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
    }
    let p = format!("{}/../../resources/ltxv/weights/vae/ltx-2.5-video-vae-conv-bf16.safetensors", env!("CARGO_MANIFEST_DIR"));
    Path::new(&p).exists().then_some(p)
}

fn weights_path(env: &str, rel: &str) -> Option<String> {
    if let Ok(p) = std::env::var(env) {
        return (!p.is_empty() && Path::new(&p).exists()).then_some(p);
    }
    let p = format!(
        "{}/../../resources/ltxv/weights/latent_upscale_models/{rel}",
        env!("CARGO_MANIFEST_DIR")
    );
    Path::new(&p).exists().then_some(p)
}

fn spatial_weights_path() -> Option<String> {
    weights_path("BRAIN_LTXV_UPSAMPLER_SPATIAL", "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
}

fn temporal_weights_path() -> Option<String> {
    weights_path("BRAIN_LTXV_UPSAMPLER_TEMPORAL", "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors")
}

struct Fixture {
    t: Vec<checkpoint::safetensors::StTensor>,
}

impl Fixture {
    fn get(&self, name: &str) -> &[f32] {
        &self.t.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no golden {name}")).data
    }
}

fn run(label: &str, cfg: &LatentUpsamplerConfig, weights_path: Option<String>) {
    let Some(wp) = weights_path else {
        brain_testutil::skip(&format!("{label}: set BRAIN_LTXV_UPSAMPLER_{} to the real checkpoint", label.to_uppercase()));
        return;
    };
    let fx_path = brain_testutil::testdata(&format!("golden/ltxv/upsampler/{label}.safetensors"));
    if !Path::new(&fx_path).exists() {
        brain_testutil::skip(&format!("fixture {fx_path} absent - run tools/goldens/ltxv_upsampler_dump_reference.py"));
        return;
    }

    let raw = checkpoint::safetensors::read(&wp).expect("read real upscaler weights");
    let w = import_upsampler(raw, cfg).expect("import real upscaler weights");
    let fx = Fixture { t: checkpoint::safetensors::read(&fx_path).expect("read golden") };

    let input = fx.get("input");
    // golden input is `[in_channels, t, h, w]` - recover t/h/w from the
    // manifest-known `in_channels` and the dumper's own fixed `4x4` spatial
    // size (2 frames), matching `--frames 2 --size 4`.
    let ic = cfg.in_channels as usize;
    let n_thw = input.len() / ic;
    // The dumper's synthetic latent is always `t=2, h=w=size` with
    // `size*size*t == n_thw` - reconstructed instead of hardcoded so a future
    // `--frames`/`--size` regeneration is not silently mismatched here.
    let t = 2usize;
    let hw = n_thw / t;
    let size = (hw as f64).sqrt() as u32;
    assert_eq!((size * size) as usize, hw, "golden input {n_thw} elements/channel doesn't factor as t=2 * size^2");

    let up = LatentUpsampler::build(cfg, &w, t as u32, size, size, None);
    let out = up.upsample(input);

    report(&format!("{label}: output"), &out, fx.get("output"), 0.999999);

    let taps = [
        "initial_conv",
        "initial_norm",
        "initial_activation",
        "res_blocks.0",
        "res_blocks.1",
        "res_blocks.2",
        "res_blocks.3",
        "upsampler",
        "post_upsample_res_blocks.0",
        "post_upsample_res_blocks.1",
        "post_upsample_res_blocks.2",
        "post_upsample_res_blocks.3",
        "final_conv",
    ];
    for name in taps {
        let got = up.read_tap(name).unwrap_or_else(|| panic!("{label}: no Rust tap {name}"));
        let want = fx.get(&format!("tap_{name}"));
        report(&format!("{label}: tap {name}"), &got, want, 0.999999);
    }
}

#[test]
fn ltxv_upsampler_spatial_matches_reference() {
    run("spatial", &LatentUpsamplerConfig::spatial_x2(), spatial_weights_path());
}

#[test]
fn ltxv_upsampler_temporal_matches_reference() {
    run("temporal", &LatentUpsamplerConfig::temporal_x2(), temporal_weights_path());
}

/// The importer against BOTH real shipped files, both directions.
#[test]
fn ltxv_upsampler_import_covers_both_shipped_checkpoints() {
    for (cfg, wp, env) in [
        (LatentUpsamplerConfig::spatial_x2(), spatial_weights_path(), "BRAIN_LTXV_UPSAMPLER_SPATIAL"),
        (LatentUpsamplerConfig::temporal_x2(), temporal_weights_path(), "BRAIN_LTXV_UPSAMPLER_TEMPORAL"),
    ] {
        let Some(wp) = wp else {
            brain_testutil::skip(&format!("set {env} to the real checkpoint"));
            continue;
        };
        let raw = checkpoint::safetensors::read(&wp).expect("read real upscaler weights");
        let n = raw.len();
        let manifest = cfg.tensor_manifest();
        let w = import_upsampler(raw, &cfg).expect("import real upscaler weights");
        assert_eq!(n, 72, "shipped checkpoint has {n} tensors, expected 72");
        assert_eq!(w.len(), manifest.len());
    }
}

/// **The upscaler runs in RAW VAE latent space, and something has to put it
/// there.** `ltx_core.model.upsampler.model.upsample_video` un-normalizes
/// the diffusion latent with the VAE's `per_channel_statistics`, upsamples,
/// and re-normalizes; `ltxv::upsampler::upsample_video` is that function.
///
/// Gated as the exact composition rather than as a statistic, because a
/// statistic is not invariant here: what the sandwich preserves is the
/// variance of a latent that actually lives in the diffusion distribution
/// (measured on a real 25-frame 960x544 stage-1 latent: per-frame std
/// 1.070/0.960/1.013/1.069 in, 1.014/0.919/0.994/1.074 out, against
/// 0.504/0.524/0.530/0.465 for the bare call), and an i.i.d. draw is not
/// such a latent - un-normalizing one by per-channel statistics it does not
/// have scales it by an arbitrary amount.
///
/// So this asserts the two things that ARE invariant: that
/// `upsample_video` is exactly `normalize . upsample . un_normalize` in that
/// order and direction, and that this genuinely differs from the bare call -
/// the second half being what stops the gate passing if `upsample_video`
/// ever quietly became `upsample` again. Compared on BIT PATTERNS, not `==`
/// on `f32`.
#[test]
fn the_upscaler_is_un_normalized_around_exactly_as_the_reference_does_it() {
    let (Some(up_path), Some(vae_path)) = (spatial_weights_path(), vae_weights_path()) else {
        return brain_testutil::skip("set BRAIN_LTXV_UPSAMPLER_SPATIAL + BRAIN_LTXV_VAE to the real checkpoints");
    };
    let cfg = LatentUpsamplerConfig::spatial_x2();
    let uw = import_upsampler(checkpoint::safetensors::read(&up_path).expect("read upscaler"), &cfg).expect("import upscaler");
    let vw = ltxv::import::import_vae(checkpoint::safetensors::read(&vae_path).expect("read vae"), &ltxv::vae3d::LtxVaeConfig::conv25()).expect("import vae");
    let (mean, std) = ltxv::vae3d::per_channel_statistics(&vw);
    assert!(std.iter().all(|&v| v.abs() > 0.0), "a zero per-channel std would make `normalize` a division by zero");

    let (c, t, h, w) = (mean.len(), 2usize, 4usize, 4usize);
    let mut rng = data::rng::Rng::new(7);
    let latent: Vec<f32> = (0..c * t * h * w).map(|_| rng.next_gaussian() as f32).collect();
    let ups = LatentUpsampler::build(&cfg, &uw, t as u32, h as u32, w as u32, Some("gpu"));

    // The reference's three steps, written out here so the helper cannot be
    // the only statement of its own contract.
    let plane_in = latent.len() / c;
    let mut raw = latent.clone();
    for (ci, chunk) in raw.chunks_exact_mut(plane_in).enumerate() {
        for v in chunk {
            *v = *v * std[ci] + mean[ci];
        }
    }
    let mut want = ups.upsample(&raw);
    let plane_out = want.len() / c;
    for (ci, chunk) in want.chunks_exact_mut(plane_out).enumerate() {
        for v in chunk {
            *v = (*v - mean[ci]) / std[ci];
        }
    }

    let got = ltxv::upsampler::upsample_video(&ups, &mean, &std, &latent);
    assert_eq!(got.len(), want.len(), "{} values vs {}", got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "value {i} differs from the reference composition: {a} vs {b}");
    }

    let bare = ups.upsample(&latent);
    let scale = bare.iter().fold(0.0f32, |a, &b| a.max(b.abs())).max(1e-6);
    let rel = max_abs(&got, &bare) / scale;
    eprintln!("un-normalized-around vs bare: relative max deviation {rel:.4}");
    assert!(rel > 0.05, "the sandwich made no difference (relative max deviation {rel:.4}) - `upsample_video` is not doing anything");
}
