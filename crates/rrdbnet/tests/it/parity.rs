// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RRDBNet forward parity, stage by stage, against the reference.
//!
//! Goldens come from `tools/goldens/rrdbnet_dump_reference.py`, which prefers
//! basicsr's own `RRDBNet` and otherwise reconstructs it from the paper - never
//! from a second reading of brain's code.
//!
//! TWO gates, deliberately:
//!
//! * **`tiny_*`** - a small config whose weights travel WITH the goldens, so it
//!   runs anywhere `make testdata/fetch` has run, with no 67 MB checkpoint and
//!   no network. Its dims are chosen so `num_feat` (16), `num_grow_ch` (8) and
//!   the image side (32) all differ: a degenerate config would hide a
//!   width-for-width swap.
//! * **`x4plus_*`** - the released checkpoint at its real 64/32/23 shape, which
//!   is the thing anyone actually runs. Skips unless `BRAIN_ESRGAN` names it.
//!
//! Run:
//!   BRAIN_ESRGAN=/path/to/RealESRGAN_x4plus.pth \
//!     cargo test --release -p brain-rrdbnet --test parity -- --nocapture

use std::collections::HashMap;

use rrdbnet::config::RrdbConfig;
use rrdbnet::caps::Upscaler;
use rrdbnet::model::{Rrdb, KERNELS};
use vae::blocks::Tensors;

/// Every stage of a correct fp32 replay sits at 1.0 to ten digits.
const GATE: f64 = 0.999_999_9;

/// Cosine cannot see a dropped scale factor, and this
/// net has two `* 0.2` residuals that a cosine-only ladder would miss
/// completely. So `rel_l2` is asserted alongside it, at every rung.
const REL_L2_GATE: f64 = 2e-4;

fn testdata(rel: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/{rel}")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().max(1e-30);
    (num / den).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn goldens() -> Option<Vec<checkpoint::safetensors::StTensor>> {
    let p = testdata("golden/esrgan/rrdbnet.safetensors");
    if !std::path::Path::new(&p).exists() {
        brain_testutil::skip(&format!("{p} absent - run `make testdata/fetch` (or tools/goldens/rrdbnet_dump_reference.py)"));
        return None;
    }
    Some(checkpoint::safetensors::read(&p).expect("read goldens"))
}

/// Compare every rung and report the worst. Returns the number that failed.
fn ladder(m: &Rrdb, g: &[checkpoint::safetensors::StTensor], prefix: &str) -> usize {
    let find = |n: &str| g.iter().find(|t| t.name == format!("{prefix}{n}"));
    let mut failed = 0usize;
    let (mut worst, mut worst_at) = (1.0f64, String::new());
    let mut compared = 0usize;
    for name in ["conv_first", "body.0", "body_out", "up1", "up2", "out"] {
        let Some(want) = find(name) else { continue };
        let got = m.read_tap(if name == "out" { "out" } else { name });
        assert_eq!(got.len(), want.data.len(), "{prefix}{name}: {} floats vs {}", got.len(), want.data.len());
        let c = cosine(&got, &want.data);
        let r = rel_l2(&got, &want.data);
        eprintln!("  {name:12} cosine={c:.10}  rel_l2={r:.3e}  max_abs={:.3e}", max_abs(&got, &want.data));
        if c < worst {
            worst = c;
            worst_at = name.into();
        }
        if c < GATE || r > REL_L2_GATE {
            failed += 1;
        }
        compared += 1;
    }
    assert!(compared >= 5, "only {compared} rungs compared - the goldens are incomplete");
    eprintln!("{prefix}: {compared} rungs, {failed} failed, worst {worst_at} at cosine {worst:.10}");
    failed
}

/// The checkpoint-free gate: weights come from the goldens themselves.
#[test]
fn tiny_forward_matches_the_reference() {
    let Some(g) = goldens() else { return };

    // Rebuild the reference's own weights, which the dumper wrote alongside.
    let mut w: Tensors = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for t in &g {
        if let Some(name) = t.name.strip_prefix("tiny_w_") {
            shapes.insert(name.to_string(), t.shape.clone());
            w.insert(name.to_string(), (t.shape.clone(), t.data.clone()));
        }
    }
    assert!(!w.is_empty(), "goldens carry no tiny_w_* weights");

    let cfg = RrdbConfig::from_tensors(&shapes).expect("derive config");
    eprintln!(
        "tiny: feat={} grow={} blocks={} scale={}x",
        cfg.num_feat, cfg.num_grow_ch, cfg.num_block, cfg.scale
    );
    // The dims must not be degenerate, or a width swap passes (lessons #4).
    assert_ne!(cfg.num_feat, cfg.num_grow_ch, "toy dims must differ");

    let w = rrdbnet::import::validate(w, &cfg).expect("weights cover the config");
    let input = g.iter().find(|t| t.name == "tiny_input").expect("tiny_input");
    let (h, wd) = (input.shape[2] as u32, input.shape[3] as u32);
    assert_ne!(h, cfg.num_feat, "image side must differ from the channel widths");

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let m = Rrdb::new(gpu, cfg, &w, h, wd, true);
    m.run(&input.data);
    assert_eq!(ladder(&m, &g, "tiny_"), 0, "tiny ladder");
}

/// The released x4plus checkpoint at its real shape.
#[test]
fn x4plus_forward_matches_the_reference() {
    let Some(g) = goldens() else { return };
    if !g.iter().any(|t| t.name == "x4plus_out") {
        brain_testutil::skip("goldens carry no x4plus_* (re-run the dumper with --ckpt)");
        return;
    }
    let Ok(ckpt) = std::env::var("BRAIN_ESRGAN") else {
        brain_testutil::skip("BRAIN_ESRGAN unset (point it at RealESRGAN_x4plus.pth)");
        return;
    };
    if !std::path::Path::new(&ckpt).exists() {
        brain_testutil::skip(&format!("{ckpt} does not exist"));
        return;
    }

    let (tensors, shapes, src) = rrdbnet::import::read(&ckpt).expect("read checkpoint");
    eprintln!("x4plus: {} tensors from {src:?}", tensors.len());
    let cfg = RrdbConfig::from_tensors(&shapes).expect("derive config");
    assert_eq!((cfg.num_feat, cfg.num_grow_ch, cfg.num_block, cfg.scale), (64, 32, 23, 4));
    let w = rrdbnet::import::validate(tensors, &cfg).expect("weights cover the config");

    let input = g.iter().find(|t| t.name == "x4plus_input").expect("x4plus_input");
    let (h, wd) = (input.shape[2] as u32, input.shape[3] as u32);

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let m = Rrdb::new(gpu, cfg, &w, h, wd, true);
    assert_eq!(m.out_hw(), (h * 4, wd * 4));
    m.run(&input.data);
    assert_eq!(ladder(&m, &g, "x4plus_"), 0, "x4plus ladder");
}

/// Deterministic stand-in weights for `cfg` (no checkpoint, no goldens): the
/// tiling contract is about geometry, so any weights that make every layer
/// matter will do, and these keep activations O(1) through the trunk.
fn synthetic_weights(cfg: &RrdbConfig) -> Tensors {
    let mut rng = data::rng::Lcg::new(0x5eed);
    cfg.param_list()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let fan_in = (n / shape[0]).max(1) as f32;
            let a = 1.0 / fan_in.sqrt();
            let data = (0..n).map(|_| rng.scaled(a)).collect();
            (name, (shape, data))
        })
        .collect()
}

/// Tiled output must equal the whole-image output, not approximate it.
///
/// Spec: with `tile > 0` every output pixel is computed from a window holding
/// its whole receptive field (or ending at the image edge, where the convs
/// zero-pad exactly as the whole-image pass does), so the two paths agree to
/// the last bit. Sizes are chosen so the tile divides neither side, the sides
/// are odd, one case tiles both axes and one tiles only x (the image is
/// shorter than a window in y, so y runs whole), and both scales are covered.
#[test]
fn tiled_upscale_equals_the_whole_image_pass() {
    for (scale, num_block, w, h, tile) in [(4u32, 2u32, 91u32, 85u32, 8u32), (4, 2, 83, 71, 8), (2, 2, 77, 79, 5)] {
        let cfg = RrdbConfig { in_channels: 3, out_channels: 3, num_feat: 8, num_grow_ch: 4, num_block, scale };
        let r = cfg.receptive_radius();
        assert!(w > tile + 2 * r || h > tile + 2 * r, "case {w}x{h} never tiles: window {}", tile + 2 * r);
        let weights = synthetic_weights(&cfg);
        let mut rng = data::rng::Lcg::new(7);
        let chw: Vec<f32> = (0..3 * (w * h) as usize).map(|_| rng.unit()).collect();

        let s = rrdbnet::caps::Session::new(gpu_core::testgpu::dev(&KERNELS), cfg, weights);
        let (whole, ow, oh) = s.upscale(&chw, w, h, 0, &mut |_, _| true).expect("whole");
        let (tiled, tw, th) = s.upscale(&chw, w, h, tile, &mut |_, _| true).expect("tiled");
        assert_eq!((tw, th), (ow, oh));
        assert_eq!((ow, oh), (w * scale, h * scale));
        let max = max_abs(&whole, &tiled);
        let differing = whole.iter().zip(&tiled).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("  x{scale} {w}x{h} tile {tile} R={r}: max|tiled-whole| = {max:.3e}, {differing} px not bitwise equal");
        assert_eq!(differing, 0, "x{scale} {w}x{h} tile {tile}: tiled differs from whole by up to {max:.3e}");
    }
}

/// The served action polls its token per window: fired before the call it
/// decodes nothing, fired at a window boundary the sweep stops there.
#[test]
fn a_tiled_upscale_stops_at_the_window_its_token_fires() {
    let cfg = RrdbConfig { in_channels: 3, out_channels: 3, num_feat: 8, num_grow_ch: 4, num_block: 1, scale: 2 };
    // R = 19 here, so a 60-pixel width sweeps 15 windows of 42 per row of cores.
    let (w, h, tile) = (60u32, 8u32, 4u32);
    let s = rrdbnet::caps::Session::new(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), synthetic_weights(&cfg));
    let mut rng = data::rng::Lcg::new(data::rng::random_seed());
    let hwc: Vec<f32> = (0..3 * (w * h) as usize).map(|_| rng.unit()).collect();
    let make = || capability::Invocation::new().set("tile", serde_json::json!(tile)).blob("image", capability::blob::image_blob(&hwc, w, h, 3));
    let run = |inv: &capability::Invocation, progress: &mut dyn FnMut(capability::Progress)| rrdbnet::caps::run_upscale(&s, inv, progress);
    capability_mock::cancellation::certify("rrdbnet upscale", &make, &run, capability_mock::cancellation::Bound::ONE_STEP);
}

/// The same contract on the RELEASED net (R = 349), skipped when the checkpoint
/// is absent. An image smaller than a window runs whole, so this needs one
/// larger than `tile + 2R`.
#[test]
fn tiled_upscale_equals_the_whole_image_pass_on_the_released_net() {
    let Ok(ckpt) = std::env::var("BRAIN_ESRGAN") else {
        brain_testutil::skip("BRAIN_ESRGAN unset");
        return;
    };
    if !std::path::Path::new(&ckpt).exists() {
        brain_testutil::skip(&format!("{ckpt} does not exist"));
        return;
    }
    let (tensors, shapes, _) = rrdbnet::import::read(&ckpt).expect("read");
    let cfg = RrdbConfig::from_tensors(&shapes).expect("derive");
    let w = rrdbnet::import::validate(tensors, &cfg).expect("validate");
    let tile = 16u32;
    let side = tile + 2 * cfg.receptive_radius() + 5;
    let mut rng = data::rng::Lcg::new(11);
    let chw: Vec<f32> = (0..3 * (side * side) as usize).map(|_| rng.unit()).collect();

    let s = rrdbnet::caps::Session::new(gpu_core::testgpu::dev(&KERNELS), cfg, w);
    let (whole, _, _) = s.upscale(&chw, side, side, 0, &mut |_, _| true).expect("whole");
    let (tiled, _, _) = s.upscale(&chw, side, side, tile, &mut |_, _| true).expect("tiled");
    let max = max_abs(&whole, &tiled);
    eprintln!("x4plus {side}x{side} tile {tile}: max|tiled-whole| = {max:.3e}");
    assert_eq!(max, 0.0, "tiled differs from whole");
}
