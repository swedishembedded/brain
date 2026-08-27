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
//!   runs anywhere `make fetch/testdata` has run, with no 67 MB checkpoint and
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
        brain_testutil::skip(&format!("{p} absent - run `make fetch/testdata` (or tools/goldens/rrdbnet_dump_reference.py)"));
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

/// How the tile seam responds to the halo - the measurement that makes
/// `TILE_HALO` a number rather than a guess.
///
/// The comparison is tiled-vs-**one-tile-covering-everything**, not
/// tiled-vs-whole-image. Those two differ for a reason that has nothing to do
/// with seams: the whole-image path lets the convolutions zero-pad at the image
/// border, while any tiled path replicate-pads it. Comparing against the
/// single-tile result holds the border regime fixed, so what is left IS the
/// seam.
#[test]
fn the_tile_seam_shrinks_with_the_halo() {
    let Some(g) = goldens() else { return };
    let mut w: Tensors = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    for t in &g {
        if let Some(name) = t.name.strip_prefix("tiny_w_") {
            shapes.insert(name.to_string(), t.shape.clone());
            w.insert(name.to_string(), (t.shape.clone(), t.data.clone()));
        }
    }
    let cfg = RrdbConfig::from_tensors(&shapes).expect("derive");
    let scale = cfg.scale;
    let w = rrdbnet::import::validate(w, &cfg).expect("validate");
    let input = g.iter().find(|t| t.name == "tiny_input").expect("tiny_input");
    let (h, wd) = (input.shape[2] as u32, input.shape[3] as u32);

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let s = rrdbnet::caps::Session::new(gpu, cfg, w);

    // Reference: ONE tile covering the whole image, at the halo under test -
    // same border regime, no interior seam.
    let mut prev = f32::INFINITY;
    for halo in [4u32, 8, 16, 32] {
        let (whole, ow, oh) = s.upscale_with_halo(&input.data, wd, h, wd, halo).expect("one tile");
        let (tiled, _, _) = s.upscale_with_halo(&input.data, wd, h, 12, halo).expect("tiled");
        let max = whole.iter().zip(&tiled).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let over = whole.iter().zip(&tiled).filter(|(a, b)| (*a - *b).abs() > 1.0 / 255.0).count();
        eprintln!("  halo={halo:3} max|seam| = {max:.3e}  ({over} of {} px above 1/255)", ow * oh * 3);
        assert!(max <= prev * 1.05 + 1e-6, "halo {halo} made the seam WORSE ({max:.3e} vs {prev:.3e})");
        prev = max;
        assert_eq!((ow, oh), (wd * scale, h * scale));
    }
    eprintln!("tiny seam at halo=32: {prev:.3e}");
    // On the 2-block toy a large halo DOES clear an 8-bit step. That is a fact
    // about this fixture, not about the released net - see
    // `the_tile_seam_on_the_released_net`, where the same halo does not.
    assert!(prev < 1.0 / 255.0, "seam {prev:.3e} exceeds an 8-bit step at the largest halo tried");
}

/// The same seam measurement on the RELEASED net, whose 23 blocks give a far
/// larger receptive field than the 2-block tiny config - so a halo that is
/// ample there can be far too small here. Measuring only the toy would be
/// exactly the degenerate-fixture mistake of measuring only a toy config.
#[test]
fn the_tile_seam_on_the_released_net() {
    let Some(g) = goldens() else { return };
    let Ok(ckpt) = std::env::var("BRAIN_ESRGAN") else {
        brain_testutil::skip("BRAIN_ESRGAN unset");
        return;
    };
    if !std::path::Path::new(&ckpt).exists() {
        brain_testutil::skip(&format!("{ckpt} does not exist"));
        return;
    }
    let Some(input) = g.iter().find(|t| t.name == "x4plus_input") else {
        brain_testutil::skip("no x4plus goldens");
        return;
    };
    let (tensors, shapes, _) = rrdbnet::import::read(&ckpt).expect("read");
    let cfg = RrdbConfig::from_tensors(&shapes).expect("derive");
    let w = rrdbnet::import::validate(tensors, &cfg).expect("validate");
    let (h, wd) = (input.shape[2] as u32, input.shape[3] as u32);

    let gpu = gpu_core::testgpu::dev(&KERNELS);
    let s = rrdbnet::caps::Session::new(gpu, cfg, w);
    let mut best = f32::INFINITY;
    for halo in [16u32, 32, 48] {
        let (whole, _, _) = s.upscale_with_halo(&input.data, wd, h, wd, halo).expect("one tile");
        let (tiled, _, _) = s.upscale_with_halo(&input.data, wd, h, 12, halo).expect("tiled");
        let max = whole.iter().zip(&tiled).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let over = whole.iter().zip(&tiled).filter(|(a, b)| (*a - *b).abs() > 1.0 / 255.0).count();
        eprintln!("  x4plus halo={halo:3} max|seam| = {max:.3e}  ({over} px above 1/255)");
        best = best.min(max);
    }
    eprintln!("x4plus best seam: {best:.3e}");
}
