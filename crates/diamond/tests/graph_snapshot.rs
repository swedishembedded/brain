// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A frozen forward, so REFACTORING the graph is checkable without torch.
//!
//! `parity.rs` is the real reference gate, but its fixtures are not in git and
//! need torch to regenerate — so on a normal checkout it SKIPS, and the only
//! thing standing between a graph refactor and silently different numbers is
//! `train.rs`'s gradcheck, which verifies the backward is the forward's adjoint
//! and would happily stay green if BOTH moved together.
//!
//! This pins the forward itself: a tiny deterministic-weight DIAMOND UNet on
//! the CPU backend, its output hashed and stored below. It says nothing about
//! whether the graph matches DIAMOND — `parity.rs` owns that question — only
//! that it still computes what it computed before your change. That is exactly
//! the guarantee a migration onto `vae::blocks::Builder` needs.
//!
//! If you changed the graph ON PURPOSE, re-run with `WM_SNAPSHOT_DUMP=1` to
//! print the new values, and say in the commit message why they moved.

use std::collections::HashMap;

use diamond::{DiamondConfig, DiamondUNet, Tensors};

/// Deliberately irregular: channels that differ per level (so a schedule walked
/// the wrong way cannot pass), attention on only the second level, and a
/// channel count that is NOT a multiple of the GroupNorm group size at every
/// level — `num_groups(c) = max(c/32, 1)`, so c=8 exercises the 1-group path
/// while c=48 exercises the multi-group one.
fn cfg() -> DiamondConfig {
    DiamondConfig {
        img_channels: 3,
        num_steps_conditioning: 2,
        cond_channels: 16,
        depths: vec![1, 1],
        channels: vec![8, 48],
        attn_depths: vec![false, true],
        num_actions: 4,
        h: 8,
        w: 8,
        sigma_data: 0.5,
        sigma_offset_noise: 0.3,
    }
}

fn randn(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn random_tensors(cfg: &DiamondConfig, seed: u64) -> Tensors {
    let mut t: Tensors = HashMap::new();
    for (i, (name, shape)) in cfg.param_list().into_iter().enumerate() {
        let n: usize = shape.iter().product();
        t.insert(name, (shape, randn(seed + i as u64, n, 0.25)));
    }
    t
}

/// Order-sensitive summary: a plain sum would not notice two outputs swapping
/// places, which is precisely what a mis-walked channel schedule does.
fn digest(y: &[f32]) -> (f64, f64, f32, f32) {
    let sum: f64 = y.iter().map(|&v| v as f64).sum();
    let weighted: f64 = y.iter().enumerate().map(|(i, &v)| (i + 1) as f64 * v as f64).sum();
    let lo = y.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = y.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (sum, weighted, lo, hi)
}

#[test]
fn the_forward_graph_still_computes_what_it_computed_before() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let cfg = cfg();
    let unet = DiamondUNet::new(cfg.clone(), &random_tensors(&cfg, 1), Some("cpu"));

    let n_in = (cfg.img_channels * cfg.h * cfg.w) as usize;
    let nsc = cfg.num_steps_conditioning as usize;
    unet.set_context(&randn(999, nsc * n_in, 0.5));
    let y = unet.forward(&randn(4242, n_in, 1.0), 0.7, &[1, 3]);
    assert_eq!(y.len(), n_in, "the UNet must emit one image");

    let (sum, weighted, lo, hi) = digest(&y);
    if std::env::var("WM_SNAPSHOT_DUMP").is_ok() {
        println!("sum {sum:.9}  weighted {weighted:.9}  lo {lo:.9}  hi {hi:.9}");
    }

    // Frozen 2026-08-06, CPU backend, from the hand-recorded graph — the state
    // the `vae::blocks::Builder` migration had to reproduce.
    const SUM: f64 = -9.947_709_367;
    const WEIGHTED: f64 = -73.939_637_937;

    // fp32 reassociation across a graph this deep moves the low bits, so this
    // is a tight RELATIVE bound, not a bit-identity claim. Any real change to
    // the graph moves these by far more than 1e-5.
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-6);
    assert!(rel(sum, SUM) < 1e-5, "output sum moved: {sum:.9} vs {SUM:.9}");
    assert!(rel(weighted, WEIGHTED) < 1e-5, "output ORDER moved: {weighted:.9} vs {WEIGHTED:.9}");
    assert!(lo > -50.0 && hi < 50.0, "output left a plausible range: [{lo}, {hi}]");
}
