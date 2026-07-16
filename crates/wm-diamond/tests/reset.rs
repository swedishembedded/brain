// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `reset_initial` (the interactive Enter key) must rewind DIAMOND to the exact
//! initial state: the seed context AND the noise stream. Regression for the bug
//! where Enter cleared the context to grey and let the RNG keep advancing, so
//! the "reset" produced a fresh random dream instead of the start.
//!
//! Builds a tiny random-weight UNet on the CPU backend (no fixtures) and checks
//! that stepping a fixed action sequence, rewinding with `reset_initial`, and
//! stepping the SAME actions reproduces byte-identical frames.

use std::collections::HashMap;
use wm_core::WorldModel;
use wm_diamond::{DiamondConfig, DiamondUNet, DiamondWorldModel, Tensors};

fn cfg() -> DiamondConfig {
    DiamondConfig {
        img_channels: 3, num_steps_conditioning: 2, cond_channels: 16,
        depths: vec![1, 1], channels: vec![8, 8], attn_depths: vec![false, true],
        num_actions: 4, h: 8, w: 8, sigma_data: 0.5, sigma_offset_noise: 0.3,
    }
}

fn randn(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
    }).collect()
}

fn random_tensors(cfg: &DiamondConfig, seed: u64) -> Tensors {
    let mut t: Tensors = HashMap::new();
    for (i, (name, shape)) in cfg.param_list().into_iter().enumerate() {
        let n: usize = shape.iter().product();
        t.insert(name, (shape, randn(seed + i as u64, n, 0.25)));
    }
    t
}

#[test]
fn reset_initial_rewinds_context_and_noise() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let cfg = cfg();
    let frame_len = (cfg.img_channels * cfg.h * cfg.w) as usize;
    let nsc = cfg.num_steps_conditioning as usize;
    let unet = DiamondUNet::new(cfg.clone(), &random_tensors(&cfg, 1), Some("cpu"));
    let mut model = DiamondWorldModel::new(unet, 12345);

    // A non-trivial seed context so "initial" is clearly not grey/zeros.
    let ctx = randn(999, nsc * frame_len, 0.5).iter().map(|v| (v * 0.5 + 0.5).clamp(0.0, 1.0)).collect::<Vec<_>>();
    let actions = [1u32, 3, 2, 0, 3];

    model.reset(&ctx, &[]);
    let first_a = model.step(actions[0]);
    let run_a: Vec<Vec<f32>> = actions[1..].iter().map(|&a| model.step(a)).collect();

    // Rewind and replay the identical action sequence.
    model.reset_initial();
    let first_b = model.step(actions[0]);
    let run_b: Vec<Vec<f32>> = actions[1..].iter().map(|&a| model.step(a)).collect();

    // Byte-identical: the first frame and every subsequent frame must match.
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&first_a), bits(&first_b), "first frame differs after reset_initial");
    for (i, (a, b)) in run_a.iter().zip(&run_b).enumerate() {
        assert_eq!(bits(a), bits(b), "frame {} differs after reset_initial", i + 1);
    }

    // Sanity: the seed context actually influences the first frame — a reset to
    // EMPTY context (grey) must produce a different first frame, proving the
    // rewind restored real context rather than trivially matching a blank state.
    model.reset(&[], &[]);
    let first_grey = model.step(actions[0]);
    assert_ne!(bits(&first_a), bits(&first_grey), "seed context had no effect on the frame");
}
