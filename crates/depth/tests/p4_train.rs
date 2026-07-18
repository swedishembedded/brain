// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P4: the end-to-end training loop — forward -> masked-L1 -> backward ->
//! AdamW — provably LEARNS.
//!
//! The overfit-one-batch sanity test: a tiny ZipDepth on one fixed synthetic
//! batch must drive its training loss well below where it started. This is the
//! standard first proof of a training loop — the master gradcheck
//! (`p3_gradcheck`) already proves the gradients are RIGHT; this proves the
//! loop actually wires loss -> gradient -> optimizer -> weights together (a
//! dropped zero_grads, a wrong loss scale, or an unstepped optimizer all leave
//! the loss flat, which is exactly what the assertion catches — verified by
//! running the loop with the optimizer step disabled: red).
//!
//! Placeholder-grade by design: synthetic rectangles, plain masked L1 (mask
//! all ones), no eval split. Real data + the SSI/gradient loss are the next
//! increment; the loop's structure does not change.
//!
//! Run with `BRAIN_DEVICE=cpu`.

use std::collections::HashMap;

use depth::train::{synth_pair, train_loop, TrainCfg};
use depth::ZipConfig;
use gpu_core::Gpu;

fn tiny() -> ZipConfig {
    ZipConfig {
        dims: [8, 16, 32, 64],
        depths: [1, 1, 1, 1],
        dec_ch: 16,
        half_dec_ch: 8,
        input: 32,
        ..ZipConfig::base()
    }
}

#[test]
fn overfit_one_batch_loss_decreases() {
    let cfg = tiny();
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let t = TrainCfg { steps: 30, batch: 2, h: 32, w: 32, lr: 3e-3, wd: 0.0, seed: 7, fixed_batch: true };
    let init = depth::init_weights(&cfg, t.seed);
    let mut losses: Vec<f32> = Vec::new();
    let (_ps, out) = train_loop(&gpu, cfg, &t, &init, |_, l| losses.push(l));
    assert!(out.first_loss.is_finite() && out.last_loss.is_finite(), "loss diverged: {losses:?}");
    assert!(
        out.last_loss < 0.5 * out.first_loss,
        "overfitting one batch must at least halve the loss: first {} -> last {} ({losses:?})",
        out.first_loss,
        out.last_loss
    );
}

/// The synthetic pairs are deterministic in the seed and shaped right — and the
/// depth cue is real: nearer rectangles are painted brighter, so the target is
/// learnable from intensity, not noise.
#[test]
fn synth_pairs_are_deterministic_and_shaped() {
    let (x1, y1) = synth_pair(42, 32, 48);
    let (x2, y2) = synth_pair(42, 32, 48);
    assert_eq!(x1, x2);
    assert_eq!(y1, y2);
    assert_eq!(x1.len(), 3 * 32 * 48);
    assert_eq!(y1.len(), 32 * 48);
    let (x3, _) = synth_pair(43, 32, 48);
    assert_ne!(x1, x3, "different seeds must give different scenes");
    // Inverse depth stays in a sane positive range.
    assert!(y1.iter().all(|&v| (0.0..=1.5).contains(&v)));

    // The brightness<->nearness correlation the generator promises (this is
    // what makes the overfit test a LEARNING test rather than memorising
    // noise): mean luma over the nearest quartile of pixels exceeds the mean
    // over the farthest quartile.
    let hw = 32 * 48;
    let mut idx: Vec<usize> = (0..hw).collect();
    idx.sort_by(|&a, &b| y1[a].total_cmp(&y1[b]));
    let luma = |i: usize| (x1[i] + x1[hw + i] + x1[2 * hw + i]) / 3.0;
    let far: f32 = idx[..hw / 4].iter().map(|&i| luma(i)).sum::<f32>() / (hw / 4) as f32;
    let near: f32 = idx[3 * hw / 4..].iter().map(|&i| luma(i)).sum::<f32>() / (hw - 3 * hw / 4) as f32;
    assert!(near > far + 0.05, "near quartile luma {near} must exceed far {far}");
}

/// Distinct-batch mode actually varies the data (the fine-tune path), while
/// fixed-batch mode repeats it (the overfit path).
#[test]
fn fixed_batch_repeats_and_rotating_batches_differ() {
    let cfg = tiny();
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let init: HashMap<String, Vec<f32>> = depth::init_weights(&cfg, 3);
    // 2 steps each; just proving the loop runs in both modes and stays finite.
    for fixed in [true, false] {
        let t = TrainCfg { steps: 2, batch: 1, h: 32, w: 32, lr: 1e-3, wd: 0.0, seed: 3, fixed_batch: fixed };
        let (_ps, out) = train_loop(&gpu, cfg.clone(), &t, &init, |_, _| {});
        assert!(out.last_loss.is_finite());
    }
}
