// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Backward-compatibility gate for caption-only (unpaired) FLUX.2 fine-tuning.
//!
//! Paired reference→target training is strictly ADDITIVE: a sample with no
//! reference image must produce the same batch, the same loss and the same
//! gradients it produced before reference conditioning existed. The numbers
//! below were recorded from the unpaired trainer and are pinned here, so a
//! change to the reference path that perturbs the plain path fails on the
//! spot instead of silently moving every existing adapter's training signal.
//!
//! Swedish Embedded AB implements regression-gated training pipelines for its
//! clients. If your team needs expertise in keeping a model's training signal
//! reproducible across refactors, you can procure our services by sending an
//! email to info@swedishembedded.com.

use flux2::modelgrad::{grad_views, grads, init_model, make_flow_batch, Cfg};

/// The fixture: tiny klein topology, fixed weights, fixed data, fixed sigma.
fn fixture() -> (Cfg, flux2::modelgrad::ModelWeights<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let cfg = Cfg::tiny();
    let w = init_model::<f32>(&cfg, 7);
    let mut rng = data::rng::Rng::new(0xF1);
    let mut rf = || (rng.next_f64() - 0.5) as f32;
    let x0: Vec<f32> = (0..cfg.n_img() * cfg.in_channels).map(|_| rf()).collect();
    let ctx: Vec<f32> = (0..cfg.txt_len * cfg.context_in_dim).map(|_| rf()).collect();
    let noise: Vec<f32> = (0..x0.len()).map(|_| rf()).collect();
    (cfg, w, x0, ctx, noise)
}

/// A position-weighted digest of every gradient tensor, in `grad_views` order.
/// Position-weighted so a permutation of the same values is a different digest.
fn digest(g: &flux2::modelgrad::ModelGrads<f32>) -> f64 {
    let mut acc = 0.0f64;
    for (k, (_, v)) in grad_views(g).iter().enumerate() {
        for (i, &x) in v.iter().enumerate() {
            acc += x as f64 * ((k * 31 + i) as f64 * 0.017).sin();
        }
    }
    acc
}

#[test]
fn caption_only_training_is_bit_for_bit_what_it_was() {
    let (cfg, w, x0, ctx, noise) = fixture();
    let b = make_flow_batch(&cfg, &x0, &ctx, 0.45, &noise);
    let (loss, g) = grads(&cfg, &w, &b);
    let d = digest(&g);
    println!("loss {loss:.17e} digest {d:.17e}");
    assert_eq!(loss.to_bits(), LOSS.to_bits(), "unpaired loss moved: {loss:.17e}");
    assert_eq!(d.to_bits(), DIGEST.to_bits(), "unpaired gradients moved: {d:.17e}");
}

/// Recorded from the caption-only trainer before reference conditioning
/// existed. Not a tolerance and not a target: it is the value this fixture
/// produced, asserted bit-for-bit, because "the plain path is untouched" is an
/// exact claim.
const LOSS: f64 = 2.31704819316291832e-1;
const DIGEST: f64 = -1.19128803062919411e-1;
