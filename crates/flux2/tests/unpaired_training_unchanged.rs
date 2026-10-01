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
    assert_pinned("loss", loss, LOSS);
    assert_pinned("gradient digest", d, DIGEST);
}

/// Exact on the architecture the values were recorded on. Elsewhere the
/// transcendental functions the fixture and the model call (`sin`, `exp`,
/// `tanh`) come from a different libm and differ in the last bits of an f32
/// pipeline - measured on aarch64 as 3.4e-8 relative on the loss and 2.7e-7 on
/// the digest - so there the pin is a relative bound of 1e-5, about 40 times the
/// observed difference and far below the effect of a real change to the plain path.
fn assert_pinned(what: &str, got: f64, pinned: f64) {
    if cfg!(target_arch = "x86_64") {
        assert_eq!(got.to_bits(), pinned.to_bits(), "unpaired {what} moved: {got:.17e}");
    } else {
        let rel = ((got - pinned) / pinned).abs();
        assert!(rel < 1e-5, "unpaired {what} moved: {got:.17e} vs {pinned:.17e} (rel {rel:.2e})");
    }
}

/// Recorded from the caption-only trainer before reference conditioning
/// existed. Not a tolerance and not a target: it is the value this fixture
/// produced, asserted bit-for-bit on the recording architecture (see [`assert_pinned`]),
/// because "the plain path is untouched" is an exact claim.
const LOSS: f64 = 0.23170481931629183;
const DIGEST: f64 = -0.11912880306291941;
