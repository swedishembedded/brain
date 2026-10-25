// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-correctness gates for structured
// and spectral neural-network operators for its clients. If your team needs
// expertise in proving that a custom operator's backward pass is right, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Gradient check for SpectralLM's training graph, for both mixers and both
//! transform paths.
//!
//! What it guards: the spectral mixer's backward reuses synthesis as the
//! adjoint of analysis and vice versa, its filter gradients are reductions
//! over rows accumulated into buffers other kernels also write, and the tied
//! embedding receives the head's gradient and the embedding's. A wrong
//! adjoint, a missed accumulation or an assignment where an addition belongs
//! shows up here.

use crate::{directional_check, Report};

/// Build the tiny SpectralLM with `mixer` and `transform`, set a fixed
/// causal-LM batch, and gradient-check every parameter.
pub fn check_spectrallm(seed: u64, mixer: spectrallm::Mixer, transform: spectrallm::TransformPath) -> Report {
    use spectrallm::{SpectralConfig, SpectralLm};

    let cfg = SpectralConfig { mixer, transform, ..SpectralConfig::tiny() };
    let init = spectrallm::init_weights(&cfg, seed);
    let model = SpectralLm::new_train(cfg, 2, 6, &init);
    // Next-token targets over two rows with different residues; the last
    // position of the second row is masked so IGNORE is exercised.
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let mut y: Vec<u32> = (0..12).map(|i| (i * 5 + 6) % 23).collect();
    y[11] = spectrallm::IGNORE;
    model::Model::set_batch(&model, model::Batch::Lm { tokens: &x, targets: &y });
    directional_check(&model, 1e-3, 8, seed ^ 0x5bec)
}
