// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of spectral-mixer language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The run's `activations=` precision reaches SpectralLM through the generic path (`Model::set_saved_policy`): SpectralLM
//! keeps only the residual between layers and rebuilds each one before its backward, so `recompute` is accepted as
//! what it already does, and keeping everything or narrower residuals are refused by name
//! (`model::saved::check_always_recomputes`).

use spectrallm::{SpectralConfig, SpectralLm};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(spectrallm::PIPELINES))
}

#[test]
fn spectrallm_answers_the_activations_key_from_its_recomputing_backward() {
    let cfg = SpectralConfig::tiny();
    let init = spectrallm::init_weights(&cfg, 12);
    let mut m = SpectralLm::new_on(gpu_core::testgpu::dev(pipelines()), cfg, 2, 8, &init, true);
    model::saved::check_always_recomputes("spectrallm", &mut m).unwrap();
}
