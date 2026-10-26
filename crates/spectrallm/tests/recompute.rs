// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of spectral-mixer language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The run's `activations=` precision reaches SpectralLM through the generic path (`Model::set_saved_policy`):
//! SpectralLM keeps only the residual between layers and rebuilds each one before its backward, so `recompute` is
//! accepted as what it already does and keeping every layer is refused by name
//! (`model::saved::check_always_recomputes`), and holding those residuals in bf16 or FP8 trains to the format's
//! precision on fewer device bytes (`model::saved::check_quantized_boundaries`). One test in its own binary, the
//! contracts in turn, so nothing else allocates while the device bytes are compared.

use model::{Batch, Model};
use spectrallm::{SpectralConfig, SpectralLm};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(spectrallm::PIPELINES))
}

#[test]
fn spectrallm_recomputes_and_narrows_its_boundaries() {
    // Enough layers that the narrow boundaries outweigh the two f32 working buffers they decode into, a width FP8
    // rows pack (four codes to a word), and 48 tokens so FP8's rounding averages out in the head-width norm weights.
    let cfg = SpectralConfig { n_layers: 6, d_model: 12, ..SpectralConfig::tiny() };
    let init = spectrallm::init_weights(&cfg, 12);
    let x: Vec<u32> = (0..48u32).map(|i| (i * 13 + i / 8) % cfg.vocab).collect();
    let y: Vec<u32> = (0..48u32).map(|i| (i * 11 + 5) % cfg.vocab).collect();
    let build = || SpectralLm::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 4, 12, &init, true);
    model::saved::check_always_recomputes("spectrallm", &mut build()).unwrap();
    let step = |m: &SpectralLm| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_quantized_boundaries("spectrallm", build, step).unwrap();
}
