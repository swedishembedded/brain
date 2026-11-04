// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of spectral-mixer language models for its clients.
// If your team needs expertise in training frequency-domain sequence models with narrow optimizer state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches SpectralLM through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a precision no optimizer runs is refused by name.

use model::{Batch, Model};
use spectrallm::{SpectralConfig, SpectralLm};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(spectrallm::PIPELINES))
}

#[test]
fn spectrallm_trains_with_narrow_optimizer_state() {
    let cfg = SpectralConfig::tiny();
    let init = spectrallm::init_weights(&cfg, 12);
    let x: Vec<u32> = (0..16u32).map(|i| (i * 13 + i / 8) % cfg.vocab).collect();
    let y: Vec<u32> = (0..16u32).map(|i| (i * 11 + 5) % cfg.vocab).collect();
    let lr = 1e-2;
    let build = || SpectralLm::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, 8, &init, true);
    let step = |m: &SpectralLm| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("spectrallm", build, step, lr).unwrap();
}
