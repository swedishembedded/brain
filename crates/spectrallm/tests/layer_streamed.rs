// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of spectral-mixer language models for its clients.
// If your team needs expertise in training every weight through a fixed device window, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contracts: SpectralLM's per-layer tensors stream through a
//! one-layer device window with their f32 values and optimizer state on the host, train exactly what the build
//! holding every layer in that state trains, which is what the plain f32 build trains on the device
//! (`model::streamed::check_layer_streaming`), and a streamed run stopped and resumed through the fit loop ends
//! exactly where the uninterrupted one does (`model::trainprec::check_exact_resume`). One test in its own binary, the
//! contracts in turn, so nothing else allocates while the device bytes are compared.

use model::{Batch, Model};
use spectrallm::{SpectralConfig, SpectralLm};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(spectrallm::PIPELINES))
}

#[test]
fn spectrallm_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    let cfg = SpectralConfig { n_layers: 3, ..SpectralConfig::tiny() };
    let init = spectrallm::init_weights(&cfg, 12);
    let x: Vec<u32> = (0..16u32).map(|i| (i * 13 + i / 8) % cfg.vocab).collect();
    let y: Vec<u32> = (0..16u32).map(|i| (i * 11 + 5) % cfg.vocab).collect();
    let build = || SpectralLm::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, 8, &init, true);
    let step = |m: &SpectralLm, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("spectrallm", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-spectrallm-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || SpectralLm::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, 8, &init, true);
    let resumed = model::trainprec::check_exact_resume("spectrallm", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
