// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use model::{Batch, Model};
use spectrallm::{SpectralConfig, SpectralLm};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(spectrallm::PIPELINES))
}

/// A build that streams its layers from construction never holds them all on the device, and trains what the build
/// streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn spectrallm_builds_its_layers_straight_into_the_window() {
    let cfg = SpectralConfig { n_layers: 3, ..SpectralConfig::tiny() };
    let init = spectrallm::init_weights(&cfg, 12);
    let x: Vec<u32> = (0..16u32).map(|i| (i * 13 + i / 8) % cfg.vocab).collect();
    let y: Vec<u32> = (0..16u32).map(|i| (i * 11 + 5) % cfg.vocab).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || SpectralLm::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, 8, &init, true);
    let build_streamed = || SpectralLm::new_resident_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, 8, &init, &resident);
    let step = |m: &SpectralLm, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("spectrallm", build, build_streamed, step).unwrap();
}
