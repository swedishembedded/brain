// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use glmdsa::{Glm, GlmConfig};
use model::Model;

/// A build that streams its layers (routed and shared experts included) from construction never holds them all on the
/// device, and trains what the build streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn glm_builds_its_layers_straight_into_the_window() {
    // Three layers (one dense, two MoE): the window serves both kinds.
    let cfg = GlmConfig { vocab: 23, block_size: 16, n_layers: 3, ..GlmConfig::tiny() };
    let init = glmdsa::init_weights(&cfg, 11);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg.clone(), 2, 8, &init);
    let build_streamed = || Glm::new_resident_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg.clone(), 2, 8, &init, &resident);
    let step = |m: &Glm, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("glmdsa", build, build_streamed, step).unwrap();
}
