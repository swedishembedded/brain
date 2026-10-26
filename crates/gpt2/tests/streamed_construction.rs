// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use gpt2::{Gpt, GptConfig};
use model::Model;

/// A build that streams its layers from construction never holds them all on the device, and trains what the build
/// streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn gpt2_builds_its_layers_straight_into_the_window() {
    let cfg = GptConfig { vocab: 23, block_size: 16, n_layers: 3, ..GptConfig::tiny() };
    let init = gpt2::init_weights(&cfg, 11);
    let x: Vec<u32> = (0..16).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..16).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Gpt::new_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), cfg.clone(), 2, 8, &init);
    let build_streamed = || Gpt::new_resident_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), cfg.clone(), 2, 8, &init, &resident);
    let step = |m: &Gpt, t: u32, clip: Option<f32>| {
        m.zero_grads();
        Gpt::set_batch(m, &x, &y);
        let loss = Gpt::forward(m);
        Gpt::backward(m);
        Model::adamw_step(m, t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("gpt2", build, build_streamed, step).unwrap();
}
