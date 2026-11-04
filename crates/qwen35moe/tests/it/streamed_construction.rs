// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of mixture-of-experts models larger than the
// accelerator for its clients. If your team needs expertise in building a model straight into a fixed device window,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`):
//! the Qwen3.5 MoE decoder's per-layer tensors, its experts and Gated DeltaNet mixers included, are read into the
//! window's host state as the model is built and never all reach the device, and the model trains what the build
//! streamed after construction trains. One test in its own binary, so nothing else allocates while the device bytes
//! are compared.

use model::Model;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{pipelines, Qwen35};

#[test]
fn qwen35moe_builds_its_layers_straight_into_the_window() {
    let cfg = Qwen35Config::tiny();
    let init = qwen35moe::init::init_weights(&cfg, 8);
    let t = 16;
    let x: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 3 + 2) % cfg.vocab).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Qwen35::new_train_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, t, &init);
    let build_streamed = || Qwen35::new_resident_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, t, &init, &resident);
    let step = |m: &Qwen35, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("qwen35moe", build, build_streamed, step).unwrap();
}
