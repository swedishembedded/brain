// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of hybrid linear-attention mixture-of-experts language
// models for its clients. If your team needs expertise in training every expert through a fixed device window, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contracts: the per-layer tensors of the Qwen3.5 MoE decoder, its
//! Gated DeltaNet mixers and routed experts included, stream through a one-layer device window with their f32 values
//! and optimizer state on the host, train exactly what the build holding every layer in that state trains, which is
//! what the plain f32 build trains on the device (`model::streamed::check_layer_streaming`), and a streamed run
//! stopped and resumed through the fit loop ends exactly where the uninterrupted one does
//! (`model::trainprec::check_exact_resume`). The model is eager: its forward and backward report each layer's edges
//! as they run it. One test in its own binary, the contracts in turn, so nothing else allocates while the device bytes
//! are compared.

use model::Model;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{pipelines, Qwen35};

#[test]
fn qwen35moe_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    let cfg = Qwen35Config::tiny();
    let init = qwen35moe::init::init_weights(&cfg, 8);
    let t = 16;
    let x: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 3 + 2) % cfg.vocab).collect();
    let build = || Qwen35::new_train_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, t, &init);
    let step = |m: &Qwen35, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("qwen35moe", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-qwen35moe-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || Qwen35::new_train_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, 8, &init);
    let resumed = model::trainprec::check_exact_resume("qwen35moe", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
