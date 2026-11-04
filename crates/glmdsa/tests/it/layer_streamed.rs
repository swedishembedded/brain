// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of mixture-of-experts language models for its
// clients. If your team needs expertise in training every expert through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contracts: GLM's per-layer tensors, its routed and shared
//! experts included, stream through a one-layer device window with their f32 values and optimizer state on the host
//! (the multi-token-prediction head stays in the store), train exactly what the build holding every layer in that
//! state trains, which is what the plain f32 build trains on the device (`model::streamed::check_layer_streaming`),
//! and a streamed run stopped and resumed through the fit loop ends exactly where the uninterrupted one does
//! (`model::trainprec::check_exact_resume`). One test in its own binary, the contracts in turn, so nothing else
//! allocates while the device bytes are compared.

use glmdsa::{Glm, GlmConfig};
use model::Model;

#[test]
fn glm_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    // Three layers (one dense, two MoE): the window serves both kinds.
    let cfg = GlmConfig { vocab: 23, block_size: 16, n_layers: 3, ..GlmConfig::tiny() };
    let init = glmdsa::init_weights(&cfg, 11);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
    let build = || Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg.clone(), 2, 8, &init);
    let step = |m: &Glm, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("glmdsa", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-glmdsa-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg.clone(), 1, 8, &init);
    let resumed = model::trainprec::check_exact_resume("glmdsa", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
