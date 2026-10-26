// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of mixture-of-experts language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Activation recompute through the shared contract (`model::saved::check_recompute`): a build that keeps only the
//! residual between layers and replays each layer (its routed experts included) before its backward computes the
//! loss and gradients of the build that keeps every layer, bit for bit, on fewer device bytes. One test in its own
//! binary, so nothing else allocates while the device bytes are compared.

use glmdsa::{Glm, GlmConfig};

#[test]
fn glm_recomputes_every_layer_exactly() {
    // Four layers (one dense, three MoE): the MoE layers share one set of activation buffers.
    let cfg = GlmConfig { vocab: 23, block_size: 16, n_layers: 4, ..GlmConfig::tiny() };
    let init = glmdsa::init_weights(&cfg, 11);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
    let build = || Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg.clone(), 2, 8, &init);
    let step = |m: &Glm| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_recompute("glmdsa", build, step).unwrap();
}
