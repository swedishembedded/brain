// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of mixture-of-experts language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Activation recompute through the shared contracts: a build that keeps only the residual between layers and replays
//! each layer (its routed experts included) before its backward computes the loss and gradients of the build that
//! keeps every layer, bit for bit, on fewer device bytes (`model::saved::check_recompute`), and holding those
//! residuals in bf16 or FP8 trains to the format's precision on fewer still
//! (`model::saved::check_quantized_boundaries`). One test in its own binary, the contracts in turn, so nothing else
//! allocates while the device bytes are compared.

use deepseek2::config::DeepseekV2Config;
use deepseek2::model::DeepseekV2;

#[test]
fn deepseek2_recomputes_every_layer_and_narrows_its_boundaries() {
    // Four layers (one dense, three MoE): the MoE layers share one set of activation buffers. 48 tokens: FP8
    // boundaries flip a few top-k routing choices against the f32 build, and over a dozen tokens one flipped token
    // is a visible share of an expert's gradient.
    let mut cfg = DeepseekV2Config::tiny();
    cfg.shape.n_layers = 4;
    let init = <DeepseekV2 as model::Model>::init_weights(&cfg, 11);
    let v = cfg.vocab();
    let x: Vec<u32> = (0..48).map(|i| (i * 5 + 1) % v).collect();
    let y: Vec<u32> = (0..48).map(|i| (i * 5 + 2) % v).collect();
    let build = || DeepseekV2::new_on(gpu_core::testgpu::dev(deepseek2::model::PIPELINES), cfg.clone(), 4, 12, &init, true);
    let step = |m: &DeepseekV2| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_recompute("deepseek2", build, step).unwrap();
    model::saved::check_quantized_boundaries("deepseek2", build, step).unwrap();
}
