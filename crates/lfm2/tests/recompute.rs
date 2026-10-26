// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of hybrid convolution/attention language models for its
// clients. If your team needs expertise in training long contexts within a fixed device budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Activation recompute through the shared contracts: a build that keeps only the residual between layers and replays
//! each layer before its backward computes the loss and gradients of the build that keeps every layer, bit for bit,
//! on fewer device bytes (`model::saved::check_recompute`), and holding those residuals in bf16 or FP8 trains to the
//! format's precision on fewer still (`model::saved::check_quantized_boundaries`). One test in its own binary, the
//! contracts in turn, so nothing else allocates while the device bytes are compared.

use lfm2::config::{LayerType, LfmConfig};
use lfm2::model::Lfm;

#[test]
fn lfm2_recomputes_every_layer_and_narrows_its_boundaries() {
    // Four conv and four attention layers: each kind shares one set of activation buffers across several layers, and
    // the narrow boundaries outweigh the two f32 working buffers they decode into.
    let mut cfg = LfmConfig::tiny();
    cfg.layer_types = [LayerType::Conv, LayerType::Attention].repeat(4);
    let init = lfm2::init::init_weights(&cfg, 11);
    let (b, t) = (2u32, 6u32);
    let x: Vec<u32> = (0..b * t).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let build = || Lfm::new_train(cfg.clone(), b, t, &init);
    let step = |m: &Lfm| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_recompute("lfm2", build, step).unwrap();
    model::saved::check_quantized_boundaries("lfm2", build, step).unwrap();
}
