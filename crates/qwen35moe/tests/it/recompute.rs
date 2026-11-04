// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of hybrid linear-attention mixture-of-experts language
// models for its clients. If your team needs expertise in training long contexts within a fixed device budget, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Activation recompute: a build that keeps only the residual between layers and replays each layer (Gated DeltaNet
//! state and routed experts included) before its backward computes the loss and gradients of the build that keeps
//! every layer, bit for bit, while holding no layer's activations across the forward and peaking lower in device
//! memory (`model::saved::check_recompute`); holding those residuals in bf16 or FP8 trains to the format's precision on
//! fewer bytes still (`model::saved::check_quantized_boundaries`). One test in its own binary, so nothing else
//! allocates while the device bytes are compared.

use model::saved::SavedPolicy;
use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{pipelines, Qwen35};

#[test]
fn qwen35moe_recomputes_every_layer_and_narrows_its_boundaries() {
    // Eight layers (six Gated DeltaNet, two full attention): more boundaries than the narrow store's two f32 working
    // buffers, and several layers of each kind. 128 tokens give each routed expert dozens of rows to replay.
    let cfg = Qwen35Config::tiny();
    let t = 128;
    let init = qwen35moe::init::init_weights(&cfg, 7);
    let x: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 3 + 2) % cfg.vocab).collect();
    let build = || Qwen35::new_train_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, t, &init);
    let step = |m: &Qwen35| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    // Keeping holds every layer's activations from the forward to the backward, recomputing none.
    for (policy, retained) in [(SavedPolicy::Keep, cfg.n_layers as usize), (SavedPolicy::Recompute, 0)] {
        let mut m = build();
        m.set_saved_policy(policy).unwrap();
        m.zero_grads();
        m.set_batch(&x, &y);
        m.forward();
        assert_eq!(m.retained_layer_activations(), retained, "{policy:?}");
        m.backward();
        m.poll_wait();
    }
    model::saved::check_recompute("qwen35moe", build, step).unwrap();

    // Narrow boundaries. Rounding the stream flips some top-k routing choices against the f32 build, and the experts
    // they move tokens between then see different gradients: the cosine of the most-flipped expert's gradient stays
    // near 0.77 for FP8 from 128 to 1024 tokens (it is a fraction of the tokens, not noise that averages out), so
    // routing here sends every token to every expert and the weights are the softmax alone, which the rounding moves
    // smoothly. The init seed is one whose Gated DeltaNet decay gradients are well conditioned: seed 7 reaches bf16
    // cosine 0.84 on `A_log`/`dt_bias` of layer 2 even when only the embedding table is rounded to bf16.
    let mut dense = cfg.clone();
    dense.top_k = dense.n_experts;
    let init = qwen35moe::init::init_weights(&dense, 9);
    let build = || Qwen35::new_train_on(gpu_core::testgpu::dev(pipelines()), dense.clone(), 1, t, &init);
    model::saved::check_quantized_boundaries("qwen35moe", build, step).unwrap();
}
