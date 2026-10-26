// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of mixture-of-experts language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Activation recompute through the shared contracts: a build that keeps only the residual between layers and replays
//! each layer (every expert included) before its backward computes the loss and gradients of the build that keeps
//! every layer, bit for bit, on fewer device bytes (`model::saved::check_recompute`), and holding those residuals in
//! bf16 or FP8 trains to the format's precision on fewer still (`model::saved::check_quantized_boundaries`). One test
//! in its own binary, the contracts in turn, so nothing else allocates while the device bytes are compared.

use model::Model as _;
use toymoe::train::{Config, Trainer};

fn config(top_k: u32) -> Config {
    // Six layers: more boundaries than the narrow store's two f32 working buffers.
    Config { vocab: 64, block_size: 64, n_layers: 6, d_model: 32, n_heads: 4, n_experts: 3, top_k, d_ff: 64, aux_coef: 0.01, z_coef: 1e-4 }
}

#[test]
fn toymoe_recomputes_every_layer_and_narrows_its_boundaries() {
    let (b, t) = (2u32, 64u32);
    let x: Vec<u32> = (0..b * t).map(|i| i * 7 % 64).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 7 + 1) % 64).collect();
    let step = |m: &Trainer| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };

    let sparse = config(2);
    let init = Trainer::init_weights(&sparse, 5);
    let build = || Trainer::new(sparse.clone(), b, t, &init);
    model::saved::check_recompute("toymoe", build, step).unwrap();

    // Rounding the stream flips some top-k routing choices against the f32 build, and the experts they move tokens
    // between then see different gradients no token count averages out: the narrow boundaries are held to the
    // contract with every token routed to every expert, the gate weights alone differing smoothly.
    let dense = config(3);
    let init = Trainer::init_weights(&dense, 5);
    let build = || Trainer::new(dense.clone(), b, t, &init);
    model::saved::check_quantized_boundaries("toymoe", build, step).unwrap();
}
