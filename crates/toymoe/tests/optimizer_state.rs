// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of mixture-of-experts language models for its clients.
// If your team needs expertise in training sparse expert models with narrow optimizer state, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches the toy mixture-of-experts through the generic path every `ParamStore`
//! trainer shares (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on
//! the first step, and a host-only format is refused by name. `Trainer::new` builds its own device with the moment
//! kernels linked.

use model::Model as _;
use toymoe::train::{Config, Trainer, MOE_ADAM};

#[test]
fn moe_trains_with_narrow_optimizer_state() {
    let cfg = Config {
        vocab: 64,
        block_size: 16,
        n_layers: 1,
        d_model: 32,
        n_heads: 4,
        n_experts: 3,
        top_k: 2,
        d_ff: 64,
        aux_coef: 0.01,
        z_coef: 1e-4,
    };
    let init = Trainer::init_weights(&cfg, 5);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 64) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 64) as u32).collect();
    let lr = 1e-2;
    let build = || Trainer::new(cfg.clone(), 2, 8, &init);
    let step = |m: &Trainer| {
        m.zero_grads();
        m.set_batch(&x, &y);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, MOE_ADAM, None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("toymoe", build, step, lr).unwrap();
}
