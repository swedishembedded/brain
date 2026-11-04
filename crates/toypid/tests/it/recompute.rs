// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of control transformers for its clients. If your team needs
// expertise in training sequence models within a fixed device budget, you can procure our services by sending an email
// to info@swedishembedded.com.

//! Activation recompute through the shared contracts: a build that keeps only the residual between layers and replays
//! each layer before its backward computes the loss and gradients of the build that keeps every layer, bit for bit,
//! on fewer device bytes (`model::saved::check_recompute`), and holding those residuals in bf16 or FP8 trains to the
//! format's precision on fewer still (`model::saved::check_quantized_boundaries`). One test in its own binary, the
//! contracts in turn, so nothing else allocates while the device bytes are compared.

use toypid::data::init_weights;
use toypid::{Pid, PidConfig};

#[test]
fn toypid_recomputes_every_layer_and_narrows_its_boundaries() {
    // Six layers: more boundaries than the narrow store's two f32 working buffers.
    let cfg = PidConfig { block_size: 32, n_layers: 6, d_model: 32, n_heads: 4, d_ff: 64, ..PidConfig::default_small() };
    let init = init_weights(&cfg, 3);
    let (b, t) = (4u32, 32u32);
    // Every position is supervised, so the gradients average over a batch's worth of rows.
    let x: Vec<u32> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 5 + 1) % cfg.u_bins).collect();
    let build = || Pid::new(cfg.clone(), b, t, &init);
    let step = |m: &Pid| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_recompute("toypid", build, step).unwrap();
    model::saved::check_quantized_boundaries("toypid", build, step).unwrap();
}
