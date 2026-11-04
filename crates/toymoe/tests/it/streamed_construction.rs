// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use model::Model as _;
use toymoe::train::{Config, Trainer, MOE_ADAM};

/// A build that streams its layers (every expert included) from construction never holds them all on the device, and
/// trains what the build streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn toymoe_builds_its_layers_straight_into_the_window() {
    let cfg = Config { vocab: 64, block_size: 16, n_layers: 3, d_model: 32, n_heads: 4, n_experts: 3, top_k: 2, d_ff: 64, aux_coef: 0.01, z_coef: 1e-4 };
    let init = Trainer::init_weights(&cfg, 5);
    let x: Vec<u32> = (0..16).map(|i| i * 7 % 64).collect();
    let y: Vec<u32> = (0..16).map(|i| (i * 7 + 1) % 64).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Trainer::new(cfg.clone(), 2, 8, &init);
    let build_streamed = || Trainer::new_resident_on(cfg.clone(), 2, 8, &init, &resident);
    let step = |m: &Trainer, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, MOE_ADAM, clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("toymoe", build, build_streamed, step).unwrap();
}
