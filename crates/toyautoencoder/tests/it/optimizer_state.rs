// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of compact neural networks for its clients.
// If your team needs expertise in training small models with narrow optimizer state, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches the autoencoder through the generic path every `ParamStore` trainer
//! shares (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the
//! first step, and a precision no optimizer runs is refused by name. `Autoencoder::new` builds its own device with the moment
//! kernels linked.

use toyautoencoder::{Autoencoder, AutoencoderConfig};

#[test]
fn autoencoder_trains_with_narrow_optimizer_state() {
    let cfg = AutoencoderConfig::tiny();
    let init = toyautoencoder::init_weights(&cfg, 7);
    let b = 3u32;
    let x: Vec<f32> = (0..(b * cfg.in_dim)).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    let lr = 5e-3;
    let build = || Autoencoder::new(cfg.clone(), b, &init);
    let step = |m: &Autoencoder| {
        m.zero_grads();
        m.set_batch(&x, &x);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("toyautoencoder", build, step, lr).unwrap();
}
