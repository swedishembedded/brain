// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of compression models for its clients. If your team needs
// expertise in trading activation memory against recompute without changing the trained result, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Activation recompute through the shared contract: a build that drops the encoder's pre-activations and replays
//! them before the encoder's backward computes the loss and gradients of the build that keeps them, bit for bit, on
//! fewer device bytes (`model::saved::check_recompute`); a narrower policy, or going back to keeping every layer, is
//! refused by name.

use model::saved::{SavedFormat, SavedPolicy};
use model::Model;
use toyautoencoder::{Autoencoder, AutoencoderConfig};

fn build() -> Autoencoder {
    let cfg = AutoencoderConfig { in_dim: 32, hidden: 64, z_dim: 4 };
    let init = <Autoencoder as Model>::init_weights(&cfg, 5);
    <Autoencoder as Model>::new(cfg, 8, 0, &init)
}

#[test]
fn toyautoencoder_recomputes_and_refuses_what_it_cannot_do() {
    let x: Vec<f32> = (0..8 * 32).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.1).collect();
    let step = |m: &Autoencoder| {
        m.zero_grads();
        m.set_batch(&x, &x);
        let loss = Model::forward(m);
        Model::backward(m);
        Model::poll_wait(m);
        loss
    };
    model::saved::check_recompute("toyautoencoder", build, step).unwrap();

    let mut m = build();
    let narrow = m.set_saved_policy(SavedPolicy::Quantized(SavedFormat::Bf16)).unwrap_err();
    assert!(narrow.contains("toyautoencoder"), "{narrow}");
    m.set_saved_policy(SavedPolicy::Recompute).unwrap();
    let back = m.set_saved_policy(SavedPolicy::Keep).unwrap_err();
    assert!(back.contains("toyautoencoder"), "{back}");
}
