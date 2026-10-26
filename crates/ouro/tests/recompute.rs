// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of looped (recurrent-depth) language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The run's `activations=` precision reaches Ouro through the generic path (`Model::set_saved_policy`): Ouro keeps
//! only the residual between the layers of every pass and rebuilds each one before its backward, so `recompute` is
//! accepted as what it already does and keeping every layer is refused by name
//! (`model::saved::check_always_recomputes`), and holding those residuals in bf16 or FP8 trains to the format's
//! precision on fewer device bytes (`model::saved::check_quantized_boundaries`). One test in its own binary, the
//! contracts in turn, so nothing else allocates while the device bytes are compared.

use model::{Batch, Model};
use ouro::model::Ouro;
use ouro::OuroConfig;

#[test]
fn ouro_recomputes_and_narrows_its_boundaries() {
    let cfg = OuroConfig::tiny();
    let init = ouro::init::init_weights(&cfg, 3);
    let v = cfg.vocab;
    let x: Vec<u32> = (0..48).map(|i| (i * 5 + 2) % v).collect();
    let y: Vec<u32> = (0..48).map(|i| (i * 5 + 3) % v).collect();
    let build = || Ouro::new_train(cfg.clone(), 4, 12, &init);
    model::saved::check_always_recomputes("ouro", &mut build()).unwrap();
    let step = |m: &Ouro| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_quantized_boundaries("ouro", build, step).unwrap();
}
