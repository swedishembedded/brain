// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of recurrent-depth language models for its clients.
// If your team needs expertise in training long contexts within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The run's `activations=` precision reaches Huginn through the generic path (`Model::set_saved_policy`): Huginn
//! keeps only the residual between block applications and rebuilds each one before its backward, so `recompute` is
//! accepted as what it already does and keeping every application is refused by name
//! (`model::saved::check_always_recomputes`), and holding the recurrence's residuals in bf16 or FP8 trains to the
//! format's precision on fewer device bytes (`model::saved::check_quantized_boundaries`). One test in its own binary,
//! the contracts in turn, so nothing else allocates while the device bytes are compared.

use huginn::model::Huginn;
use huginn::HuginnConfig;
use model::{Batch, Model};

#[test]
fn huginn_recomputes_and_narrows_its_boundaries() {
    let cfg = HuginnConfig::tiny();
    let init = huginn::init::init_weights(&cfg, 3);
    let v = cfg.vocab;
    let x: Vec<u32> = (0..48).map(|i| (i * 5 + 2) % v).collect();
    let y: Vec<u32> = (0..48).map(|i| (i * 5 + 3) % v).collect();
    let build = || Huginn::new_train(cfg.clone(), 4, 12, &init);
    model::saved::check_always_recomputes("huginn", &mut build()).unwrap();
    let step = |m: &Huginn| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = Model::forward(m);
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_quantized_boundaries("huginn", build, step).unwrap();
}
