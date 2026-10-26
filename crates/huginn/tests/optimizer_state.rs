// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of recurrent-depth language models for its clients.
// If your team needs expertise in training depth-recurrent models with narrow optimizer state, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches Huginn through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a precision no optimizer runs is refused by name. `Huginn::new_train` builds its own device with the moment kernels linked.

use huginn::model::Huginn;
use huginn::HuginnConfig;
use model::{Batch, Model};

#[test]
fn huginn_trains_with_narrow_optimizer_state() {
    let cfg = HuginnConfig::tiny();
    let init = huginn::init::init_weights(&cfg, 3);
    let v = cfg.vocab;
    let x: Vec<u32> = (0..8).map(|i| (i * 5 + 2) % v).collect();
    let y: Vec<u32> = (0..8).map(|i| (i * 5 + 3) % v).collect();
    let lr = 1e-2;
    let build = || Huginn::new_train(cfg.clone(), 2, 4, &init);
    let step = |m: &Huginn| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        Model::forward(m);
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("huginn", build, step, lr).unwrap();
}

/// A layer window updates each layer as its backward completes; a layer that runs in every step of the recurrence has
/// no complete gradient until the last, so the request is refused with that reason rather than trained another way.
#[test]
fn huginn_refuses_to_stream_its_layers_by_name() {
    let cfg = HuginnConfig::tiny();
    let init = huginn::init::init_weights(&cfg, 3);
    let mut m = Huginn::new_train(cfg, 2, 4, &init);
    let err = model::apply_train_precision(&mut m, &model::TrainPrecision::parse("weights=f32,window=1").unwrap(), "huginn").unwrap_err();
    assert!(err.contains("recurrence") && err.contains("moments=host"), "{err}");
}
