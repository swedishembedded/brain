// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of looped language models for its clients.
// If your team needs expertise in training parameter-shared looped models with narrow optimizer state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches Ouro through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a precision no optimizer runs is refused by name. `Ouro::new_train` builds its own device with the moment kernels linked.

use model::{Batch, Model as _};
use ouro::model::Ouro;
use ouro::OuroConfig;

#[test]
fn ouro_trains_with_narrow_optimizer_state() {
    let cfg = OuroConfig::tiny();
    let init = ouro::init::init_weights(&cfg, 3);
    let v = cfg.vocab;
    let x: Vec<u32> = (0..8).map(|i| (i * 5 + 2) % v).collect();
    let y: Vec<u32> = (0..8).map(|i| (i * 5 + 3) % v).collect();
    let lr = 1e-2;
    let build = || Ouro::new_train(cfg.clone(), 2, 4, &init);
    let step = |m: &Ouro| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("ouro", build, step, lr).unwrap();
}

/// A layer window updates each layer as its backward completes; a layer that runs in every step of the recurrence has
/// no complete gradient until the last, so the request is refused with that reason rather than trained another way.
#[test]
fn ouro_refuses_to_stream_its_layers_by_name() {
    let cfg = OuroConfig::tiny();
    let init = ouro::init::init_weights(&cfg, 3);
    let mut m = Ouro::new_train(cfg, 2, 4, &init);
    let err = model::apply_train_precision(&mut m, &model::TrainPrecision::parse("weights=f32,window=1").unwrap(), "ouro").unwrap_err();
    assert!(err.contains("recurrence") && err.contains("moments=host"), "{err}");
}
