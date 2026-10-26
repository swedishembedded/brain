// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of hybrid convolution-attention language models for its
// clients. If your team needs expertise in training compact models with narrow optimizer state, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches LFM2 through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a host-only format is refused by name. `Lfm::new_train` builds its own device with the moment kernels linked.

use lfm2::{Lfm, LfmConfig};

#[test]
fn lfm2_trains_with_narrow_optimizer_state() {
    let cfg = LfmConfig::tiny();
    let init = lfm2::init::init_weights(&cfg, 11);
    let (b, t) = (2u32, 8u32);
    let n = b * t;
    let tokens: Vec<u32> = (0..n).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let targets: Vec<u32> = (0..n).map(|i| (i * 7 + 4) % cfg.vocab).collect();
    let lr = 1e-2;
    let build = || Lfm::new_train(cfg.clone(), b, t, &init);
    let step = |m: &Lfm| {
        m.zero_grads();
        m.set_batch(&tokens, &targets);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("lfm2", build, step, lr).unwrap();
}
