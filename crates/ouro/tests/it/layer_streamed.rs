// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of looped (recurrent-depth) language models for its
// clients. If your team needs expertise in training every weight through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning of a model that runs each layer once in every pass of its recurrence, through the
//! shared contracts: every visit loads the layer into a one-layer window, its gradient is carried on the host between
//! visits and the layer is updated once the backward is complete, which trains exactly what the build holding every
//! layer in that state trains and what the plain f32 build trains on the device
//! (`model::streamed::check_layer_streaming`); a streamed run stopped and resumed through the fit loop ends exactly where
//! the uninterrupted one does (`model::trainprec::check_exact_resume`). One test in its own binary, the contracts in
//! turn, so nothing else allocates while the device bytes are compared.

use model::{Batch, Model};
use ouro::model::Ouro;
use ouro::OuroConfig;

#[test]
fn ouro_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    let cfg = OuroConfig::tiny();
    let init = ouro::init::init_weights(&cfg, 3);
    let v = cfg.vocab;
    let x: Vec<u32> = (0..8).map(|i| (i * 5 + 2) % v).collect();
    let y: Vec<u32> = (0..8).map(|i| (i * 5 + 3) % v).collect();
    let build = || Ouro::new_train(cfg.clone(), 2, 4, &init);
    let step = |m: &Ouro, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("ouro", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-ouro-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || Ouro::new_train(cfg.clone(), 1, 8, &init);
    let resumed = model::trainprec::check_exact_resume("ouro", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
