// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use model::{Batch, Model};
use ouro::model::Ouro;
use ouro::OuroConfig;

/// A build that streams its layers from construction never holds them all on the device, and trains what the build
/// streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn ouro_builds_its_layers_straight_into_the_window() {
    let cfg = OuroConfig::tiny();
    let init = ouro::init::init_weights(&cfg, 3);
    let v = cfg.vocab;
    let x: Vec<u32> = (0..8).map(|i| (i * 5 + 2) % v).collect();
    let y: Vec<u32> = (0..8).map(|i| (i * 5 + 3) % v).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Ouro::new_train(cfg.clone(), 2, 4, &init);
    let build_streamed = || Ouro::new_resident_on(cfg.clone(), 2, 4, &init, &resident);
    let step = |m: &Ouro, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = m.forward();
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("ouro", build, build_streamed, step).unwrap();
}
