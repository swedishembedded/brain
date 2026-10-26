// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use lfm2::config::{LayerType, LfmConfig};
use lfm2::model::Lfm;
use model::Model;

/// A build that streams its layers (convolution and attention alike) from construction never holds them all on the
/// device, and trains what the build streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn lfm2_builds_its_layers_straight_into_the_window() {
    let mut cfg = LfmConfig::tiny();
    cfg.layer_types = vec![LayerType::Conv, LayerType::Attention, LayerType::Conv, LayerType::Attention];
    let init = lfm2::init::init_weights(&cfg, 11);
    let (b, t) = (2u32, 8u32);
    let x: Vec<u32> = (0..b * t).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Lfm::new_train(cfg.clone(), b, t, &init);
    let build_streamed = || Lfm::new_resident_on(cfg.clone(), b, t, &init, &resident);
    let step = |m: &Lfm, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("lfm2", build, build_streamed, step).unwrap();
}
