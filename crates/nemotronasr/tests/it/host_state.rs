// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of speech recognition models for its clients. If your
// team needs expertise in holding a trainer's optimizer state in narrow formats, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches the Nemotron transducer and acoustic model, whose parameters and
//! update live on the host, through the generic path (`model::hostopt::check_host_state_formats`): bf16 and int8-block
//! moments train like f32 on the first step, and a precision the host optimizer does not run is refused by name.

use model::{Batch, Model};
use nemotronasr::train::{AcousticConfig, AcousticModel, Transducer, TransducerConfig};

#[test]
fn the_transducer_trains_with_narrow_host_optimizer_state() {
    let tcfg = TransducerConfig::tiny();
    let init = Transducer::init(&tcfg, 7);
    let (t, dh) = (tcfg.t_frames as usize, tcfg.cfg.decoder_hidden as usize);
    let enc: Vec<f32> = (0..t * dh).map(|i| ((i * 7 % 11) as f32 / 11.0) - 0.5).collect();
    let labels = [1u32, 3];
    let lr = 1e-2;
    let build = || Transducer::new_from(tcfg.clone(), &init);
    let step = |m: &Transducer| {
        m.zero_grads();
        m.set_batch(Batch::Tensor { tokens: Some(&labels), inputs: &enc, targets: &[] });
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
    };
    model::hostopt::check_host_state_formats("nemotronasr transducer", build, step, lr).unwrap();
}

#[test]
fn the_acoustic_model_trains_with_narrow_host_optimizer_state() {
    let acfg = AcousticConfig::tiny();
    let init = AcousticModel::init(&acfg, 5);
    let (t, c) = (acfg.t_frames as usize, acfg.cfg.hidden as usize);
    let feats: Vec<f32> = (0..t * c).map(|i| ((i * 5 % 13) as f32 / 13.0 - 0.5) * 0.8).collect();
    let labels = [1u32, 3];
    let lr = 1e-2;
    let build = || AcousticModel::new(acfg.clone(), 1, acfg.t_frames, &init);
    let step = |m: &AcousticModel| {
        m.zero_grads();
        m.set_batch(Batch::Tensor { tokens: Some(&labels), inputs: &feats, targets: &[] });
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
    };
    model::hostopt::check_host_state_formats("nemotronasr acoustic model", build, step, lr).unwrap();
}
