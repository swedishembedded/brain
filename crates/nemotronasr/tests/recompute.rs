// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of speech recognition models for its clients. If your
// team needs expertise in trading activation memory for recomputation, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The Nemotron transducer and acoustic model keep nothing between their forward and backward: the backward reruns
//! the forward and rebuilds every layer's residual checkpoints. They accept `activations=recompute` as what they
//! already do and refuse keeping every layer by name (`model::saved::check_always_recomputes`).

use nemotronasr::train::{AcousticConfig, AcousticModel, Transducer, TransducerConfig};

#[test]
fn the_transducer_always_recomputes() {
    let tcfg = TransducerConfig::tiny();
    let mut m = Transducer::new_from(tcfg.clone(), &Transducer::init(&tcfg, 7));
    model::saved::check_always_recomputes("nemotronasr transducer", &mut m).unwrap();
}

#[test]
fn the_acoustic_model_always_recomputes() {
    let acfg = AcousticConfig::tiny();
    let mut m = <AcousticModel as model::Model>::new(acfg.clone(), 1, acfg.t_frames, &AcousticModel::init(&acfg, 5));
    model::saved::check_always_recomputes("nemotronasr acoustic model", &mut m).unwrap();
}
