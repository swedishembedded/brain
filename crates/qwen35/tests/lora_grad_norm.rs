// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in finding why a fine-tune diverges then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The global gradient norm a clipped optimiser step measured is readable
//! after the step and equals the norm of the gradients the model holds, so a
//! diverging run can be told apart from a clipped one in its log.

mod common;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::Model;
use qwen35::model::{pipelines, Qwen35};

#[test]
fn the_clipped_step_reports_the_norm_of_the_averaged_gradient() {
    let init = weights(&lora_config(), 0.3);
    let m = Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), lora_config(), 1, T as u32, &init, &opts(Dtype::BF16));
    assert_eq!(Model::last_grad_norm(&m), None, "no step yet");
    let (_, grads) = loss_and_grads(&m);
    let raw: f64 = grads.iter().flat_map(|(_, g)| g.iter()).map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    assert!(raw > 0.0);
    let accum = 0.5;
    m.adamw_step(1, 1e-4, 0.0, Default::default(), Some(1e9), accum);
    m.poll_wait();
    let got = f64::from(Model::last_grad_norm(&m).expect("a clipped step reports its norm"));
    assert!((got - raw * f64::from(accum)).abs() < 1e-4 * raw, "reported {got}, the gradients hold {}", raw * f64::from(accum));
    m.adamw_step(2, 1e-4, 0.0, Default::default(), None, 1.0);
    m.poll_wait();
    assert_eq!(Model::last_grad_norm(&m), None, "an unclipped step measures nothing");
}
