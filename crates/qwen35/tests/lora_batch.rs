// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A training batch of more than one sequence on the CPU backend: the
//! Gated-DeltaNet backward must stay inside its buffers and produce the
//! gradient of the batch, which is the mean of its sequences' gradients.

mod common;

use std::collections::HashMap;

use common::*;
use gpu_core::Gpu;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn build(b: u32, init: &HashMap<String, Vec<f32>>) -> Qwen35 {
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), lora_config(), b, T as u32, init, &TrainOpts::default())
}

#[test]
fn a_two_sequence_batch_trains_on_the_cpu_backend_like_its_sequences_averaged() {
    let init = weights(&lora_config(), 0.3);
    let b2 = build(2, &init);
    let mut x = TOKENS.to_vec();
    let mut y = TARGETS.to_vec();
    x.extend(TOKENS.iter().rev());
    y.extend(TARGETS.iter().rev());
    b2.set_batch(&x, &y);
    b2.zero_grads();
    let loss2 = b2.forward();
    b2.backward();

    let single = |x: &[u32], y: &[u32]| {
        let m = build(1, &init);
        m.set_batch(x, y);
        m.zero_grads();
        let l = m.forward();
        m.backward();
        (l, m)
    };
    let rev_x: Vec<u32> = TOKENS.iter().rev().copied().collect();
    let rev_y: Vec<u32> = TARGETS.iter().rev().copied().collect();
    let (la, ma) = single(&TOKENS, &TARGETS);
    let (lb, mb) = single(&rev_x, &rev_y);
    assert!((loss2 - (la + lb) / 2.0).abs() < 1e-4 * loss2.abs().max(1.0), "loss {loss2} vs {la},{lb}");
    for n in b2.param_names() {
        let want: Vec<f32> = ma.read_grad(&n).iter().zip(mb.read_grad(&n)).map(|(a, b)| (a + b) / 2.0).collect();
        let got = b2.read_grad(&n);
        if want.iter().all(|v| *v == 0.0) {
            continue;
        }
        let c = cosine(&got, &want);
        assert!(c > 0.999, "{n}: batch gradient cosine {c}");
    }
}
