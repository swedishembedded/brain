// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A Gated-DeltaNet layer trained in segments keeps one segment's
//! intermediates instead of the whole sequence's, and computes the same loss
//! and adapter gradients as the one-shot layer - with the recurrent state and
//! the causal conv's history crossing every cut.

mod common;

use common::*;
use gpu_core::Gpu;
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn build(init: &std::collections::HashMap<String, Vec<f32>>, gdn_segment_tokens: Option<u32>, recompute: bool) -> Qwen35 {
    let opts = TrainOpts { gdn_segment_tokens, recompute, ..TrainOpts::default() };
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), lora_config(), 1, T as u32, init, &opts)
}

#[test]
fn segmented_gdn_layers_train_the_same_adapters_as_whole_sequence_layers() {
    let cfg = lora_config();
    let init = weights(&cfg, 0.3);
    // t = 12 runs as chunks of 4; segments of 4 rows cut the sequence in three
    // (the shortest legal cut is the conv history, 3 rows).
    let want = loss_and_grads(&build(&init, None, false));
    for recompute in [false, true] {
        for seg in [4, 8] {
            assert_same_training_signal(loss_and_grads(&build(&init, Some(seg), recompute)), want.clone(), 1e-5, 0.99999);
        }
    }
    assert!(want.1.iter().any(|(_, g)| g.iter().any(|x| *x != 0.0)));
}
