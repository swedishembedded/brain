// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded DoRA and LoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A DoRA adapter over a Qwen3.5 base held in 4-bit NormalFloat or block-scaled FP8. DoRA rescales
//! every output channel by the norm of its adapted weight row, so it reads the frozen base through the
//! shared bounded tile decode (never an fp32 copy of the whole weight). The only approximation is the
//! weight rounding: an fp32 build over the values the packed base holds computes the same loss and
//! adapter gradients, and the adapter still learns.

mod common;

use std::collections::HashMap;

use common::*;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::adapter::DeviceKind;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

const BASES: [Dtype; 2] = [Dtype::NF4, Dtype::F8E4M3];

fn cfg() -> Qwen35Config {
    // `tiny_i8` keeps every contraction dimension a multiple of the quantisation group.
    Qwen35Config { block_size: T as u32, lora: Some(lora_cfg(3, 6.0).with_kind(DeviceKind::Dora)), ..Qwen35Config::tiny_i8() }
}

fn build(cfg: &Qwen35Config, init: &HashMap<String, Vec<f32>>, base: Dtype) -> Qwen35 {
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg.clone(), 1, T as u32, init, &opts(base))
}

#[test]
fn a_dora_adapter_over_a_packed_base_trains_what_an_fp32_build_over_its_decoded_values_does() {
    for base in BASES {
        let cfg = cfg();
        let init = weights(&cfg, 0.3);
        let packed = build(&cfg, &init, base);
        assert_eq!(packed.linear_dtype("blocks.0.mlp.gate.weight"), Some(base), "the base is held as asked, not widened");
        // What the packed base really holds, as the fp32 base of the reference.
        let mut held = init.clone();
        for name in cfg.param_list().into_iter().map(|(n, _)| n).filter(|n| packed.linear_dtype(n).is_some()) {
            held.insert(name.clone(), packed.read_weight(&name));
        }
        let want = loss_and_grads(&build(&cfg, &held, Dtype::F32));
        assert_same_training_signal(loss_and_grads(&packed), want, 1e-4, 0.9999);
    }
}

#[test]
fn a_dora_adapter_over_a_packed_base_still_overfits_a_fixed_batch() {
    for base in BASES {
        let cfg = cfg();
        let init = weights(&cfg, 0.0);
        let m = build(&cfg, &init, base);
        m.set_batch(&TOKENS, &TARGETS);
        let before = m.forward();
        for step in 1..=120 {
            m.zero_grads();
            m.forward();
            m.backward();
            m.adamw_step(step, 3e-2, 0.0, Default::default(), Some(1.0), 1.0);
            m.poll_wait();
        }
        let after = m.forward();
        assert!(after < before * 0.8, "{base:?}: DoRA over the packed base did not learn: {before} -> {after}");
    }
}
