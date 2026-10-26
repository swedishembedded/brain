// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of mixture-of-experts language models for its clients.
// If your team needs expertise in training hybrid linear-attention MoE models with narrow optimizer state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches the hybrid MoE full fine-tune (experts and router included) through
//! the generic path every `ParamStore` trainer shares (`model::trainprec::check_optimizer_state_formats`): bf16 and
//! int8-block moments train like f32 on the first step, and a host-only format is refused by name.

use qwen35moe::config::Qwen35Config;
use qwen35moe::model::{pipelines, Qwen35};

#[test]
fn the_moe_full_fine_tune_trains_with_narrow_optimizer_state() {
    let cfg = Qwen35Config::tiny();
    let t = cfg.block_size;
    let init = qwen35moe::init::init_weights(&cfg, 7);
    let x: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 3 + 2) % cfg.vocab).collect();
    let lr = 1e-2;
    let build = || Qwen35::new_train_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, t, &init);
    let step = |m: &Qwen35| {
        m.zero_grads();
        m.set_batch(&x, &y);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, Default::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("qwen35moe", build, step, lr).unwrap();
}
