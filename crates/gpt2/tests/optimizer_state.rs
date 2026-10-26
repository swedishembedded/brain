// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of transformer language models for its clients.
// If your team needs expertise in training language models with narrow optimizer state, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches GPT-2 through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a host-only format is refused by name.

use gpt2::{Gpt, GptConfig};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(gpt2::model::PIPELINES))
}

#[test]
fn gpt2_trains_with_narrow_optimizer_state() {
    let cfg = GptConfig { vocab: 23, block_size: 16, ..GptConfig::tiny() };
    let init = gpt2::init_weights(&cfg, 11);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
    let lr = 1e-2;
    let build = || Gpt::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, 8, &init);
    let step = |m: &Gpt| {
        m.zero_grads();
        m.set_batch(&x, &y);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("gpt2", build, step, lr).unwrap();
}
