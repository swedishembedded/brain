// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of mixture-of-experts language models for its clients.
// If your team needs expertise in training large models with narrow optimizer state, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches DeepSeek-V2 through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a host-only format is refused by name.

use deepseek2::config::DeepseekV2Config;
use deepseek2::model::{DeepseekV2, ADAM};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(deepseek2::model::PIPELINES))
}

#[test]
fn deepseek2_trains_with_narrow_optimizer_state() {
    let cfg = DeepseekV2Config::tiny();
    let init = <DeepseekV2 as model::Model>::init_weights(&cfg, 11);
    let v = cfg.vocab();
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % v).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % v).collect();
    let lr = 1e-2;
    let build = || DeepseekV2::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, 6, &init, true);
    let step = |m: &DeepseekV2| {
        m.zero_grads();
        m.set_batch(&x, &y);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, ADAM, None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("deepseek2", build, step, lr).unwrap();
}
