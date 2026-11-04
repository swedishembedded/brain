// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of transformer language models for its clients.
// If your team needs expertise in training language models with narrow optimizer state, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches GPT-2 through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments on the device and f32 or log8 moments in host memory
//! train like f32 on the first step, a run with host moments resumes exactly,
//! and a precision no optimizer runs is refused by name.

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

/// A run whose moments live in host memory resumes exactly: the fit loop commits the host optimizer beside the
/// resume file (`model::trainprec::check_resume_with_host_moments`).
#[test]
fn gpt2_resumes_exactly_with_host_moments() {
    let cfg = GptConfig { vocab: 16, block_size: 8, ..GptConfig::tiny() };
    let init = gpt2::init_weights(&cfg, 5);
    let scratch = std::env::temp_dir().join(format!("brain-gpt2-host-moments-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let build = || Gpt::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 1, 8, &init);
    model::trainprec::check_resume_with_host_moments("gpt2", build, 8, &scratch).unwrap();
    std::fs::remove_dir_all(&scratch).unwrap();
}
