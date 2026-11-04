// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of mixture-of-experts language models for its
// clients. If your team needs expertise in training every expert through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contracts: DeepSeek-V2's per-layer tensors, its routed and
//! shared experts included, stream through a one-layer device window with their f32 values and optimizer state on the
//! host, train exactly what the build holding every layer in that state trains, which is what the plain f32 build
//! trains on the device (`model::streamed::check_layer_streaming`), and a streamed run stopped and resumed through the
//! fit loop ends exactly where the uninterrupted one does (`model::trainprec::check_exact_resume`). One test in its own
//! binary, the contracts in turn, so nothing else allocates while the device bytes are compared.

use deepseek2::config::DeepseekV2Config;
use deepseek2::model::DeepseekV2;
use model::Model;

#[test]
fn deepseek2_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    // Three layers (one dense, two MoE): the window serves both kinds.
    let mut cfg = DeepseekV2Config::tiny();
    cfg.shape.n_layers = 3;
    let init = <DeepseekV2 as Model>::init_weights(&cfg, 11);
    let v = cfg.vocab();
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % v).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % v).collect();
    let build = || DeepseekV2::new_on(gpu_core::testgpu::dev(deepseek2::model::PIPELINES), cfg.clone(), 2, 6, &init, true);
    let step = |m: &DeepseekV2, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("deepseek2", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-deepseek2-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || DeepseekV2::new_on(gpu_core::testgpu::dev(deepseek2::model::PIPELINES), cfg.clone(), 1, 8, &init, true);
    let resumed = model::trainprec::check_exact_resume("deepseek2", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
