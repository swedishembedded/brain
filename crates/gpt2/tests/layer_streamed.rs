// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in training every weight through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contract (`model::streamed::check_layer_streaming`): gpt2's
//! per-layer tensors stream through a one-layer device window with their f32 values and optimizer state on the host,
//! and train exactly what the build holding every layer in that state trains, which is what the plain f32 build
//! trains on the device; a streamed run stopped and resumed through the fit loop ends exactly where the
//! uninterrupted one does (`model::trainprec::check_exact_resume`). One test in its own binary, the contracts in turn,
//! so nothing else allocates while the device bytes are compared.

use gpt2::{Gpt, GptConfig};
use model::Model;

#[test]
fn gpt2_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    let cfg = GptConfig { vocab: 23, block_size: 16, n_layers: 3, ..GptConfig::tiny() };
    let init = gpt2::init_weights(&cfg, 11);
    let x: Vec<u32> = (0..16).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..16).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let build = || Gpt::new_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), cfg.clone(), 2, 8, &init);
    let step = |m: &Gpt, t: u32, clip: Option<f32>| {
        m.zero_grads();
        Gpt::set_batch(m, &x, &y);
        let loss = Gpt::forward(m);
        Gpt::backward(m);
        Model::adamw_step(m, t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("gpt2", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-gpt2-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || Gpt::new_on(gpu_core::testgpu::dev(gpt2::model::PIPELINES), cfg.clone(), 1, 8, &init);
    let resumed = model::trainprec::check_exact_resume("gpt2", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
