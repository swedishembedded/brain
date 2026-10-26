// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in training every weight through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contract (`model::streamed::check_layer_streaming`): lfm2's
//! per-layer tensors (convolution and attention layers alike) stream through a one-layer device window with their f32
//! values and optimizer state on the host, and train exactly what the build holding every layer in that state trains,
//! which is what the plain f32 build trains on the device; a streamed run stopped and resumed through the fit loop
//! ends exactly where the uninterrupted one does (`model::trainprec::check_exact_resume`). One test in its own binary,
//! the contracts in turn, so nothing else allocates while the device bytes are compared.

use lfm2::config::{LayerType, LfmConfig};
use lfm2::model::{Lfm, IGNORE};
use model::Model;

#[test]
fn lfm2_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    let mut cfg = LfmConfig::tiny();
    cfg.layer_types = vec![LayerType::Conv, LayerType::Attention, LayerType::Conv, LayerType::Attention];
    let init = lfm2::init::init_weights(&cfg, 11);
    let (b, t) = (2u32, 8u32);
    let x: Vec<u32> = (0..b * t).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 5 + 2) % cfg.vocab).collect();
    let build = || Lfm::new_train(cfg.clone(), b, t, &init);
    let step = |m: &Lfm, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("lfm2", build, step).unwrap();

    // The long-context head scatters the input gradient of the supervised rows only, so each backward pass of the
    // update must start from a zero one.
    let sparse: Vec<u32> = y.iter().enumerate().map(|(i, &v)| if i % 3 == 0 { v } else { IGNORE }).collect();
    let gathered = || Lfm::new_train_chunked(cfg.clone(), b, t, &init, 1 << 20, b * t / 2);
    let gathered_step = |m: &Lfm, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &sparse);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("lfm2 (gathered head)", gathered, gathered_step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-lfm2-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || Lfm::new_train(cfg.clone(), 1, 8, &init);
    let resumed = model::trainprec::check_exact_resume("lfm2", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
