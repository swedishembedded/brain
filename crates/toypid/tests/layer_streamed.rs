// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in training every weight through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contracts: the transformer's per-layer tensors stream through a
//! one-layer device window with their f32 values and optimizer state on the host, and train exactly what the build
//! holding every layer in that state trains, which is what the plain f32 build trains on the device
//! (`model::streamed::check_layer_streaming`); a streamed run stopped and resumed through the fit loop ends exactly
//! where the uninterrupted one does (`model::trainprec::check_exact_resume`). One test in its own binary, the
//! contracts in turn, so nothing else allocates while the device bytes are compared.

use model::Model;
use toypid::data::init_weights;
use toypid::{Pid, PidConfig};

#[test]
fn toypid_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    let cfg = PidConfig { block_size: 16, n_layers: 3, d_model: 32, n_heads: 4, d_ff: 64, ..PidConfig::default_small() };
    let init = init_weights(&cfg, 3);
    let (b, t) = (2u32, 8u32);
    let x: Vec<u32> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let y: Vec<u32> = (0..b * t).map(|i| (i * 5 + 1) % cfg.u_bins).collect();
    let build = || Pid::new(cfg.clone(), b, t, &init);
    let step = |m: &Pid, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&x, &y);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("toypid", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-toypid-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || Pid::new(cfg.clone(), 1, 8, &init);
    let resumed = model::trainprec::check_exact_resume("toypid", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
