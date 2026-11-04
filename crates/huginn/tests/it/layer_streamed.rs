// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in training every weight through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning of a recurrent-depth model, through the shared contracts: the prelude, core and coda
//! blocks stream through a one-layer device window, each core block loaded at every recurrence step it runs in, its
//! gradient carried on the host between those visits and the block updated once the backward is complete. That trains
//! exactly what the build holding every layer in that state trains and what the plain f32 build trains on the device
//! (`model::streamed::check_layer_streaming`); a streamed run stopped and resumed through the fit loop ends exactly
//! where the uninterrupted one does (`model::trainprec::check_exact_resume`). One test in its own binary, the
//! contracts in turn, so nothing else allocates while the device bytes are compared.

use huginn::model::Huginn;
use huginn::HuginnConfig;
use model::{Batch, Model};

#[test]
fn huginn_streams_its_blocks_through_a_one_layer_window_and_resumes_exactly() {
    let cfg = HuginnConfig::tiny();
    let init = huginn::init::init_weights(&cfg, 3);
    let v = cfg.vocab;
    let x: Vec<u32> = (0..8).map(|i| (i * 5 + 2) % v).collect();
    let y: Vec<u32> = (0..8).map(|i| (i * 5 + 3) % v).collect();
    let build = || Huginn::new_train(cfg.clone(), 2, 4, &init);
    let step = |m: &Huginn, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        let loss = Model::forward(m);
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("huginn", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-huginn-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumable = || Huginn::new_train(cfg.clone(), 1, 8, &init);
    let resumed = model::trainprec::check_exact_resume("huginn", resumable, 8, &scratch, "weights=f32,window=1");
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}
