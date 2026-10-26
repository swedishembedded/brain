// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use model::Model;
use toyseq2seq::{Seq2Seq, Seq2SeqConfig};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(toyseq2seq::model::PIPELINES))
}

/// A build that streams its encoder and decoder layers from construction never holds them all on the device, and
/// trains what the build streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn toyseq2seq_builds_its_layers_straight_into_the_window() {
    let cfg = Seq2SeqConfig { n_enc: 2, n_dec: 2, ..Seq2SeqConfig::tiny() };
    let init = toyseq2seq::init_weights(&cfg, 11);
    let (b, t, s) = (2u32, cfg.block_size, cfg.src_block_size);
    let src: Vec<u32> = (0..b * s).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let tgt: Vec<u32> = (0..b * t).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let labels: Vec<u32> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Seq2Seq::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), b, t, &init);
    let build_streamed = || Seq2Seq::new_resident_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), b, t, &init, &resident);
    let step = |m: &Seq2Seq, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&src, &tgt, &labels);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("toyseq2seq", build, build_streamed, step).unwrap();
}
