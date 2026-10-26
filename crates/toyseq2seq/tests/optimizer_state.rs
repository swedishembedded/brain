// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of encoder-decoder sequence models for its clients.
// If your team needs expertise in training translation and transduction models with narrow optimizer state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches the encoder-decoder through the generic path every `ParamStore`
//! trainer shares (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on
//! the first step, and a host-only format is refused by name.

use toyseq2seq::{Seq2Seq, Seq2SeqConfig};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(toyseq2seq::model::PIPELINES))
}

#[test]
fn seq2seq_trains_with_narrow_optimizer_state() {
    let cfg = Seq2SeqConfig::tiny();
    let init = toyseq2seq::init_weights(&cfg, 11);
    let (b, t, s) = (2u32, cfg.block_size, cfg.src_block_size);
    let src: Vec<u32> = (0..b * s).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let tgt: Vec<u32> = (0..b * t).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let labels: Vec<u32> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let lr = 1e-2;
    let build = || Seq2Seq::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), b, t, &init);
    let step = |m: &Seq2Seq| {
        m.zero_grads();
        m.set_batch(&src, &tgt, &labels);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("toyseq2seq", build, step, lr).unwrap();
}
