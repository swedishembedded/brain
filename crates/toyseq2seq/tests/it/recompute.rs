// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of encoder-decoder language models for its clients. If
// your team needs expertise in training long sequences within a fixed device budget, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Activation recompute through the shared contracts: a build that keeps only the residual between layers of the
//! encoder and of the decoder and replays each layer (the decoder's cross-attention to the encoder memory included)
//! before its backward computes the loss and gradients of the build that keeps every layer, bit for bit, on fewer
//! device bytes (`model::saved::check_recompute`), and holding those residuals in bf16 or FP8 trains to the format's
//! precision on fewer still (`model::saved::check_quantized_boundaries`). One test in its own binary, the contracts
//! in turn, so nothing else allocates while the device bytes are compared.

use toyseq2seq::{Seq2Seq, Seq2SeqConfig};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(toyseq2seq::model::PIPELINES))
}

#[test]
fn toyseq2seq_recomputes_every_layer_and_narrows_its_boundaries() {
    // Six encoder and six decoder layers: more boundaries than each narrow store's two f32 working buffers.
    let cfg = Seq2SeqConfig { vocab: 32, block_size: 16, src_block_size: 16, n_enc: 6, n_dec: 6, d_model: 32, n_heads: 4, d_ff: 64 };
    let init = toyseq2seq::init_weights(&cfg, 11);
    let (b, t, s) = (2u32, cfg.block_size, cfg.src_block_size);
    let src: Vec<u32> = (0..b * s).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let tgt: Vec<u32> = (0..b * t).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let labels: Vec<u32> = (0..b * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let build = || Seq2Seq::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), b, t, &init);
    let step = |m: &Seq2Seq| {
        m.zero_grads();
        m.set_batch(&src, &tgt, &labels);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_recompute("toyseq2seq", build, step).unwrap();
    model::saved::check_quantized_boundaries("toyseq2seq", build, step).unwrap();
}
