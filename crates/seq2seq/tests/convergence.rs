// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Convergence integration test for the encoder-decoder Transformer.
//!
//! This is a *learnability* guard, mirroring `gpt/tests/convergence.rs`: it
//! trains the real seq2seq engine end-to-end (forward + full backprop through the
//! bidirectional encoder, the causal + cross attention decoder, and AdamW, on
//! whatever device `BRAIN_DEVICE` selects) on a tiny synthetic task a correct
//! implementation *must* learn, and asserts the loss drops far below the task's
//! marginal-entropy floor.
//!
//! Task — COPY: the encoder reads a random source string `s` of length `L` over a
//! small alphabet; the decoder is teacher-forced with `[BOS, s_0, …, s_{L-2}]` and
//! must predict `[s_0, …, s_{L-1}]`. Because the decoder's causal self-attention
//! cannot see `s_i` before predicting it, the only way to drive the loss to ~0 is
//! to learn a cross-attention copy circuit that reads position `i` from the
//! encoder memory. This exercises exactly the new architecture (encoder
//! bidirectional self-attn + decoder cross-attn) end to end — the finite-
//! difference `check_seq2seq` validates per-op gradients; this validates that the
//! whole loop actually learns.
//!
//! The training loop here mirrors `model::train::fit`'s control flow (cosine-LR
//! warmup, AdamW with global-norm clip), but feeds `Batch::Seq2Seq` directly:
//! `fit` only drives single-stream `Batch::Lm` from a token `.bin` dataset, which
//! cannot express a separate source/target stream. The threshold is calibrated
//! against measured CPU (Cranelift JIT) runs with a wide margin.
//!
//! Skipped when `MOE_SKIP_GPU_TESTS` is set (same gate as the rest of the suite).

use data::rng::Rng;
use model::{cosine_lr, Batch, FitOpts, Model};
use seq2seq::{Seq2Seq, Seq2SeqConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// Alphabet of `n_sym` symbols, plus a dedicated BOS id = `n_sym`. Vocab is
/// `n_sym + 1`.
struct Task {
    n_sym: u32,
    len: u32,
    bos: u32,
}

impl Task {
    fn vocab(&self) -> u32 {
        self.n_sym + 1
    }

    /// Sample one batch of `b` copy examples. Returns flat (src, tgt, labels).
    fn batch(&self, b: u32, rng: &mut Rng) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let l = self.len as usize;
        let mut src = Vec::with_capacity(b as usize * l);
        let mut tgt = Vec::with_capacity(b as usize * l);
        let mut labels = Vec::with_capacity(b as usize * l);
        for _ in 0..b {
            let s: Vec<u32> = (0..l)
                .map(|_| rng.gen_range_inclusive(0, self.n_sym as i64 - 1) as u32)
                .collect();
            // encoder input: the source string.
            src.extend_from_slice(&s);
            // decoder input: BOS then s shifted right (teacher forcing).
            tgt.push(self.bos);
            tgt.extend_from_slice(&s[..l - 1]);
            // labels: predict the full source (no masking — every position counts).
            labels.extend_from_slice(&s);
        }
        (src, tgt, labels)
    }
}

#[test]
fn engine_learns_copy_via_cross_attention() {
    if skip() {
        return;
    }

    let task = Task { n_sym: 6, len: 5, bos: 6 };
    let cfg = Seq2SeqConfig {
        vocab: task.vocab(),
        block_size: task.len,     // decoder length
        src_block_size: task.len, // encoder length
        n_enc: 2,
        n_dec: 2,
        d_model: 32,
        n_heads: 4,
        d_ff: 128,
    };
    let b = 32u32;
    let steps = 400u32;

    let init = Seq2Seq::init_weights(&cfg, 1234);
    let model = Seq2Seq::new_on(gpu_core::testgpu::dev(seq2seq::model::PIPELINES), cfg, b, task.len, &init);

    let opts = FitOpts {
        steps,
        lr: 3e-3,
        min_lr: 3e-4,
        warmup: 20,
        decay_iters: steps * 2, // don't crater the LR before the model can fit
        weight_decay: 0.1,
        grad_clip: 1.0,
        ..Default::default()
    };

    let mut rng = Rng::new(7);
    // initial loss (averaged over a few batches)
    let mut init_loss = 0.0;
    for _ in 0..5 {
        let (s, t, y) = task.batch(b, &mut rng);
        Model::set_batch(&model, Batch::Seq2Seq { src: &s, tgt: &t, labels: &y });
        init_loss += model.forward();
    }
    init_loss /= 5.0;

    let mut last = init_loss;
    for step in 0..steps {
        let lr = cosine_lr(step, &opts);
        let (s, t, y) = task.batch(b, &mut rng);
        Model::set_batch(&model, Batch::Seq2Seq { src: &s, tgt: &t, labels: &y });
        model.zero_grads();
        last = model.forward();
        model.backward();
        model.adamw_step(step + 1, lr, opts.weight_decay, Some(opts.grad_clip), 1.0);
        model.poll_wait();
    }

    println!("seq2seq copy: init {init_loss:.4} -> final {last:.4} (marginal ln(6) ~= 1.792)");

    // Marginal entropy of a uniform symbol is ln(6) ~= 1.792 — what a model with
    // no copy circuit (decoder can't see s_i) is stuck at. A correct encoder-
    // decoder with working cross-attention drives this far below the floor
    // (measured ~0.05 on the CPU backend); assert a wide-margin threshold.
    assert!(
        last < 0.5,
        "seq2seq copy failed to converge: {init_loss:.4} -> {last:.4} \
         (expected < 0.5, marginal floor ln(6) ~= 1.792)"
    );
}
