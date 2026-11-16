// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models with gradient accumulation for its
// clients. If your team needs expertise in training stacks whose micro-batches add up to one step then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Gradient accumulation adds every micro-batch's gradient: a chat full fine-tune runs `grad_accum` single-record
//! micro-batches into one optimizer step, so a backward that kept only its own gradient trains on one record a
//! step whatever `grad_accum` says.

use model::Model;
use ouro::config::OuroConfig;
use ouro::init::init_weights;
use ouro::model::Ouro;

#[test]
fn a_second_backward_adds_its_gradient() {
    // Short rows, and rows long enough for the training attention's flash kernels and the two-stage gain gradient.
    for (b, t) in [(2u32, 8u32), (2, 1024)] {
        accumulates(b, t);
    }
}

fn accumulates(b: u32, t: u32) {
    // Heads of 64, a width the device's tensor-core flash attention takes.
    let cfg = OuroConfig { block_size: t, d_model: 128, n_heads: 2, n_kv_heads: 2, head_dim: 64, ..OuroConfig::tiny() };
    let w = init_weights(&cfg, 3);
    let m = Ouro::new_train(cfg.clone(), b, t, &w);
    let tokens: Vec<u32> = (0..b * t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    // Half the positions supervised, as in a chat record.
    let targets: Vec<u32> = (0..b * t).map(|i| if i % 2 == 0 { (i * 7 + 8) % cfg.vocab } else { ouro::model::IGNORE }).collect();
    m.set_batch(model::Batch::Lm { tokens: &tokens, targets: &targets });
    model::conformance_grad::backwards_accumulate(&m).unwrap_or_else(|e| panic!("{b} x {t}: {e}"));
}
