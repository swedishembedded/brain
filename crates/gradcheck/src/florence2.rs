// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference gate for Florence-2's BART text encoder-decoder
//! training graph (`florence2::train::Florence2Trainer`).
//!
//! `Florence2Trainer` does not implement `model::Model` - its batch is
//! `(prompt_ids, decoder_ids, targets)` over a genuine encoder-decoder with
//! a text-only encoder input (no vision splice - DaViT is out of scope for
//! M6, see `florence2::train`'s own module doc), not `model::Batch`'s
//! single-stream or `Seq2Seq{src,tgt,labels}` shape (florence2's encoder
//! `src` is embeddings the caller already projected, not raw ids the model
//! embeds itself) - so the blanket `impl<M: model::Model> CheckModel for M`
//! does not apply. `Florence2Trainer` already exposes exactly `CheckModel`'s
//! surface directly (a REAL mean cross-entropy `loss()`, not a proxy `<r,y>`
//! linear trick like `check_clip`/`check_t5` need - florence2 has an actual
//! token-classification head), so this file is a direct `impl`, no wrapper
//! struct.

use std::collections::HashMap;

use florence2::text::{BartConfig, LoraCfg};
use florence2::train::{self, Florence2Trainer};

use crate::{directional_check, CheckModel, Report};

/// Local newtype: Rust's coherence rules reject `impl CheckModel for
/// Florence2Trainer` directly (a foreign type, against this crate's own
/// blanket `impl<M: model::Model> CheckModel for M`) - the same reason
/// `T5Harness`/`Timesfm3Harness`/`deepseekocr2::Harness` all wrap rather
/// than impl on the model type itself.
struct Harness(Florence2Trainer);

impl CheckModel for Harness {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.loss()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

/// `BartConfig::tiny()` sized down further for a cheap finite-difference
/// sweep, one fixed synthetic example: `t_prompt=3`, `max_t=3` (decoder
/// length), std-0.15 init (the same conditioning fix
/// `deepseek2/tests/gradcheck.rs`'s own `FIXTURE_INIT_STD` doc explains -
/// measured on THIS fixture too, not assumed from that one).
fn harness(lora: Option<LoraCfg>, seed: u64) -> Harness {
    let cfg = BartConfig::tiny();
    let (t_prompt, max_t) = (3u32, 3u32);
    let names = train::param_list(&cfg, t_prompt, max_t, lora);
    let init: HashMap<String, Vec<f32>> = train::init_weights(&names, 0.15, seed);
    let m = Florence2Trainer::new(gpu_core::testgpu::dev(train::PIPELINES), cfg, t_prompt, max_t, lora, &init);
    let prompt_ids: Vec<u32> = (0..t_prompt).map(|i| (i * 5 + 1) % cfg.vocab_size).collect();
    let decoder_ids: Vec<u32> = (0..max_t).map(|i| (i * 3 + 2) % cfg.vocab_size).collect();
    let targets: Vec<u32> = (0..max_t).map(|i| (i * 7 + 3) % cfg.vocab_size).collect();
    m.set_batch(&prompt_ids, &decoder_ids, &targets);
    Harness(m)
}

/// Full fine-tune: every base tensor trainable, nothing frozen.
pub fn check_florence2(seed: u64) -> Report {
    let m = harness(None, seed);
    directional_check(&m, 5e-3, 4, seed)
}

/// LoRA: base weights `Role::Frozen`, only `.lora_a`/`.lora_b` trainable -
/// `directional_check`'s own `param_names()` walk already restricts to
/// `ps.trainable`, so this is the SAME call over a different `ParamStore`.
pub fn check_florence2_lora(seed: u64) -> Report {
    let m = harness(Some(LoraCfg { rank: 2, alpha: 4.0 }), seed);
    directional_check(&m, 5e-3, 4, seed)
}
