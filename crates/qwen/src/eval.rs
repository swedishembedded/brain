// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Held-out chat-sample scoring: teacher-forced masked loss + token
//! accuracy over [`data::chat::ChatSample`], for a base checkpoint alone or
//! with a named LoRA adapter folded in. This is Gate B of the Definition of
//! Done's "a way to validate that model has learned ideas from the
//! dataset" -- unlike `crates/qwen/tests/lora_learning_gate.rs` (Gate A,
//! synthetic, no checkpoint needed), this runs against a REAL base
//! checkpoint and a REAL bench-exported dataset, so it lives behind
//! `brain qwen eval` rather than as an always-on test.
//!
//! Loss/accuracy are computed ONLY over positions the sample itself marks
//! trainable (`ChatSample::encode`'s mask) -- prompt/context tokens the
//! model was never asked to predict never count, matching exactly what
//! `qwen::finetune::finetune` supervises during training.

use std::collections::HashMap;

use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;

use crate::config::QwenConfig;
use crate::model::Qwen;

/// Aggregate score over a held-out set: `loss` is mean per-token
/// cross-entropy (NaN if every sample was skipped), `token_accuracy` is the
/// fraction of trainable positions where greedy argmax matched the true
/// next token, `samples`/`skipped` account for every input sample so a
/// caller can tell "scored 0 out of 0" from "scored 0 out of 40".
#[derive(Debug, Clone, Copy)]
pub struct ChatScore {
    pub loss: f32,
    pub token_accuracy: f64,
    pub positions: usize,
    pub samples: usize,
    pub skipped: usize,
}

/// Build a servable [`Qwen`] from `weights`, optionally folding a LoRA
/// `adapter` (an adapter-only safetensors file, `qwen::lora::save_adapter`'s
/// output) into the base tensors first -- the same zero-inference-overhead
/// path a resident uses to serve a named adapter, so scoring an adapter
/// exercises exactly what serving it would do.
fn load_scored_model(weights: &str, adapter: Option<&str>, t: u32) -> Qwen {
    match adapter {
        None => Qwen::load_inference(weights, 1, t),
        Some(a) => {
            let c = checkpoint::load(weights);
            let mut tensors: HashMap<String, Vec<f32>> = c.by_role("");
            let mut cfg = QwenConfig::from_json(&c.header["config"]);
            crate::lora::fold_adapter_into(&mut tensors, a).expect("fold adapter into base tensors");
            // Folded: the delta is already baked into the base tensors, so
            // this Qwen has no separate lora_a/lora_b params to build.
            cfg.lora = None;
            cfg.block_size = t;
            Qwen::new(cfg, 1, t, &tensors)
        }
    }
}

fn argmax(s: &[f32]) -> u32 {
    let mut bi = 0usize;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi as u32
}

/// Score `weights` (optionally with `adapter` folded in) against `samples`,
/// teacher-forced: for each sample's `(ids, mask)`, position `i` predicts
/// `ids[i+1]` from context `ids[..=i]`, counted only where `mask[i+1]` is
/// `true` -- the exact convention `data::loader::TokenDataset` uses during
/// training (`mask[start+1+t]` gates `y[t] = data[start+1+t]`), so this
/// scores exactly what training supervised, nothing else. A sample that
/// fails to encode (see `ChatSample::encode`'s prefix-stability doc) or
/// whose length exceeds `block` is skipped, not silently dropped from the
/// count -- see [`ChatScore::skipped`].
pub fn score_chat(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, samples: &[ChatSample], block: u32) -> ChatScore {
    let model = load_scored_model(weights, adapter, block);
    let vocab = model.cfg.vocab as usize;
    let cap = model.ctx_len();

    let mut total_nll = 0.0f64;
    let mut positions = 0usize;
    let mut correct = 0usize;
    let mut skipped = 0usize;

    for s in samples {
        let Ok((ids, mask)) = s.encode(tok, tmpl) else {
            skipped += 1;
            continue;
        };
        if ids.len() < 2 || ids.len() > cap {
            skipped += 1;
            continue;
        }
        let logits = model.logits_all(&ids);
        for i in 0..ids.len() - 1 {
            if !mask[i + 1] {
                continue;
            }
            let target = ids[i + 1] as usize;
            let row = &logits[i * vocab..(i + 1) * vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&v| (v - max).exp()).sum();
            let log_prob = row[target] - max - sum_exp.ln();
            total_nll -= log_prob as f64;
            positions += 1;
            if argmax(row) as usize == target {
                correct += 1;
            }
        }
    }

    ChatScore {
        loss: if positions > 0 { (total_nll / positions as f64) as f32 } else { f32::NAN },
        token_accuracy: if positions > 0 { correct as f64 / positions as f64 } else { 0.0 },
        positions,
        samples: samples.len() - skipped,
        skipped,
    }
}
