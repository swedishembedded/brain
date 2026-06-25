// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synchronous one-token-per-`step` streaming pump.
//!
//! [`gpt::sample::generate`] runs the whole token loop and returns every token at
//! once. The HSM controller instead needs *one* token per `step()` call so it can
//! emit one `brain_text_chunk` per token under run-to-completion. [`StreamPump`]
//! is that refactor: it holds the growing context + config and advances exactly
//! one token per [`StreamPump::step`], decoding the token to its text delta via
//! the model's `itos` table. It returns `None` at EOS / `max_new`.

use crate::sample::{sample_logits, Rng};
use crate::{GenConfig, InferModel};

/// One synchronous streaming generation. Owns the prompt+generated context and a
/// seeded RNG; each [`step`](StreamPump::step) pumps one token.
pub struct StreamPump {
    ctx: Vec<u32>,
    block: usize,
    vocab: usize,
    generated: usize,
    cfg: GenConfig,
    rng: Rng,
    done: bool,
}

impl StreamPump {
    /// Seed a pump from a text `prompt` (encoded via the model's `itos`, falling
    /// back to byte values when no char vocab is present).
    pub fn new(model: &dyn InferModel, prompt: &str, cfg: GenConfig) -> StreamPump {
        let block = model.block_size() as usize;
        let vocab = model.vocab() as usize;
        let ctx = encode_prompt(model, prompt);
        let rng = Rng::new(cfg.seed);
        StreamPump { ctx, block, vocab, generated: 0, cfg, rng, done: false }
    }

    /// Advance one token. Returns `Some(text_delta)` for the newly generated
    /// token, or `None` once EOS / `max_new` is hit (terminal).
    pub fn step(&mut self, model: &dyn InferModel) -> Option<String> {
        if self.done || self.generated >= self.cfg.max_new {
            self.done = true;
            return None;
        }
        // Context window cropped to the block size.
        let window: Vec<u32> = if self.ctx.len() > self.block {
            self.ctx[self.ctx.len() - self.block..].to_vec()
        } else {
            self.ctx.clone()
        };
        let logits = model.logits_all(&window);
        let last = &logits[logits.len() - self.vocab..];
        let next = sample_logits(last, self.cfg.temperature, self.cfg.top_k, &mut self.rng);

        if Some(next) == self.cfg.eos {
            self.done = true;
            return None;
        }
        self.ctx.push(next);
        self.generated += 1;
        Some(decode_token(model, next))
    }

    /// Number of tokens generated so far.
    pub fn generated(&self) -> usize {
        self.generated
    }
}

/// Encode a prompt string into token ids. Char-vocab models map each char to its
/// `itos` index; otherwise tokens are raw byte values (matches [`FakeInferModel`]
/// and keeps untrained-model smoke tests trivial).
fn encode_prompt(model: &dyn InferModel, prompt: &str) -> Vec<u32> {
    match model.itos() {
        Some(itos) => prompt
            .chars()
            .filter_map(|c| itos.iter().position(|&x| x == c).map(|i| i as u32))
            .collect(),
        None => prompt.bytes().map(|b| b as u32).collect(),
    }
}

/// Decode a single token id to its text delta.
fn decode_token(model: &dyn InferModel, tok: u32) -> String {
    match model.itos() {
        Some(itos) => itos.get(tok as usize).copied().unwrap_or('?').to_string(),
        None => char::from_u32(tok).unwrap_or('?').to_string(),
    }
}
