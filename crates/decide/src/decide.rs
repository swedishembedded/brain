// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision model as one object: tokenizer, encoder and head.
//!
//! ```text
//! state  --tokenize--> windows -\
//!                                 pack --> encode once --> head --> scores
//! options --tokenize--> slots  -/                                     |
//!                                                    host softmax per question
//! ```
//!
//! **The state is encoded once per request, not once per question.** Every
//! question's options query the same encoding, so a request with ten questions
//! costs one encode plus ten cheap scorings rather than ten encodes. That is
//! the whole economic argument for this shape, and it is why the instructions
//! travel with the options.
//!
//! Two parameter stores, not one: the encoder arrives pretrained and the head
//! starts from noise, so they take different learning rates. Two optimizers
//! express that directly, with no per-tensor multiplier to keep in sync.

use std::collections::HashMap;

use data::tokenizer::Tokenizer;
use data::wordpiece::WordPiece;
use gpu_core::Gpu;

use crate::config::EncoderConfig;
use crate::head::Head;
use crate::kern;
use crate::loss::{decision_loss, LossConfig};
use crate::model::Encoder;
use crate::pack::{Packer, SEG_SLOT};
use crate::primitives::{Answer, Question};

/// How a model is sized. A call may use less of any of it.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Packed token rows: state windows plus every option's slot.
    pub cap_rows: u32,
    /// Options in one request, across all its questions.
    pub cap_slots: u32,
    /// The longest span, and so the state window size.
    pub max_span: u32,
    /// Tokens a window shares with the next, so a fact spanning a cut is still
    /// seen whole by one of them.
    pub overlap: u32,
}

impl Default for Limits {
    /// Sized for the shape this model is for: a few thousand tokens of state
    /// and a full 255-option question, with windows well inside the learned
    /// position table.
    fn default() -> Limits {
        Limits { cap_rows: 4096, cap_slots: 320, max_span: 256, overlap: 32 }
    }
}

pub struct Decide {
    pub enc: Encoder,
    pub head: Head,
    pub tok: WordPiece,
    pub cfg: EncoderConfig,
    pub limits: Limits,
    enc_opt: Option<optim::Optim>,
    head_opt: Option<optim::Optim>,
    step: u32,
}

/// One training example: a state, a question, and which of its options is
/// correct.
pub struct Example<'a> {
    pub state: &'a str,
    pub question: &'a Question,
    pub gold: usize,
}

impl Decide {
    /// Build on an existing device. `enc_init` is normally an imported
    /// checkpoint and `head_init` normally fresh noise.
    pub fn new_on(
        gpu: Gpu,
        cfg: EncoderConfig,
        tok: WordPiece,
        limits: Limits,
        enc_init: &HashMap<String, Vec<f32>>,
        head_init: &HashMap<String, Vec<f32>>,
        train: bool,
    ) -> Decide {
        let ids = kern::Ids::resolve(&gpu);
        // Every buffer this model dispatches against is allocated once at
        // build time and lives as long as the model, which is exactly the
        // shape the step cache is safe on: it keeps an entry's buffers alive,
        // so arming it on a handle that allocated per call would pin every
        // temporary ever used. Requests repeat shapes constantly (the same
        // option set, similar state lengths), and without this the step list
        // is rebuilt from scratch on every single call.
        gpu.enable_step_cache(8192);
        let enc = if train {
            Encoder::new_train_on(gpu.share(), cfg.clone(), limits.cap_rows, limits.max_span, enc_init)
        } else {
            Encoder::new_on(gpu.share(), cfg.clone(), limits.cap_rows, limits.max_span, enc_init)
        };
        let head = Head::new_on(gpu, cfg.clone(), limits.cap_rows, limits.cap_slots, head_init, train);
        Decide {
            enc,
            head,
            tok,
            cfg,
            limits,
            enc_opt: train.then(|| ids.optimizer()),
            head_opt: train.then(|| ids.optimizer()),
            step: 0,
            }
    }

    /// Tokenize and pack one request, then encode it and score every option.
    ///
    /// Returns the raw scores grouped per question, in the order the questions
    /// were supplied.
    pub fn score(&mut self, state: &str, questions: &[Question]) -> Result<Vec<Vec<f32>>, String> {
        for q in questions {
            q.validate()?;
        }
        let pad = self.tok.token_to_id("[PAD]").unwrap_or(0);
        let mut packer = Packer::new(&self.cfg, pad);
        let state_ids = self.tok.encode(state);
        if state_ids.is_empty() {
            return Err("the state tokenized to nothing".into());
        }
        packer.push_state(&state_ids, self.limits.max_span, self.limits.overlap);

        let mut arity = Vec::with_capacity(questions.len());
        let mut cls_rows = Vec::new();
        for q in questions {
            let slots = q.slots();
            arity.push(slots.len());
            for s in &slots {
                let ids = self.tok.encode(s);
                let ids = if ids.len() > self.limits.max_span as usize {
                    // A slot longer than a window would outrun the position
                    // table. Truncating keeps the [CLS] the head reads, which
                    // is the part that matters, and is reported by returning
                    // the shortened span rather than silently padding.
                    ids[..self.limits.max_span as usize].to_vec()
                } else {
                    ids
                };
                let (row0, _) = packer.push_slot(&ids);
                cls_rows.push(row0);
            }
        }
        let packed = packer.finish();
        if packed.ids.len() > self.limits.cap_rows as usize {
            return Err(format!(
                "request needs {} packed rows but this model was built for {} - raise Limits::cap_rows \
                 or send a shorter state",
                packed.ids.len(),
                self.limits.cap_rows
            ));
        }
        if cls_rows.len() > self.limits.cap_slots as usize {
            return Err(format!(
                "request has {} options but this model was built for {}",
                cls_rows.len(),
                self.limits.cap_slots
            ));
        }
        // The state's rows are everything before the first slot. With the
        // packer's alignment this is contiguous from row 0.
        let state_rows = packed
            .types
            .iter()
            .position(|&t| t == SEG_SLOT)
            .map(|i| i as u32)
            .unwrap_or(packed.ids.len() as u32);

        self.enc.set_batch(&packed.ids, &packed.types, &packed.spans);
        self.enc.forward();
        // MUST NOT BE REMOVED. The head holds a different `Gpu` handle to the
        // same device, and a submit on one handle is not ordered against a
        // submit on another. Without this the head dispatches against a hidden
        // buffer the encoder has not written yet, reads zeros, and the whole
        // model still trains and still answers - always wrongly. It cost a
        // below-chance accuracy run to find.
        self.enc.poll_wait();
        // Disjoint field borrows: the head is taken mutably while the
        // encoder's buffers are read.
        if self.enc.is_trainable() {
            self.head.set_call(self.enc.hidden_buf(), Some(self.enc.seed_buf()), state_rows, &cls_rows);
        } else {
            self.head.set_call(self.enc.hidden_buf(), None, state_rows, &cls_rows);
        }
        let flat = self.head.forward();

        let mut out = Vec::with_capacity(questions.len());
        let mut at = 0usize;
        for n in arity {
            out.push(flat[at..at + n].to_vec());
            at += n;
        }
        Ok(out)
    }

    /// Answer every question about one state.
    pub fn decide(&mut self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, String> {
        let scores = self.score(state, questions)?;
        Ok(questions.iter().zip(&scores).map(|(q, s)| q.answer(s)).collect())
    }

    /// One optimizer step on one example. Returns the loss.
    pub fn train_step(&mut self, ex: &Example, loss: &LossConfig, enc_lr: f32, head_lr: f32) -> Result<f32, String> {
        let scores = self.score(ex.state, std::slice::from_ref(ex.question))?;
        let (l, d_score) = decision_loss(&scores[0], ex.gold, loss);

        self.enc.zero_grads();
        self.head.zero_grads();
        // The head writes its hidden-state gradient straight into the
        // encoder's seed buffer, so the two halves need no copy between them.
        self.head.backward(self.enc.seed_buf(), &d_score);
        // The head and the encoder hold DIFFERENT handles to one device, and a
        // submit on one is not ordered against a submit on the other. The
        // encoder's reverse pass reads the seed buffer the head's reverse pass
        // writes, so it has to wait for it - see the forward's own wait.
        self.head.poll_wait();
        self.enc.backward_seeded();

        self.step += 1;
        let t = self.step;
        // Two rates, because the encoder arrives pretrained and the head does
        // not: one rate would either move the head too slowly to learn or move
        // the encoder fast enough to forget what it was imported for.
        // EACH HALF STEPS ITS OWN PARAMETERS ON ITS OWN HANDLE. Stepping the
        // head's weights through the encoder's handle raced the head's next
        // forward, which then read pre-update weights - the model evaluated as
        // a near-uniform distribution while the SAME weights, reloaded into a
        // fresh process, answered correctly. Same root cause as the forward's
        // own wait.
        if let Some(o) = &self.enc_opt {
            self.enc.adamw_step(o, t, enc_lr, 0.01, Some(1.0));
        }
        if let Some(o) = &self.head_opt {
            self.head.adamw_step(o, t, head_lr, 0.01, Some(1.0));
        }
        Ok(l)
    }

    /// Write the head's weights to a brain `.safetensors`.
    ///
    /// Only the head: the encoder is imported from a published checkpoint and
    /// re-importing it is free, so a run's artifact is the part that did not
    /// exist before it.
    pub fn save_head(&self, path: &str) -> Result<(), String> {
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
            self.head.weights().into_iter().map(|(n, v)| (n, vec![v.len() as u64], v)).collect();
        checkpoint::save(path, self.cfg.to_json(), &tensors);
        Ok(())
    }

    /// Steps taken so far - the AdamW time index, which a resumed run must
    /// carry so the bias correction stays continuous.
    pub fn steps_taken(&self) -> u32 {
        self.step
    }
}
