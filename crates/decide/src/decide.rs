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

/// Recorded dispatches kept per handle.
///
/// The encoder's tape is a few hundred dispatches per distinct packed layout,
/// and training walks many layouts (every message is a different length), so
/// this is sized to hold a working set of them rather than one. Entries cost
/// only their own bookkeeping here: every buffer they pin is a model buffer
/// that outlives the cache anyway.
const STEP_CACHE_ENTRIES: usize = 65536;

pub struct Decide {
    pub enc: Encoder,
    /// Device-side scratch the kept features are uploaded into. Grown to the
    /// largest call seen and reused, so learning from a rollout allocates
    /// once rather than per step.
    kept: Option<KeptBuf>,
    pub head: Head,
    pub tok: WordPiece,
    pub cfg: EncoderConfig,
    pub limits: Limits,
    enc_opt: Option<optim::Optim>,
    head_opt: Option<optim::Optim>,
    step: u32,
    frozen_encoder: bool,
    /// How many leading spans of the last call were state windows - what
    /// `state_embedding` has to pool over.
    last_windows: usize,
}

/// A request laid out for the device: the packed token stream, where each
/// option's `[CLS]` row sits, how many options each question has, and where
/// the state's rows end.
pub struct Request {
    pub packed: crate::pack::Packed,
    pub cls_rows: Vec<u32>,
    pub arity: Vec<usize>,
    pub state_rows: u32,
}

/// Device scratch for [`Features`] on their way back to the head.
struct KeptBuf {
    buf: gpu_core::DeviceBuffer,
    len_floats: usize,
}

/// The encoder's output for one call, kept so that a FROZEN encoder is not
/// run again to produce a number that cannot have changed.
///
/// PPO reads every rollout step once per epoch, so with four epochs the same
/// observation and the same options were tokenized and pushed through a
/// six-layer, 22M-parameter encoder five times over - once to act, four more
/// to learn from having acted - and the encoder is not being trained, so four
/// of those five runs computed a constant. Measured on this repository's DOOM
/// sample: the update was 80% of a training iteration's wall clock and the
/// game itself was 1.5%.
///
/// Only the rows the head actually reads are kept: the state's, which it
/// cross-attends over, and one `[CLS]` row per option. The option's other
/// tokens have done their work inside the encoder by then.
#[derive(Clone)]
pub struct Features {
    /// `(state_rows + n_slots) * hidden`, state rows first.
    hidden: Vec<f32>,
    state_rows: u32,
    n_slots: u32,
}

impl Features {
    /// Floats kept. What caching a rollout costs in memory.
    pub fn len(&self) -> usize {
        self.hidden.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hidden.is_empty()
    }
}

/// What [`Decide::repr_snapshot`] hands the confidence signals.
pub struct ReprSnapshot {
    /// One flattened hidden slab per encoder layer, state rows only - the
    /// `{h_l}` of the convergence signal.
    pub layers: Vec<Vec<f32>>,
    /// The encoder's mean-pooled sentence embedding of the state.
    pub reference: Vec<f32>,
    /// The head's post-LayerNorm output for the option that was chosen.
    pub decision: Vec<f32>,
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
        let enc = if train {
            Encoder::new_train_on(gpu.share(), cfg.clone(), limits.cap_rows, limits.max_span, enc_init)
        } else {
            Encoder::new_on(gpu.share(), cfg.clone(), limits.cap_rows, limits.max_span, enc_init)
        };
        let head = Head::new_on(gpu, cfg.clone(), limits.cap_rows, limits.cap_slots, head_init, train);
        // Every buffer this model dispatches against is allocated once at build
        // time and lives as long as the model, which is exactly the shape the
        // step cache is safe on: an entry keeps its buffers alive, so arming it
        // on a handle that allocates per call would pin every temporary it ever
        // used. Requests repeat shapes constantly - the same option set, the
        // same state length - and without this the whole tape is re-recorded
        // from scratch on every call, which for a packed encoder is several
        // hundred bind groups.
        //
        // ARMED PER HANDLE, AFTER BOTH HALVES EXIST. `Gpu::share` hands back an
        // independent handle with its own memo, so arming the handle that is
        // about to be moved into one half leaves the OTHER half uncached - and
        // the encoder is the half with 96% of the dispatches.
        enc.gpu().enable_step_cache(STEP_CACHE_ENTRIES);
        head.gpu().enable_step_cache(STEP_CACHE_ENTRIES);
        Decide {
            enc,
            head,
            tok,
            cfg,
            limits,
            enc_opt: train.then(|| ids.optimizer()),
            head_opt: train.then(|| ids.optimizer()),
            step: 0,
            frozen_encoder: false,
            last_windows: 1,
            kept: None,
        }
    }

    /// Tokenize and pack one request, then encode it and score every option.
    ///
    /// Returns the raw scores grouped per question, in the order the questions
    /// were supplied.
    pub fn score(&mut self, state: &str, questions: &[Question]) -> Result<Vec<Vec<f32>>, String> {
        let req = self.pack_request(state, questions)?;
        let flat = self.run_packed(&req);
        let mut out = Vec::with_capacity(req.arity.len());
        let mut at = 0usize;
        for &n in &req.arity {
            out.push(flat[at..at + n].to_vec());
            at += n;
        }
        Ok(out)
    }

    /// Tokenize and pack one request without touching the device.
    ///
    /// Separate from [`Decide::run_packed`] because the two halves have
    /// different costs and different failure modes: everything that can be
    /// rejected is rejected here, on the host, before a single dispatch is
    /// recorded.
    pub fn pack_request(&self, state: &str, questions: &[Question]) -> Result<Request, String> {
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

        Ok(Request { packed, cls_rows, arity, state_rows })
    }

    /// Encode a packed request and score every option, flat and in pack order.
    pub fn run_packed(&mut self, req: &Request) -> Vec<f32> {
        self.last_windows = req.packed.windows;
        self.enc.set_batch(&req.packed.ids, &req.packed.types, &req.packed.spans);
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
            self.head.set_call(self.enc.hidden_buf(), Some(self.enc.seed_buf()), req.state_rows, &req.cls_rows);
        } else {
            self.head.set_call(self.enc.hidden_buf(), None, req.state_rows, &req.cls_rows);
        }
        self.head.forward()
    }

    /// Score one question and keep the encoder's output for it.
    ///
    /// The scores are what the caller would have got from [`Self::score`];
    /// the [`Features`] are what makes learning from this step later cost the
    /// head alone. See [`Features`].
    pub fn score_keeping(
        &mut self,
        state: &str,
        question: &Question,
    ) -> Result<(Vec<f32>, Features), String> {
        let req = self.pack_request(state, std::slice::from_ref(question))?;
        let scores = self.run_packed(&req);
        let h = self.cfg.d_model as usize;
        let rows = req.packed.ids.len();
        let slab = self.enc.gpu().read(self.enc.hidden_buf(), rows * h);
        let mut hidden = Vec::with_capacity((req.state_rows as usize + req.cls_rows.len()) * h);
        hidden.extend_from_slice(&slab[..req.state_rows as usize * h]);
        for &r in &req.cls_rows {
            let at = r as usize * h;
            hidden.extend_from_slice(&slab[at..at + h]);
        }
        Ok((
            scores,
            Features {
                hidden,
                state_rows: req.state_rows,
                n_slots: req.cls_rows.len() as u32,
            },
        ))
    }

    /// One accumulation step from kept features, running the head alone.
    ///
    /// Exactly what [`Self::accumulate`] computes when the encoder is frozen,
    /// without the encoder: its output for this state and these options is
    /// already known and cannot have changed. Refuses to run on a TRAINABLE
    /// encoder, where the features would be stale by one update.
    pub fn accumulate_kept(
        &mut self,
        f: &Features,
        objective: impl FnOnce(&[f32]) -> (f32, Vec<f32>),
    ) -> Result<f32, String> {
        if !self.frozen_encoder {
            return Err("accumulate_kept needs a frozen encoder: a trainable one \
                        changes what the features would have been"
                .into());
        }
        if self.kept.as_ref().is_none_or(|b| b.len_floats < f.hidden.len()) {
            let cap = f.hidden.len().max(1);
            self.kept = Some(KeptBuf {
                buf: self.enc.gpu().buffer(
                    "kept_hidden",
                    (cap * 4) as u64,
                    gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
                ),
                len_floats: cap,
            });
        }
        let kept = self.kept.as_ref().expect("just allocated");
        self.enc.gpu().write_f32(&kept.buf, &f.hidden);
        self.enc.poll_wait();
        // The rows were compacted when they were kept: the state first, then
        // one row per option, so the slot rows are the ones after the state.
        let cls: Vec<u32> = (0..f.n_slots).map(|i| f.state_rows + i).collect();
        // The head still needs somewhere to put its hidden-state gradient,
        // even though a frozen encoder never reads it - the same arrangement
        // the text path uses, and the reason the encoder's reverse pass is
        // simply not run rather than not wired.
        let seed = self.enc.is_trainable().then(|| self.enc.seed_buf());
        self.head.set_call(&kept.buf, seed, f.state_rows, &cls);
        let scores = self.head.forward();
        let (l, d_score) = objective(&scores);
        assert_eq!(d_score.len(), scores.len(), "one score gradient per option");
        self.head.backward(self.enc.seed_buf(), &d_score);
        self.head.poll_wait();
        Ok(l)
    }

    /// The head's scores for kept features, forward only.
    ///
    /// [`Self::accumulate_kept`] without the reverse pass, for when the
    /// question is what a set of weights WOULD say rather than how to change
    /// them - reading a reference policy off a batch already collected, for
    /// instance, where running a backward pass would be both wasted work and
    /// a gradient nobody asked for.
    pub fn score_kept(&mut self, f: &Features) -> Result<Vec<f32>, String> {
        if !self.frozen_encoder {
            return Err("score_kept needs a frozen encoder: a trainable one \
                        changes what the features would have been"
                .into());
        }
        if self.kept.as_ref().is_none_or(|b| b.len_floats < f.hidden.len()) {
            let cap = f.hidden.len().max(1);
            self.kept = Some(KeptBuf {
                buf: self.enc.gpu().buffer(
                    "kept_hidden",
                    (cap * 4) as u64,
                    gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
                ),
                len_floats: cap,
            });
        }
        let kept = self.kept.as_ref().expect("just allocated");
        self.enc.gpu().write_f32(&kept.buf, &f.hidden);
        self.enc.poll_wait();
        let cls: Vec<u32> = (0..f.n_slots).map(|i| f.state_rows + i).collect();
        let seed = self.enc.is_trainable().then(|| self.enc.seed_buf());
        self.head.set_call(&kept.buf, seed, f.state_rows, &cls);
        Ok(self.head.forward())
    }

    /// Answer every question about one state.
    pub fn decide(&mut self, state: &str, questions: &[Question]) -> Result<Vec<Answer>, String> {
        let scores = self.score(state, questions)?;
        Ok(questions.iter().zip(&scores).map(|(q, s)| q.answer(s)).collect())
    }

    /// One optimizer step on one example. Returns the loss.
    pub fn train_step(&mut self, ex: &Example, loss: &LossConfig, enc_lr: f32, head_lr: f32) -> Result<f32, String> {
        self.train_step_with(ex.state, ex.question, enc_lr, head_lr, |scores| {
            decision_loss(scores, ex.gold, loss)
        })
    }

    /// One optimizer step under a CALLER-SUPPLIED objective.
    ///
    /// `objective` receives this call's raw option scores and returns
    /// `(loss, dL/d(score))`. Everything downstream of that - the head's
    /// reverse pass, the encoder's, and both optimizers - is identical
    /// whatever the objective is, which is what lets a policy gradient
    /// ([`crate::policy`]) reuse the whole datapath instead of forking it.
    ///
    /// The objective is host code and sees host floats, so it may do anything:
    /// sample an action, look up a return, clip a ratio.
    pub fn train_step_with(
        &mut self,
        state: &str,
        question: &Question,
        enc_lr: f32,
        head_lr: f32,
        objective: impl FnOnce(&[f32]) -> (f32, Vec<f32>),
    ) -> Result<f32, String> {
        self.zero_grads();
        let l = self.accumulate(state, question, objective)?;
        self.adamw(enc_lr, head_lr);
        Ok(l)
    }

    /// Clear both halves' parameter gradients.
    ///
    /// Public because a caller accumulating a MINIBATCH owns the cycle: zero
    /// once, [`Decide::accumulate`] over the batch, then [`Decide::adamw`] once.
    /// [`Decide::train_step_with`] is that cycle for a batch of one.
    pub fn zero_grads(&mut self) {
        if !self.frozen_encoder {
            self.enc.zero_grads();
        }
        self.head.zero_grads();
    }

    /// Forward and backward for ONE example, ACCUMULATING into the parameter
    /// gradients without stepping the optimizer.
    ///
    /// The building block of a minibatch. Parameter gradients accumulate by
    /// construction here, so a caller sums several examples and steps once -
    /// which is what every policy-gradient implementation does, and what a
    /// per-example step is not: a single transition's gradient is a very noisy
    /// estimate, and Adam applied to it directly chases the noise.
    pub fn accumulate(
        &mut self,
        state: &str,
        question: &Question,
        objective: impl FnOnce(&[f32]) -> (f32, Vec<f32>),
    ) -> Result<f32, String> {
        let scores = self.score(state, std::slice::from_ref(question))?;
        let (l, d_score) = objective(&scores[0]);
        assert_eq!(d_score.len(), scores[0].len(), "one score gradient per option");

        // The head writes its hidden-state gradient straight into the
        // encoder's seed buffer, so the two halves need no copy between them.
        self.head.backward(self.enc.seed_buf(), &d_score);
        // The head and the encoder hold DIFFERENT handles to one device, and a
        // submit on one is not ordered against a submit on the other. The
        // encoder's reverse pass reads the seed buffer the head's reverse pass
        // writes, so it has to wait for it - see the forward's own wait.
        self.head.poll_wait();
        // The encoder's reverse pass is roughly two thirds of a training step,
        // and a frozen encoder has no use for it: nothing downstream reads the
        // gradient it would compute. The head still writes into the seed
        // buffer because that is where its own reverse pass puts the
        // hidden-state gradient; it is simply never consumed.
        if !self.frozen_encoder {
            self.enc.backward_seeded();
        }
        Ok(l)
    }

    /// The encoder's mean-pooled sentence embedding of the state, for the call
    /// just made.
    ///
    /// One readback of the hidden states, no per-layer traffic - cheap enough
    /// for a rollout loop, unlike [`Decide::repr_snapshot`]. This is the fixed
    /// feature a critic is fitted on when the encoder is frozen.
    pub fn state_embedding(&self) -> Vec<f32> {
        let h = self.cfg.d_model as usize;
        let pooled = self.enc.pooled_mean();
        let spans = self.enc.spans();
        // EVERY state window, length-weighted - not just the first.
        //
        // This returned `pooled[..h]`, the first window alone. A conversation
        // averages nearly two windows here, so the embedding covered roughly
        // its first half and missed the closing turns - which on this task
        // carry almost all of the signal. A linear probe on it scored 0.569
        // where the same probe on a whole-conversation embedding scores much
        // higher, and nothing said so: a partial embedding is a perfectly
        // ordinary-looking vector.
        //
        // Windows overlap by design, so their shared tokens count twice in
        // this mean. That is a small bias toward the middle of a conversation
        // and far smaller than ignoring its second half.
        let windows = self.last_windows.max(1).min(spans.len());
        let total: f32 = spans[..windows].iter().map(|&(_, l)| l as f32).sum();
        if total <= 0.0 {
            return pooled[..h].to_vec();
        }
        let mut out = vec![0.0f32; h];
        for (w, &(_, len)) in spans[..windows].iter().enumerate() {
            let k = len as f32 / total;
            for (c, o) in out.iter_mut().enumerate() {
                *o += k * pooled[w * h + c];
            }
        }
        out
    }

    /// The representations the confidence signals of [`crate::routing`] are
    /// computed from, for the call just made.
    ///
    /// Reads every layer's hidden slab back off the device, so this is an
    /// inspection path: call it when routing a decision, never inside a
    /// training loop.
    pub fn repr_snapshot(&self, state_rows: usize, chosen_slot: usize) -> ReprSnapshot {
        let h = self.cfg.d_model as usize;
        let keep = state_rows * h;
        let layers: Vec<Vec<f32>> = (0..self.cfg.n_layers as usize)
            .map(|l| {
                let mut v = self.enc.layer_out(l);
                // State rows only. The option slots are packed into the same
                // buffer, and their spread says nothing about whether the
                // stack settled on THIS conversation.
                v.truncate(keep);
                v
            })
            .collect();
        // The first span is the state's first window - the sentence embedding
        // the projection in equation (1) is compared against.
        let reference = self.enc.pooled_mean()[..h].to_vec();
        ReprSnapshot { layers, reference, decision: self.head.decision_repr(chosen_slot) }
    }

    /// Advance both halves by one AdamW update.
    ///
    /// Two rates, because the encoder arrives pretrained and the head does
    /// not: one rate would either move the head too slowly to learn or move
    /// the encoder fast enough to forget what it was imported for.
    ///
    /// EACH HALF STEPS ITS OWN PARAMETERS ON ITS OWN HANDLE. Stepping the
    /// head's weights through the encoder's handle raced the head's next
    /// forward, which then read pre-update weights - the model evaluated as a
    /// near-uniform distribution while the SAME weights, reloaded into a fresh
    /// process, answered correctly. Same root cause as the forward's own wait.
    pub fn adamw(&mut self, enc_lr: f32, head_lr: f32) {
        self.adamw_scaled(enc_lr, head_lr, 1.0)
    }

    /// [`Decide::adamw`] with the accumulated gradient scaled by `scale`.
    ///
    /// A minibatch of `n` examples accumulates `n` gradients, so `1.0 / n`
    /// turns the sum into the mean and keeps one learning rate meaningful
    /// across batch sizes.
    pub fn adamw_scaled(&mut self, enc_lr: f32, head_lr: f32, scale: f32) {
        self.step += 1;
        let t = self.step;
        if let Some(o) = &self.enc_opt {
            if !self.frozen_encoder {
                self.enc.adamw_step_scaled(o, t, enc_lr, 0.01, Some(1.0), scale);
            }
        }
        if let Some(o) = &self.head_opt {
            self.head.adamw_step_scaled(o, t, head_lr, 0.01, Some(1.0), scale);
        }
    }

    /// Train the head only, leaving the imported encoder exactly as it was.
    ///
    /// Two reasons a caller wants this, and they point the same way:
    ///
    /// * **Stability.** A reinforcement signal is far noisier than a labelled
    ///   one, and moving 22M pretrained parameters on a few hundred
    ///   high-variance gradients is a good way to destroy the language
    ///   understanding that made the option text readable in the first place.
    /// * **Speed.** The encoder's reverse pass is about two thirds of a
    ///   training step. Freezing it skips that pass AND its optimizer update,
    ///   so a step costs roughly what a forward does.
    ///
    /// The forward is unchanged, so the frozen encoder still supplies the
    /// representation; only the head learns from it.
    pub fn set_encoder_frozen(&mut self, frozen: bool) {
        self.frozen_encoder = frozen;
    }

    /// Whether the encoder is being held fixed.
    pub fn encoder_frozen(&self) -> bool {
        self.frozen_encoder
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

    /// Whether the encoder's weights are held still. A caller that wants to
    /// keep the encoder's output and reuse it has to know.
    pub fn encoder_is_frozen(&self) -> bool {
        self.frozen_encoder
    }

    /// Every head parameter, for snapshotting mid-run.
    pub fn head_weights(&self) -> Vec<(String, Vec<f32>)> {
        self.head.weights()
    }

    /// Put a snapshot back. See [`decide::head::Head::set_weights`].
    pub fn set_head_weights(&self, w: &[(String, Vec<f32>)]) {
        self.head.set_weights(w);
    }

    /// Steps taken so far - the AdamW time index, which a resumed run must
    /// carry so the bias correction stays continuous.
    pub fn steps_taken(&self) -> u32 {
        self.step
    }
}
