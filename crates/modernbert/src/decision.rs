// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`LayaDecision`] - the trunk, the head and the tokenizer as ONE model that
//! answers a question and can be trained on the answer.
//!
//! This crate had the three pieces and no composition of them: a caller had
//! to hold a [`ModernBert`], a [`LayaHead`] and a `QwenBpe`, build the packed
//! sequence itself, and know about the cross-handle `poll_wait` between the
//! two device halves. That put the *only* assembly of a Laya model in
//! `brain`'s SDK, which meant this crate could not train, evaluate or test
//! one without the SDK on top of it.
//!
//! So this is the direct counterpart of `decide::decide::Decide`, for the
//! other decision backbone, and it carries the same contract:
//!
//! * [`LayaDecision::score`] - one packed `(state, question)` forward,
//!   returning raw option logits and the act head's raw logits.
//! * [`LayaDecision::accumulate`] / [`LayaDecision::adamw_scaled`] /
//!   [`LayaDecision::zero_grads`] - the minibatch cycle, with the objective
//!   supplied by the caller as host code over `&[f32]` scores, so a proper
//!   scoring rule, a policy gradient ([`rlcd::reinforce`]) or a plain
//!   cross-entropy all reuse one datapath instead of forking it.
//! * [`LayaDecision::train_step_with`] - that cycle for a batch of one.
//! * [`LayaDecision::save_head`] - the trained head as a stand-alone
//!   safetensors adapter, reloadable by [`LayaDecision::set_head_weights`].
//!
//! ## Trunk training is a CONSTRUCTION choice, not a runtime flag
//!
//! [`Training::HeadOnly`] is the normal mode and `decide`'s equivalent dial
//! (`set_encoder_frozen`) is not, because at this size the difference is not
//! a few skipped dispatches - it is whether the model fits in memory at all.
//! A `Role::Trainable` parameter carries a gradient and two AdamW moments, so
//! a trainable ModernBERT-large trunk is 395M x 4 floats ~ 6.3 GB of device
//! memory BEFORE any activation. `HeadOnly` builds the trunk `Role::Frozen`
//! (weights only, ~1.6 GB) and gives the head's backward a small scratch
//! buffer to write its hidden-state gradient into instead of the trunk's own
//! seed buffer - the gradient is computed either way, it is simply not
//! consumed.
//!
//! Three more reasons `HeadOnly` is the default, all pointing the same way:
//!
//! 1. [`LayaDecision::save_head`] writes the HEAD only. A head reattached to
//!    a moved trunk is not the model that was saved - the same argument
//!    `decide::decide::Decide::save_head` makes, and it refuses for the same
//!    reason.
//! 2. The trunk's reverse pass and its optimizer step are the overwhelming
//!    majority of a training step's cost at this size.
//! 3. The released checkpoint's trunk is already fine-tuned for exactly this
//!    task. A short run on a handful of examples is far more likely to damage
//!    it than to improve it.
//!
//! [`Training::HeadAndTrunk`] is there because the published Laya loop does
//! fine-tune the trunk (at a 4x lower learning rate than the head), and
//! because a randomly-initialized trunk CANNOT be frozen and still learn:
//! every option marker is the same `[MASK]` token, so at random init the
//! trunk maps them all to nearly the same hidden state and the head has
//! nothing to tell the options apart by. That is a measured property of this
//! architecture, not a hypothesis - see
//! `crates/modernbert/tests/train_convergence.rs`.
//!
//! ## What is NOT trained
//!
//! The act/escalate head. [`LayaHead::backward`] is seeded with a zero
//! act-logit gradient on every step, so those parameters come back exactly as
//! the checkpoint shipped them. That is not an oversight: no public Laya
//! source defines that head's objective (see [`rlcd::reinforce`]'s own module
//! doc), and the only published training loop does not train it either.
//!
//! Swedish Embedded AB builds decision models that answer with a calibrated
//! probability over options supplied at call time, and trains them on the
//! customer's own labelled data. If your team needs expertise in
//! encoder-plus-decision-head models, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::collections::HashMap;

use data::qwen_tokenizer::QwenBpe;
use gpu_core::{DeviceBuffer, Gpu};

use crate::config::ModernBertConfig;
use crate::laya::{tensor_manifest, LayaConfig, LayaHead};
use crate::model::ModernBert;
use crate::sequence::{build_sequence, Question, State};

/// How many option markers one call may carry. The head's marker buffers are
/// sized once at build time, so this is a capacity rather than a per-call
/// number; `decide::primitives::MAX_OPTIONS` is the same published limit on
/// the other backbone, transcribed rather than depended on (this crate sits
/// beside `brain-decide`, not above it).
pub const MAX_OPTIONS: usize = 255;

/// What a saved head attaches to, so the file needs nothing out of band.
///
/// Deliberately a small local struct rather than a reuse of
/// `decide::decide::Provenance`: this crate may not depend on that one (both
/// are model crates, and `scripts/gates/check-crate-layers.sh` enforces it).
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    /// The checkpoint these weights attach to, as a Hugging Face reference -
    /// `convaiinnovations/laya`.
    pub base: String,
    /// The fully-qualified id of this head, `vendor/repo`.
    pub id: String,
    /// What it was trained on and how. Free-form, written verbatim into the
    /// checkpoint's config so nothing about a run is lost.
    pub task: serde_json::Value,
}

/// What a [`LayaDecision`] is built to train - see the module doc for why
/// this is fixed at construction rather than a runtime flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Training {
    /// Inference only: no gradients, no optimizers, no reverse pass.
    Off,
    /// The from-scratch decision head learns; the pretrained trunk is held
    /// fixed and is not even allocated gradient/moment storage.
    HeadOnly,
    /// Both halves learn, at separate learning rates. Costs ~4x the trunk's
    /// weight bytes in device memory.
    HeadAndTrunk,
}

/// A loaded Laya decision model: ModernBERT trunk + decision head +
/// tokenizer. See the module doc.
pub struct LayaDecision {
    enc: ModernBert,
    head: LayaHead,
    tok: QwenBpe,
    cfg: ModernBertConfig,
    laya_cfg: LayaConfig,
    max_len: u32,
    head_max_len: u32,
    mode: Training,
    /// Where the head's backward writes its hidden-state gradient when the
    /// trunk is NOT being trained: a write-only sink the same shape as
    /// `ModernBert::seed_buf`, so the head's reverse pass is byte-identical
    /// in both modes and only its consumer differs. `None` in
    /// [`Training::HeadAndTrunk`] (the trunk's own seed buffer is the
    /// destination) and in [`Training::Off`] (there is no backward).
    grad_sink: Option<DeviceBuffer>,
    provenance: Provenance,
}

impl LayaDecision {
    /// Build both halves on one device.
    ///
    /// `gpu` is consumed: the trunk gets a `share()`d handle and the head
    /// keeps the original, matching how every other two-half model in this
    /// workspace is wired (and why [`LayaDecision::score`] has to
    /// `poll_wait` between them).
    ///
    /// `mode` fixes what is trainable for this model's whole life - see
    /// [`Training`] and the module doc. An inference-only build cannot later
    /// be trained; rebuild it.
    #[allow(clippy::too_many_arguments)]
    pub fn new_on(
        gpu: Gpu,
        cfg: ModernBertConfig,
        laya_cfg: LayaConfig,
        tok: QwenBpe,
        max_len: u32,
        head_max_len: u32,
        enc_init: &HashMap<String, Vec<f32>>,
        head_init: &HashMap<String, Vec<f32>>,
        mode: Training,
    ) -> LayaDecision {
        let cap_markers = MAX_OPTIONS as u32;
        // Allocated before `gpu` is split, and only when a backward exists
        // that has nowhere else to write - see `grad_sink`'s own doc.
        let grad_sink = (mode == Training::HeadOnly)
            .then(|| gpu.storage(max_len as u64 * cfg.d_model as u64));
        let enc = if mode == Training::HeadAndTrunk {
            ModernBert::new_train_on(gpu.share(), cfg.clone(), max_len, max_len, enc_init)
        } else {
            ModernBert::new_on(gpu.share(), cfg.clone(), max_len, max_len, enc_init)
        };
        let head = if mode == Training::Off {
            LayaHead::new_on(gpu, laya_cfg.clone(), max_len, max_len, cap_markers, 1, head_init)
        } else {
            LayaHead::new_train_on(gpu, laya_cfg.clone(), max_len, max_len, cap_markers, 1, head_init)
        };
        LayaDecision {
            enc,
            head,
            tok,
            cfg,
            laya_cfg,
            max_len,
            head_max_len,
            mode,
            grad_sink,
            provenance: Provenance::default(),
        }
    }

    /// Record what this head attaches to and what it was trained for - see
    /// [`Provenance`] and [`LayaDecision::save_head`].
    pub fn set_provenance(&mut self, p: Provenance) {
        self.provenance = p;
    }

    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    pub fn cfg(&self) -> &ModernBertConfig {
        &self.cfg
    }

    pub fn tokenizer(&self) -> &QwenBpe {
        &self.tok
    }

    /// `rl_agent_config.json`'s own `max_len` - the checkpoint's real trained
    /// context, which [`build_sequence`] truncates to.
    pub fn max_len(&self) -> u32 {
        self.max_len
    }

    /// `rl_agent_config.json`'s own `head_max_len` - the token budget the
    /// option markers and their texts share.
    pub fn head_max_len(&self) -> u32 {
        self.head_max_len
    }

    /// What this model was built to train.
    pub fn training(&self) -> Training {
        self.mode
    }

    /// Whether it can be trained at all.
    pub fn is_trainable(&self) -> bool {
        self.mode != Training::Off
    }

    /// Whether the trunk is held fixed - true in every mode but
    /// [`Training::HeadAndTrunk`].
    pub fn trunk_frozen(&self) -> bool {
        self.mode != Training::HeadAndTrunk
    }

    /// The head's AdamW time index.
    pub fn steps_taken(&self) -> u32 {
        self.head.steps_taken()
    }

    /// Pack one `(state, question)` and run the forward.
    ///
    /// Returns `(option_logits, act_logits)` - RAW, before any serving
    /// temperature: calibration belongs to whoever publishes the number, not
    /// to the model. `option_order` permutes the options within the packed
    /// sequence exactly as `rl_common.py::encode_record` does when training
    /// (see [`build_sequence`]); the returned logits are in THAT order.
    pub fn score(
        &mut self,
        state: &State,
        q: &Question,
        option_order: Option<&[usize]>,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let n_opts = q.render_options().len();
        if n_opts > MAX_OPTIONS {
            return Err(format!("{n_opts} options exceeds the {MAX_OPTIONS} this head is built for"));
        }
        let (ids, markers) = build_sequence(
            &self.tok,
            &self.cfg,
            state,
            q,
            self.max_len,
            self.head_max_len,
            option_order,
            false,
        );
        if ids.is_empty() || markers.is_empty() {
            return Err("laya: build_sequence produced no option markers for this question".into());
        }
        let rows = ids.len() as u32;
        let spans = [(0u32, rows)];
        self.enc.set_batch(&ids, &spans);
        self.enc.forward();
        // Cross-`Gpu`-handle synchronization, and it MUST NOT be removed: the
        // two halves hold separate handles onto one device (`gpu.share()`),
        // so a submit on one is not ordered against a submit on the other.
        // Skipping it reads a hidden state the trunk has not finished
        // writing - the model still answers, always wrongly. See
        // `ModernBert::poll_wait`'s own doc and `decide::decide::Decide::
        // run_packed`'s independently discovered identical note.
        self.enc.poll_wait();

        let marker_rows: Vec<u32> = markers.iter().map(|&m| m as u32).collect();
        let qtype = [q.qtype().index()];
        let arity = [markers.len()];
        if self.mode != Training::Off {
            // The head's reverse pass is identical in both training modes;
            // only who reads its hidden-state gradient differs.
            let d_hidden_out = match &self.grad_sink {
                Some(sink) => sink,
                None => self.enc.seed_buf(),
            };
            self.head
                .set_call_train(self.enc.hidden_buf(), d_hidden_out, &spans, &qtype, &marker_rows, &arity);
        } else {
            self.head.set_call(self.enc.hidden_buf(), &spans, &qtype, &marker_rows, &arity);
        }
        let (logits, act_logits) = self.head.forward();
        self.head.poll_wait();
        Ok((logits, act_logits))
    }

    /// Clear both halves' parameter gradients.
    ///
    /// Public because a caller accumulating a MINIBATCH owns the cycle: zero
    /// once, [`LayaDecision::accumulate`] over the batch, then
    /// [`LayaDecision::adamw_scaled`] once. A single transition's gradient is
    /// a very noisy estimate of a policy gradient, and AdamW applied to it
    /// directly chases the noise.
    pub fn zero_grads(&mut self) {
        if self.mode == Training::HeadAndTrunk {
            self.enc.zero_grads();
        }
        self.head.zero_grads();
    }

    /// Forward and backward for ONE example, ACCUMULATING into the parameter
    /// gradients without stepping the optimizer.
    ///
    /// `objective` receives this call's raw option logits (in `option_order`)
    /// and returns `(loss, dL/d(logit))`. It is host code over host floats,
    /// so it may sample, look up a return, or clip a ratio - which is what
    /// lets [`rlcd::reinforce`]'s policy gradient reuse this datapath rather
    /// than fork it.
    pub fn accumulate(
        &mut self,
        state: &State,
        q: &Question,
        option_order: Option<&[usize]>,
        objective: impl FnOnce(&[f32]) -> (f32, Vec<f32>),
    ) -> Result<f32, String> {
        if self.mode == Training::Off {
            return Err("accumulate on a model built with Training::Off".into());
        }
        let (logits, _act) = self.score(state, q, option_order)?;
        // Every option must have SURVIVED the packing budget. `build_sequence`
        // drops markers that fall past `max_len`, and training against a
        // truncated answer space teaches the wrong thing - the reference
        // skips such items outright ("options did not fit; skip rather than
        // train on a truncated answer space"). A training loop that silently
        // skipped would report a step count it did not take, so this is an
        // error naming the fix instead.
        let want = q.render_options().len();
        if logits.len() != want {
            return Err(format!(
                "laya: only {} of {want} options fit in max_len={} / head_max_len={} - training on a \
                 truncated answer space would teach the wrong answer; shorten the option text, the \
                 instructions, or the state",
                logits.len(),
                self.max_len,
                self.head_max_len
            ));
        }
        let (loss, d_logits) = objective(&logits);
        if d_logits.len() != logits.len() {
            return Err(format!(
                "objective returned {} gradients for {} option logits",
                d_logits.len(),
                logits.len()
            ));
        }
        // The act head is NOT trained - see the module doc. A zero seed is
        // the honest way to say so: the parameters still take part in the
        // reverse pass (they contribute nothing) and come back unchanged.
        let d_act = vec![0.0f32; self.laya_cfg.n_act as usize];
        self.head.backward(&d_logits, &d_act);
        // The head's reverse pass WRITES the trunk's seed buffer, on a
        // different handle. Same ordering hazard as the forward's own wait.
        self.head.poll_wait();
        if self.mode == Training::HeadAndTrunk {
            // Recorded here rather than in `score`: a frozen-trunk run never
            // reaches this line, and recording a reverse pass it will not run
            // is pure cost on every single step.
            self.enc.prepare_reverse();
            self.enc.backward_seeded();
            self.enc.poll_wait();
        }
        Ok(loss)
    }

    /// One AdamW update on whatever gradient has accumulated.
    ///
    /// `scale` turns a summed minibatch gradient into a mean (`1.0 / n`), so
    /// one learning rate stays meaningful across batch sizes. Weight decay
    /// `0.01` and gradient clipping at global norm `1.0` are the published
    /// Laya fine-tuning loop's own settings (see [`rlcd::reinforce`]).
    pub fn adamw_scaled(&mut self, trunk_lr: f32, head_lr: f32, scale: f32) {
        if self.mode == Training::HeadAndTrunk {
            self.enc.adamw_step_scaled(trunk_lr, 0.01, Some(1.0), scale);
        }
        self.head.adamw_step_scaled(head_lr, 0.01, Some(1.0), scale);
    }

    /// The full cycle for a batch of one: zero, accumulate, step.
    pub fn train_step_with(
        &mut self,
        state: &State,
        q: &Question,
        option_order: Option<&[usize]>,
        trunk_lr: f32,
        head_lr: f32,
        objective: impl FnOnce(&[f32]) -> (f32, Vec<f32>),
    ) -> Result<f32, String> {
        self.zero_grads();
        let l = self.accumulate(state, q, option_order, objective)?;
        self.adamw_scaled(trunk_lr, head_lr, 1.0);
        Ok(l)
    }

    /// Every head parameter, for snapshotting mid-run.
    pub fn head_weights(&self) -> Vec<(String, Vec<f32>)> {
        tensor_manifest(&self.laya_cfg)
            .into_iter()
            .map(|(name, _)| {
                let v = self.head.read_weight(&name);
                (name, v)
            })
            .collect()
    }

    /// Put a snapshot back - the reload half of [`LayaDecision::save_head`].
    pub fn set_head_weights(&self, w: &[(String, Vec<f32>)]) {
        for (name, v) in w {
            self.head.set_weight(name, v);
        }
    }

    /// Write the trained head as a stand-alone safetensors adapter.
    ///
    /// Only the head: the trunk is imported from a published checkpoint and
    /// re-importing it is free, so a run's artifact is the part that did not
    /// exist before it. That holds while the trunk is FROZEN, which is the
    /// default here, and stops holding the moment it is not - so this
    /// REFUSES on a model whose trunk was trained, rather than writing a file
    /// that silently reattaches the head to the published trunk. A run would
    /// otherwise select a policy on its measured score, write this, and the
    /// next run would inherit a different model under the same name and go on
    /// comparing it against that score.
    ///
    /// Tensors keep their real shapes, so a reader outside this repository
    /// needs nothing out of band.
    pub fn save_head(&self, path: &str) -> Result<(), String> {
        if !self.trunk_frozen() {
            return Err(format!(
                "cannot write {path}: the trunk was trained, and this format carries only the head. \
                 Loading it back would attach the head to the PUBLISHED trunk, which is not the \
                 model being saved"
            ));
        }
        let shapes: HashMap<String, Vec<usize>> = tensor_manifest(&self.laya_cfg).into_iter().collect();
        let mut n_params = 0u64;
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .head_weights()
            .into_iter()
            .map(|(name, v)| {
                n_params += v.len() as u64;
                let shape = shapes
                    .get(&name)
                    .map(|s| s.iter().map(|&d| d as u64).collect())
                    .unwrap_or_else(|| vec![v.len() as u64]);
                (name, shape, v)
            })
            .collect();

        let p = &self.provenance;
        let mut config = serde_json::json!({
            "model_type": "laya-decision-head",
            "d_model": self.laya_cfg.d_model,
            "head_layers": self.laya_cfg.head_layers,
            "n_heads": self.laya_cfg.n_heads,
            "ff_mult": self.laya_cfg.ff_mult,
            "act_hidden": self.laya_cfg.act_hidden,
            "n_act": self.laya_cfg.n_act,
            "eps": self.laya_cfg.eps,
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
        });
        if let Some(o) = config.as_object_mut() {
            o.insert("adapter_of".into(), serde_json::json!(p.base));
            o.insert("head".into(), serde_json::json!("laya-decision-head"));
            if !p.task.is_null() {
                o.insert("trained_for".into(), p.task.clone());
            }
        }
        let (vendor, repo) = match p.id.split_once('/') {
            Some((v, r)) => (Some(v.to_string()), Some(r.to_string())),
            None => (None, None),
        };
        let card = checkpoint::st::ModelCard {
            schema_version: 1,
            id: if p.id.is_empty() { "laya-decision-head".into() } else { p.id.clone() },
            display_name: None,
            family: "laya-decision-head".into(),
            architecture: Some("modernbert-large-laya-head".into()),
            variant_of: None,
            adapter: Some(checkpoint::st::Adapter {
                kind: "laya-decision-head".into(),
                base: (!p.base.is_empty()).then(|| p.base.clone()),
                ..Default::default()
            }),
            param_count: Some(n_params),
            embedding_dim: Some(self.laya_cfg.d_model as u64),
            license: Some("Apache-2.0".into()),
            vendor,
            repo,
            ..Default::default()
        };
        checkpoint::save_carded(path, config, &tensors, &card);
        Ok(())
    }

    /// Read a head written by [`LayaDecision::save_head`] back into the
    /// `name -> weights` map [`LayaDecision::new_on`] takes, so a reloaded
    /// pipeline is built the same way a fresh one is rather than through a
    /// second construction path.
    pub fn read_head_file(path: &str) -> Result<HashMap<String, Vec<f32>>, String> {
        let tensors = checkpoint::safetensors::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Ok(tensors.into_iter().map(|t| (t.name, t.data)).collect())
    }
}
