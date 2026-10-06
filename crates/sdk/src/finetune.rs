// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements LoRA fine-tuning of chat models on a
// product's own conversations for its clients. If your team needs expertise
// in supervised fine-tuning and held-out evaluation of language models, you
// can procure our services by sending an email to info@swedishembedded.com.

//! [`ChatFineTune`]: LoRA fine-tuning of a Qwen3 chat model on a chat
//! dataset, and [`score_chat`]: the held-out measurement that says whether it
//! helped.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::ChatFineTune::from_pretrained("/models/qwen3-0.6b/model.brain.safetensors")
//!     .dataset("train.jsonl")
//!     .held_out("held_out.jsonl")
//!     .out_dir("runs/support-1")
//!     .steps(200)
//!     .run()?;
//! if let (Some(base), Some(tuned)) = (outcome.base_score, outcome.tuned_score) {
//!     println!("held-out loss {:?} -> {:?}", base.loss, tuned.loss);
//! }
//! # Ok(()) }
//! ```
//!
//! The dataset is `generic-messages-v2` JSONL (see
//! [`crate::validate_chat_dataset`]), checked by
//! [`crate::validate_chat_dataset_for`] against the base's own tokenizer and
//! chat template before a device is claimed, and packed by
//! `data::chat::prepare_chat_samples`. Training is
//! `qwen3::finetune::finetune_lora_controlled` over `model::fit_controlled`,
//! the one training loop; scoring is `qwen3::eval::score_chat`. Nothing here
//! is a second implementation of any of them.
//!
//! **The mix.** Training draws rows uniformly with replacement from the
//! dataset and every replay file. [`ChatFineTune::replay`] files join the
//! dataset as a plain union, or at [`ChatFineTune::replay_share`] of the
//! draws together; [`ChatFineTune::replay_at`] gives one file a share of its
//! own. A share is kept by listing each group of records as many times as
//! its share needs ([`mix_repeats`]), so a replay set many times the
//! dataset, or many times smaller, is drawn as often as asked.
//!
//! **Monitoring and selection.** With [`ChatFineTune::eval_every`] the run
//! scores a monitoring set ([`ChatFineTune::monitor`], else the held-out set)
//! as it trains and records the curve ([`ChatFineTuneOutcome::curve`]); with
//! [`ChatFineTune::keep_best`] or a [`ChatFineTune::patience`] it exports the
//! adapter of the evaluation with the lowest monitoring loss instead of the
//! last step's, and a patience stops it once that loss has not improved for
//! that many evaluations. Which step was exported, and why, is on the
//! outcome, in the training record and on the adapter's card.
//!
//! **Exact resume.** A run whose [`CancelToken`] fires stops at the next
//! step boundary and leaves its training state in the out directory:
//! weights, AdamW moments, step and batch position, and the selection so
//! far. Running the same fine-tune again continues it, and ends at the same
//! adapter an uninterrupted run would have produced. A state from a run with
//! different data, options or starting point is refused, never continued.

use std::path::{Path, PathBuf};

use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;

use crate::{CancelToken, Device, Error, Result};

/// Where one run keeps its files, inside [`ChatFineTune::out_dir`].
const PREPARED_DIR: &str = "prepared";
const STATE_FILE: &str = "train.state";
const ADAPTER_FILE: &str = "adapter.safetensors";
const RECORD_FILE: &str = "training.json";

/// Training rows are at least this long, so a slightly longer next dataset
/// does not change the row length.
const MIN_BLOCK: u32 = 64;

/// A LoRA fine-tune of a Qwen3 chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ChatFineTune {
    base: String,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    held_out: Option<PathBuf>,
    monitor: Option<PathBuf>,
    replay: Vec<PathBuf>,
    replay_share: Option<f32>,
    weighted_replay: Vec<(PathBuf, f32)>,
    grad_accum: u32,
    continue_from: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    adapter_id: Option<String>,
    rank: Option<u32>,
    alpha: Option<f32>,
    weight_decay: f32,
    cooldown_steps: Option<u32>,
    steps: u32,
    lr: f32,
    seed: u64,
    max_block: Option<u32>,
    checkpoint_every: u32,
    eval_every: u32,
    patience: u32,
    keep_best: bool,
    cycle: u64,
    device: Device,
    bf16_base: bool,
    int8_base: bool,
    thinking: bool,
}

/// The tier the frozen base is held at: int8 over bf16 over fp32.
fn base_tier(bf16: bool, int8: bool) -> qwen3::Dtype {
    if int8 {
        qwen3::Dtype::I8
    } else if bf16 {
        qwen3::Dtype::BF16
    } else {
        qwen3::Dtype::F32
    }
}

impl ChatFineTune {
    /// A fine-tune of `base`: a checkpoint file, a `<models>/<vendor>/<repo>`
    /// directory, or a `vendor/repo` reference in the model store. The
    /// tokenizer (`tokenizer.json`) and chat template are read from the
    /// checkpoint's directory, as training and serving both read them.
    pub fn from_pretrained(base: impl Into<String>) -> ChatFineTune {
        ChatFineTune {
            base: base.into(),
            models_dir: None,
            dataset: None,
            held_out: None,
            monitor: None,
            replay: Vec::new(),
            replay_share: None,
            weighted_replay: Vec::new(),
            grad_accum: 1,
            continue_from: None,
            out_dir: None,
            adapter_id: None,
            rank: None,
            alpha: None,
            weight_decay: LORA_WEIGHT_DECAY,
            cooldown_steps: None,
            steps: 100,
            lr: 3e-4,
            seed: 1337,
            max_block: None,
            checkpoint_every: 0,
            eval_every: 0,
            patience: 0,
            keep_best: false,
            cycle: 0,
            device: Device::default(),
            bf16_base: false,
            int8_base: false,
            thinking: true,
        }
    }

    /// The models directory a `vendor/repo` base resolves in (default:
    /// `BRAIN_MODELS_DIR`, then the standard location).
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The chat dataset to train on (required).
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// Records to score base and tuned on, never trained on. Without it the
    /// outcome's scores are `None`.
    pub fn held_out(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out = Some(path.into());
        self
    }

    /// Records scored during training, every [`Self::eval_every`] steps,
    /// never trained on: the curve, and what the best adapter is selected on.
    /// Without it the held-out set is monitored - and a selection made on it
    /// biases the held-out score it is then measured by, so a caller that
    /// decides anything on that score gives the run a monitoring set of its own.
    pub fn monitor(mut self, path: impl Into<PathBuf>) -> Self {
        self.monitor = Some(path.into());
        self
    }

    /// Score the monitoring set every this many steps, and at the last step
    /// of a run that selects on it (default 0: never). Each evaluation is
    /// one deterministic pass over every monitoring record.
    pub fn eval_every(mut self, steps: u32) -> Self {
        self.eval_every = steps;
        self
    }

    /// Stop once the monitoring loss has not improved for this many
    /// evaluations, and export the adapter of the best one (default 0:
    /// never stop). Needs [`Self::eval_every`].
    pub fn patience(mut self, evaluations: u32) -> Self {
        self.patience = evaluations;
        self
    }

    /// Export the adapter of the evaluation with the lowest monitoring loss
    /// rather than the last step's, running every step (default off; a
    /// [`Self::patience`] implies it). Needs [`Self::eval_every`].
    pub fn keep_best(mut self, on: bool) -> Self {
        self.keep_best = on;
        self
    }

    /// Mix every record of another chat dataset into training - earlier
    /// experience replayed so the new data does not overwrite it. May be
    /// called more than once; each file is validated like the dataset. The
    /// mix is the union of the files unless [`Self::replay_share`] is set.
    pub fn replay(mut self, path: impl Into<PathBuf>) -> Self {
        self.replay.push(path.into());
        self
    }

    /// The share of training draws that come from the [`Self::replay`] files
    /// in all (0 < share < 1), however large they are next to the dataset.
    /// Examples are drawn uniformly with replacement, so a replay set many
    /// times the dataset's size would otherwise take almost every step, and
    /// one much smaller would hardly be drawn; the dataset's records and the
    /// replay's are each listed as many times as puts the replay at this
    /// share ([`mix_repeats`]). Without it the mix is the plain union.
    pub fn replay_share(mut self, share: f32) -> Self {
        self.replay_share = Some(share);
        self
    }

    /// Mix another chat dataset into training at `share` of the draws
    /// (0 < share < 1) of its own, beside the dataset and the [`Self::replay`]
    /// files: records a run must keep drawing at a set rate whatever else is
    /// trained, such as a base model's own answers replayed so the new data
    /// does not move it off them. May be called more than once; the shares
    /// of every weighted file and [`Self::replay_share`] together must leave
    /// the dataset some of the draws.
    pub fn replay_at(mut self, path: impl Into<PathBuf>, share: f32) -> Self {
        self.weighted_replay.push((path.into(), share));
        self
    }

    /// Micro-batches summed into each optimizer step (default 1). Rows are
    /// single examples, so this is the effective batch size.
    pub fn grad_accum(mut self, micro_batches: u32) -> Self {
        self.grad_accum = micro_batches;
        self
    }

    /// Continue training this adapter (a file [`ChatFineTune`] or brain's
    /// LoRA trainer wrote) instead of starting a fresh one. Its rank and
    /// alpha are the run's; its digest is recorded as the new adapter's
    /// parent (`trained_from`).
    pub fn continue_from(mut self, adapter: impl Into<PathBuf>) -> Self {
        self.continue_from = Some(adapter.into());
        self
    }

    /// Where the adapter, its training record, the packed dataset and any
    /// resume state go (required). One directory per run.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// The id written on the adapter's card (default `<base id>:local:chat:latest`).
    pub fn adapter_id(mut self, id: impl Into<String>) -> Self {
        self.adapter_id = Some(id.into());
        self
    }

    /// LoRA rank (default 8). When continuing an adapter the default is its
    /// own rank, and a different one is refused.
    pub fn rank(mut self, rank: u32) -> Self {
        self.rank = Some(rank);
        self
    }

    /// LoRA alpha; the update is scaled by `alpha / rank` (default `2 * rank`,
    /// or a continued adapter's own alpha - a different one is refused).
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// AdamW's decoupled weight decay on the adapter's matrices (default 0,
    /// as LoRA recipes use: the adapter starts at zero and decay only pulls
    /// it back there).
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Steps a run whose [`Self::patience`] runs out spends bringing the
    /// rate down to its floor before it stops (default a tenth of the
    /// steps): the anneal a run stopped on a plateau would otherwise never
    /// have, so the adapter it keeps is not the one trained at a high rate.
    pub fn cooldown(mut self, steps: u32) -> Self {
        self.cooldown_steps = Some(steps);
        self
    }

    /// Optimizer steps (default 100). The first twentieth warms the rate up,
    /// and it decays over all of them.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// Peak learning rate (default 3e-4); it decays to a tenth of this.
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Seeds the fresh adapter's initialisation and the batch order
    /// (default 1337).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The longest training row allowed. Rows are sized to the longest
    /// record; a record longer than this is refused by name rather than
    /// trained on truncated.
    pub fn max_block(mut self, tokens: u32) -> Self {
        self.max_block = Some(tokens);
        self
    }

    /// Also save the resume state every this-many steps (default 0: only when
    /// cancelled), so a crash loses at most this many steps.
    pub fn checkpoint_every(mut self, steps: u32) -> Self {
        self.checkpoint_every = steps;
        self
    }

    /// The caller's loop iteration this run belongs to, recorded in the
    /// adapter's training provenance (default 0).
    pub fn cycle(mut self, cycle: u64) -> Self {
        self.cycle = cycle;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Whether the model is trained to reason before it answers (the
    /// default: records are rendered as the base's template renders them). Off,
    /// the model is trained for no-think mode: each supervised answer follows
    /// the block the base's template leaves a no-think prompt in
    /// ([`data::chat_template::ChatTemplate::no_think_block`]) and keeps it
    /// through rendering, so training sees what a model asked not to reason is
    /// asked from - without it, a reasoning model trained on bare answers and
    /// asked with its think block closed (or open) has never seen that state.
    /// A base that never reasons is unaffected. Scoring before and after
    /// renders the same way.
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// Hold the frozen base at bf16, half the bytes of fp32 (a 7B decoder
    /// then trains on one 24 GB card); the adapters and optimiser stay fp32.
    /// Scoring before and after runs at the same tier.
    pub fn bf16_base(mut self, on: bool) -> Self {
        self.bf16_base = on;
        self
    }

    /// Hold the frozen base as int8 weights (a byte per weight and a scale per
    /// 32, about a quarter of fp32), which leaves the room of a 7B decoder's
    /// other half for activations: a longer context on one card. The weights
    /// are decoded as the matrix kernels read them and the activations stay
    /// fp32; the adapters and optimiser stay fp32. Takes precedence over
    /// [`Self::bf16_base`].
    pub fn int8_base(mut self, on: bool) -> Self {
        self.int8_base = on;
        self
    }

    /// Run to completion.
    pub fn run(&self) -> Result<ChatFineTuneOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step to `on_progress` and stopping at
    /// the next step boundary once `cancel` fires. A cancelled run exports no
    /// adapter; it leaves its resume state
    /// ([`ChatFineTuneOutcome::resume_state`]) and running the same fine-tune
    /// again continues it.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<ChatFineTuneOutcome> {
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no out directory; call .out_dir(path)".to_string()))?;
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let open = qwen3::store_checkpoint_path(&weights, &model_dir);
        let weights_str = utf8(&open)?;
        let tier = base_tier(self.bf16_base, self.int8_base);

        // Every file is checked against the base's own template before a
        // device is claimed.
        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render };
        let mut train_samples = asked.read(dataset)?;
        let dataset_records = train_samples.len();
        let mut replay_samples = Vec::new();
        for path in &self.replay {
            replay_samples.extend(asked.read(path)?);
        }
        let mut weighted_samples = Vec::with_capacity(self.weighted_replay.len());
        for (path, share) in &self.weighted_replay {
            weighted_samples.push((asked.read(path)?, *share));
        }
        let held_out = self.held_out.as_deref().map(|path| asked.read(path)).transpose()?;
        let monitor = self.monitor.as_deref().map(|path| asked.read(path)).transpose()?;
        if self.eval_every == 0 && (self.patience > 0 || self.keep_best) {
            return Err(Error::Backend("ChatFineTune: selecting on the monitoring loss needs eval_every > 0".to_string()));
        }

        let (rank, alpha, parent) = lora_shape(self.continue_from.as_deref(), self.rank, self.alpha)?;

        // The groups of the mix, the dataset first: the replay files at
        // their share when one is set, else in the dataset's own group as
        // the plain union they always were; then each weighted file.
        let replay_records = replay_samples.len() + weighted_samples.iter().map(|(s, _)| s.len()).sum::<usize>();
        let mut groups: Vec<(Vec<ChatSample>, f32)> = Vec::new();
        match self.replay_share {
            Some(share) => groups.push((std::mem::take(&mut replay_samples), share)),
            None => train_samples.append(&mut replay_samples),
        }
        groups.extend(weighted_samples);
        let shares: Vec<(usize, f32)> = std::iter::once((train_samples.len(), 0.0)).chain(groups.iter().map(|(s, share)| (s.len(), *share))).collect();
        let repeats = mix_repeats(&shares)?;
        let mut training = Vec::new();
        for (samples, &times) in std::iter::once(&train_samples).chain(groups.iter().map(|(s, _)| s)).zip(&repeats) {
            for _ in 0..times {
                training.extend(samples.iter().cloned());
            }
        }
        // The packed validation split is what a periodic evaluation scores:
        // the monitoring set, else the held-out set, else (unread, when
        // nothing is evaluated) the training set.
        let val = monitor.as_deref().or(held_out.as_deref()).unwrap_or(&training[..]);
        let cfg = qwen3::checkpoint_config(weights_str).map_err(Error::Backend)?;
        let prepared_dir = out_dir.join(PREPARED_DIR);
        let prepared = data::chat::prepare_chat_samples(&training, val, &tok, &tmpl, render, cfg.vocab as usize, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        let block = block_for(prepared.longest_example, self.max_block)?;

        crate::device::resolve(&self.device)?;
        let base_score = held_out.as_ref().map(|records| score_records(weights_str, None, &tok, &tmpl, records, block, tier, render));

        if self.grad_accum == 0 {
            return Err(Error::Backend("ChatFineTune: grad_accum must be at least 1".to_string()));
        }
        let opts = model::FitOpts {
            grad_accum: self.grad_accum,
            eval_interval: self.eval_every,
            // Every monitoring record, once, each time.
            eval_batches: 0,
            // Keeping the best without a patience is a patience the run
            // never reaches.
            patience: if self.keep_best && self.patience == 0 { u32::MAX } else { self.patience },
            weight_decay: self.weight_decay,
            cooldown_steps: self.cooldown_steps.unwrap_or(self.steps / COOLDOWN_DIVISOR),
            ..fit_opts(self.steps, block, self.lr, self.seed)
        };
        let base_digest = base_digest(&open)?;
        let identity = serde_json::json!({
            "base": base_digest,
            "dataset": digest(dataset)?,
            "replay": self.replay.iter().map(|p| digest(p)).collect::<Result<Vec<_>>>()?,
            "weighted_replay": self.weighted_replay.iter().map(|(p, share)| Ok((digest(p)?, share.to_bits()))).collect::<Result<Vec<_>>>()?,
            "monitor": self.monitor.as_deref().map(digest).transpose()?,
            "parent": parent,
            "rank": rank,
            "alpha_bits": alpha.to_bits(),
            "weight_decay_bits": self.weight_decay.to_bits(),
            "repeats": repeats,
            "grad_accum": self.grad_accum,
        });
        let state_path = out_dir.join(STATE_FILE);
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress { step: s.step, steps: s.steps, loss: s.loss, lr: s.lr });
            !cancel.is_cancelled()
        };
        let control = model::FitControl { on_step: Some(&mut on_step), state: Some(&state_path), state_every: self.checkpoint_every, identity };
        let start = match &self.continue_from {
            Some(path) => qwen3::finetune::LoraStart::Continue(utf8(path)?),
            None => qwen3::finetune::LoraStart::Fresh,
        };
        let (report, trained) = qwen3::finetune::finetune_lora_controlled(weights_str, &prepared_dir, &opts, rank, alpha, &start, control, tier).map_err(|e| Error::Backend(format!("training: {e}")))?;

        let (selected_step, selection) = match report.kept_best {
            Some(best) => (best.step, Selection::BestMonitorLoss),
            None => (report.steps_completed, Selection::LastStep),
        };
        let mut outcome = ChatFineTuneOutcome {
            status: FineTuneStatus::Completed,
            adapter: None,
            adapter_digest: None,
            record: None,
            resume_state: None,
            steps: self.steps,
            steps_completed: report.steps_completed,
            resumed_at: report.resumed_at,
            initial_loss: Some(report.initial_loss),
            final_loss: report.final_loss,
            train_records: dataset_records,
            replay_records,
            monitor_records: monitor.as_ref().map_or(0, Vec::len),
            eval_every: self.eval_every,
            patience: self.patience,
            curve: report.evaluations.iter().map(|e| MonitorPoint { step: e.step, train_loss: e.train_loss, monitor_loss: e.eval_loss }).collect(),
            selected_step,
            selection,
            stopped_early: report.stopped_early,
            block,
            rank,
            alpha,
            lr: self.lr,
            weight_decay: self.weight_decay,
            trained_from: parent.clone(),
            base_digest: Some(base_digest.clone()),
            base_score,
            tuned_score: None,
        };
        if report.interrupted {
            outcome.status = FineTuneStatus::Cancelled;
            outcome.resume_state = Some(state_path);
            return Ok(outcome);
        }

        let adapter_path = out_dir.join(ADAPTER_FILE);
        let adapter_id = self.adapter_id.clone().unwrap_or_else(|| format!("{base_id}:local:chat:latest"));
        let provenance = checkpoint::st::TrainingProvenance {
            code_revision: format!("brain {}", env!("CARGO_PKG_VERSION")),
            regime: "sft_lora".to_string(),
            seed: self.seed,
            hyperparams: serde_json::json!({
                "rank": rank,
                "alpha": alpha,
                "steps": self.steps,
                "steps_completed": outcome.steps_completed,
                "selected_step": outcome.selected_step,
                "selection": outcome.selection,
                "eval_every": self.eval_every,
                "patience": self.patience,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "block": block,
                "train_records": outcome.train_records,
                "replay_records": outcome.replay_records,
                "monitor_records": outcome.monitor_records,
            }),
            environment: gpu_core::backend_name().to_string(),
            gate: None,
            trained_from: parent,
            base_digest: Some(base_digest),
            cycle: self.cycle,
        };
        let dataset_id = digest(dataset)?;
        trained.save_adapter_with_lineage(utf8(&adapter_path)?, &adapter_id, &base_id, Some(&dataset_id), Some(provenance))
            .map_err(|e| Error::Backend(format!("{}: saving the adapter: {e}", adapter_path.display())))?;
        drop(trained);
        outcome.adapter_digest = Some(digest(&adapter_path)?);
        let adapter_str = utf8(&adapter_path)?;
        outcome.tuned_score = held_out.as_ref().map(|records| score_records(weights_str, Some(adapter_str), &tok, &tmpl, records, block, tier, render));
        outcome.adapter = Some(adapter_path);

        let record_path = out_dir.join(RECORD_FILE);
        std::fs::write(&record_path, serde_json::to_vec_pretty(&outcome.record_json(&base_id, &dataset_id)).map_err(|e| Error::Backend(e.to_string()))?)
            .map_err(|e| Error::Backend(format!("{}: {e}", record_path.display())))?;
        outcome.record = Some(record_path);
        // The run is exported; its state would only resume into the same
        // adapter again.
        std::fs::remove_file(&state_path).ok();
        Ok(outcome)
    }
}

/// One completed optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FineTuneProgress {
    /// Steps completed in the whole run (a resumed run continues the count).
    pub step: u32,
    pub steps: u32,
    /// This step's training loss.
    pub loss: f32,
    pub lr: f32,
}

/// How a fine-tune ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FineTuneStatus {
    /// Every step ran; the adapter and its record are written.
    Completed,
    /// The cancel token stopped it; the resume state is written instead.
    Cancelled,
}

/// What one fine-tune did. Anything not measured is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatFineTuneOutcome {
    pub status: FineTuneStatus,
    /// The adapter file; `None` unless completed.
    pub adapter: Option<PathBuf>,
    /// `sha256:<hex>` of the adapter file - the digest a [`crate::ChatPipeline`]
    /// serving it reports in its identity.
    pub adapter_digest: Option<String>,
    /// The JSON training record written beside the adapter.
    pub record: Option<PathBuf>,
    /// The resume state a cancelled run left; running again continues it.
    pub resume_state: Option<PathBuf>,
    pub steps: u32,
    /// Steps completed in the whole run, including any before a resume.
    pub steps_completed: u32,
    /// The step this call resumed at, when it continued a saved state.
    pub resumed_at: Option<u32>,
    /// The training loss estimated before this call's first step.
    pub initial_loss: Option<f32>,
    /// The last step's training loss; `None` when this call ran no step.
    pub final_loss: Option<f32>,
    pub train_records: usize,
    pub replay_records: usize,
    /// Records in the monitoring set ([`ChatFineTune::monitor`]); 0 without one.
    pub monitor_records: usize,
    /// Steps between evaluations; 0 when nothing was evaluated during training.
    pub eval_every: u32,
    /// Evaluations without improvement the run was allowed; 0 when it was
    /// never to stop early.
    pub patience: u32,
    /// Every evaluation taken during training, in order; empty without
    /// [`ChatFineTune::eval_every`].
    pub curve: Vec<MonitorPoint>,
    /// The step whose adapter was exported: the best evaluation's, or the
    /// last step trained.
    pub selected_step: u32,
    /// Why that step.
    pub selection: Selection,
    /// True when the patience ran out before the step budget did.
    pub stopped_early: bool,
    /// The training row length, sized to the longest record.
    pub block: u32,
    pub rank: u32,
    pub alpha: f32,
    /// The peak learning rate of the run.
    pub lr: f32,
    /// The AdamW weight decay of the run.
    pub weight_decay: f32,
    /// The digest of the adapter this run continued, if it continued one.
    pub trained_from: Option<String>,
    /// `sha256:<hex>` of the base checkpoint file the adapter was trained on -
    /// the digest its card records as `TrainingProvenance::base_digest`, so a
    /// caller binding the adapter to its base need not hash the base again.
    /// `None` when the run did not compute it.
    pub base_digest: Option<String>,
    /// The base on the held-out records; `None` without [`ChatFineTune::held_out`].
    pub base_score: Option<HeldOutScore>,
    /// Base plus the new adapter on the same records; `None` unless
    /// completed with a held-out set.
    pub tuned_score: Option<HeldOutScore>,
}

impl ChatFineTuneOutcome {
    fn record_json(&self, base_id: &str, dataset_id: &str) -> serde_json::Value {
        let score = |s: &Option<HeldOutScore>| {
            s.as_ref().map(|s| serde_json::json!({ "loss": s.loss, "token_accuracy": s.token_accuracy, "positions": s.positions, "records": s.records, "skipped": s.skipped }))
        };
        serde_json::json!({
            "base": base_id,
            "base_digest": self.base_digest,
            "dataset": dataset_id,
            "adapter_digest": self.adapter_digest,
            "trained_from": self.trained_from,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "train_records": self.train_records,
            "replay_records": self.replay_records,
            "monitor_records": self.monitor_records,
            "eval_every": self.eval_every,
            "patience": self.patience,
            "curve": self.curve.iter().map(|p| serde_json::json!({ "step": p.step, "train_loss": p.train_loss, "monitor_loss": p.monitor_loss })).collect::<Vec<_>>(),
            "selected_step": self.selected_step,
            "selection": self.selection,
            "stopped_early": self.stopped_early,
            "block": self.block,
            "rank": self.rank,
            "alpha": self.alpha,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "base_score": score(&self.base_score),
            "tuned_score": score(&self.tuned_score),
        })
    }
}

/// One evaluation of the monitoring set during training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MonitorPoint {
    /// Steps completed when it was taken.
    pub step: u32,
    /// The mean training loss of the steps since the previous evaluation.
    pub train_loss: f32,
    /// The mean per-token loss over the monitoring records.
    pub monitor_loss: f32,
}

/// Which step's adapter a fine-tune exported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Selection {
    /// The last step trained: the run did not select on the monitoring loss.
    LastStep,
    /// The evaluation with the lowest monitoring loss.
    BestMonitorLoss,
}

/// Teacher-forced cross-entropy of a model on held-out chat records, over
/// exactly the positions training supervises.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldOutScore {
    /// Mean per-token cross-entropy over the supervised positions (lower is
    /// better); `None` when no position was scored.
    pub loss: Option<f32>,
    /// Fraction of supervised positions whose greedy next token was right;
    /// `None` when no position was scored.
    pub token_accuracy: Option<f64>,
    /// Supervised positions scored.
    pub positions: usize,
    /// Records scored.
    pub records: usize,
    /// Records skipped (longer than the scoring row, or not encodable).
    pub skipped: usize,
}

/// Score `base` - optionally with `adapter` folded in, exactly as serving it
/// would - on the held-out chat records in `held_out`. `base` is resolved
/// like [`ChatFineTune::from_pretrained`]'s; the tokenizer and template come
/// from its directory. The scoring row fits the longest record, so none is
/// skipped for length.
pub fn score_chat(base: &str, adapter: Option<&Path>, held_out: &Path) -> Result<HeldOutScore> {
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    let open = qwen3::store_checkpoint_path(&weights, &model_dir);
    let weights_str = utf8(&open)?;
    qwen3::checkpoint_config(weights_str).map_err(Error::Backend)?;
    let records = read_checked(held_out, &model_dir)?;
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let mut longest = 0usize;
    for (i, r) in records.iter().enumerate() {
        let (ids, _) = r.encode(&tok, &tmpl).map_err(|e| Error::Backend(format!("{}: record {}: {e}", held_out.display(), i + 1)))?;
        longest = longest.max(ids.len());
    }
    let adapter = match adapter {
        Some(path) => {
            checkpoint::st::read_card(utf8(path)?).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
            Some(utf8(path)?)
        }
        None => None,
    };
    Ok(score_records(weights_str, adapter, &tok, &tmpl, &records, block_for(longest, None)?, qwen3::Dtype::F32, data::chat::RenderOpts::default()))
}

#[allow(clippy::too_many_arguments)]
fn score_records(weights: &str, adapter: Option<&str>, tok: &QwenBpe, tmpl: &ChatTemplate, records: &[ChatSample], block: u32, tier: qwen3::Dtype, render: data::chat::RenderOpts) -> HeldOutScore {
    let s = qwen3::eval::score_chat_rendered(weights, adapter, tok, tmpl, records, block, tier, render);
    HeldOutScore { loss: (s.positions > 0).then_some(s.loss), token_accuracy: s.token_accuracy, positions: s.positions, records: s.samples, skipped: s.skipped }
}

/// The LoRA rank and alpha of a run, and the digest of the adapter it
/// continues: a fresh adapter is rank 8 (or `rank`) at alpha `2 * rank`; a
/// continued one is its own rank - a different requested rank is refused -
/// and its own alpha. `alpha` overrides either.
pub(crate) fn lora_shape(continue_from: Option<&Path>, rank: Option<u32>, alpha: Option<f32>) -> Result<(u32, f32, Option<String>)> {
    let (rank, default_alpha, parent) = match continue_from {
        None => {
            let rank = rank.unwrap_or(8);
            (rank, 2.0 * rank as f32, None)
        }
        Some(path) => {
            let card = checkpoint::st::read_card(utf8(path)?).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
            let adapter = card.and_then(|c| c.adapter).ok_or_else(|| Error::Backend(format!("{}: not an adapter file (no adapter card)", path.display())))?;
            let own = adapter.rank.ok_or_else(|| Error::Backend(format!("{}: the adapter card has no rank", path.display())))?;
            if let Some(asked) = rank.filter(|&asked| asked != own) {
                return Err(Error::Backend(format!("{}: a rank-{own} adapter cannot be continued at rank {asked}", path.display())));
            }
            (own, adapter.alpha.unwrap_or(own as f32), Some(digest(path)?))
        }
    };
    Ok((rank, alpha.unwrap_or(default_alpha), parent))
}

/// Validate `path` against the base's tokenizer and template, then read it.
/// How a fine-tune's records are asked for: the base's tokenizer and template,
/// and for a model trained for no-think mode the block its prompt leaves open.
struct Asked<'a> {
    tok: &'a QwenBpe,
    tmpl: &'a ChatTemplate,
    no_think: Option<&'a str>,
    render: data::chat::RenderOpts,
}

impl Asked<'_> {
    /// The samples of `path` as they are trained on, each checked to encode
    /// against the template and to supervise something. For no-think mode a
    /// dialogue is one example per answer, each as the model is asked for it
    /// ([`ChatSample::answers_as_asked`]); a record that is not made of plain
    /// answers keeps its shape, its answers following the block.
    fn read(&self, path: &Path) -> Result<Vec<ChatSample>> {
        let summary = crate::validate_chat_dataset(path).map_err(Error::Backend)?;
        if summary.trained_messages == 0 {
            return Err(Error::Backend(format!("{}: holds no supervised turns (`\"train\": true`)", path.display())));
        }
        let records = ChatSample::from_jsonl(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
        let mut samples = Vec::with_capacity(records.len());
        for (at, record) in records.iter().enumerate() {
            let invalid = |e: &dyn std::fmt::Display| Error::Backend(format!("{}: record {} cannot be encoded for training: {e}", path.display(), at + 1));
            match self.no_think {
                None => samples.push(record.clone()),
                Some(block) => match record.answers_as_asked(self.tmpl, false).map_err(|e| invalid(&e))? {
                    Some(asked) => samples.extend(asked),
                    None => samples.push(record.answering_without_thinking(block)),
                },
            }
        }
        for (at, sample) in samples.iter().enumerate() {
            let (ids, mask) = sample.encode_with(self.tok, self.tmpl, self.render).map_err(|e| Error::Backend(format!("{}: example {} cannot be encoded for training: {e}", path.display(), at + 1)))?;
            if !mask.iter().any(|m| *m) {
                return Err(Error::Backend(format!("{}: example {} encodes to {} token(s) with none supervised, so training on it would be a no-op", path.display(), at + 1, ids.len())));
            }
        }
        Ok(samples)
    }
}

fn read_checked(path: &Path, model_dir: &Path) -> Result<Vec<ChatSample>> {
    let summary = crate::validate_chat_dataset_for(path, model_dir).map_err(Error::Backend)?;
    if summary.trained_messages == 0 {
        return Err(Error::Backend(format!("{}: holds no supervised turns (`\"train\": true`)", path.display())));
    }
    ChatSample::from_jsonl(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

pub(crate) fn tokenizer_and_template(model_dir: &Path) -> Result<(QwenBpe, ChatTemplate)> {
    let tmpl = ChatTemplate::from_model_dir(model_dir).map_err(|e| Error::Backend(format!("{}: {e}", model_dir.display())))?;
    let tok_path = model_dir.join("tokenizer.json");
    let tok = QwenBpe::from_file(utf8(&tok_path)?).map_err(|e| Error::Backend(format!("{}: {e}", tok_path.display())))?;
    Ok((tok, tmpl))
}

/// The smallest power-of-two row (at least [`MIN_BLOCK`]) that holds the
/// longest record, capped at `max`: a record longer than the cap is refused,
/// since training on a row that cannot hold it would supervise nothing of it.
pub(crate) fn block_for(longest: usize, max: Option<u32>) -> Result<u32> {
    if let Some(max) = max {
        if longest > max as usize {
            return Err(Error::Backend(format!("the longest record is {longest} tokens, past max_block {max}; raise max_block or shorten the record")));
        }
    }
    let mut block = MIN_BLOCK;
    while (block as usize) < longest {
        block *= 2;
    }
    Ok(max.map_or(block, |m| block.min(m)))
}

/// The most times the smallest group of a mix is listed in search of a closer
/// fit: repeats are whole numbers, and a ratio like 1.6 listed once or twice
/// is far from itself, but at a few times more it rounds within the tolerance.
const MIX_MAX_SCALE: usize = 16;

/// How far a group's realised share of the rows may lie from the share asked
/// before the mix is listed at a larger scale.
const MIX_TOLERANCE: f64 = 0.01;

/// How many times each group of records is listed so that, drawn uniformly
/// with replacement from all the rows, a group's records come up its share
/// of the time. `groups` is `(records, share)`, the dataset first: its share
/// is whatever the others leave. Each group's weight per record is its share
/// over its records; the repeats are those weights over the smallest, scaled
/// by the smallest whole factor up to [`MIX_MAX_SCALE`] that puts every
/// realised share within [`MIX_TOLERANCE`] of what was asked (the closest
/// scale when none does). An empty group is listed once, which lists nothing.
fn mix_repeats(groups: &[(usize, f32)]) -> Result<Vec<usize>> {
    let Some(((dataset, _), replays)) = groups.split_first() else {
        return Ok(Vec::new());
    };
    let mut total = 0.0f64;
    for &(_, share) in replays {
        if !(share > 0.0 && share < 1.0) {
            return Err(Error::Backend(format!("ChatFineTune: a replay share must be between 0 and 1, got {share}")));
        }
        total += f64::from(share);
    }
    if total >= 1.0 {
        return Err(Error::Backend(format!("ChatFineTune: the replay shares add up to {total}, leaving the dataset no draws")));
    }
    let shares: Vec<(usize, f64)> = std::iter::once((*dataset, 1.0 - total)).chain(replays.iter().map(|&(n, s)| (n, f64::from(s)))).collect();
    // Only a group with records is weighted; one without is listed once and adds nothing.
    let weights: Vec<Option<f64>> = shares.iter().map(|&(n, share)| (n > 0).then(|| share / n as f64)).collect();
    let Some(least) = weights.iter().flatten().copied().min_by(f64::total_cmp) else {
        return Ok(vec![1; groups.len()]);
    };
    let at_scale = |scale: usize| -> Vec<usize> { weights.iter().map(|w| w.map_or(1, |w| ((w / least) * scale as f64).round().max(1.0) as usize)).collect() };
    let deviation = |repeats: &[usize]| -> f64 {
        let rows: f64 = repeats.iter().zip(&shares).map(|(&r, &(n, _))| (r * n) as f64).sum();
        repeats.iter().zip(&shares).filter(|(_, &(n, _))| n > 0).map(|(&r, &(n, share))| ((r * n) as f64 / rows - share).abs()).fold(0.0, f64::max)
    };
    let mut best = (f64::INFINITY, Vec::new());
    for scale in 1..=MIX_MAX_SCALE {
        let repeats = at_scale(scale);
        let off = deviation(&repeats);
        if off <= MIX_TOLERANCE {
            return Ok(repeats);
        }
        if off < best.0 {
            best = (off, repeats);
        }
    }
    Ok(best.1)
}

/// AdamW weight decay of a LoRA fine-tune unless named: none. The adapter
/// starts at zero, so decay only pulls what it has learned back towards it.
pub(crate) const LORA_WEIGHT_DECAY: f32 = 0.0;

/// The warmup is this share of the steps, as the divisor: a twentieth.
const WARMUP_DIVISOR: u32 = 20;

/// A run's cooldown after its patience runs out, unless named, is this
/// share of its steps, as the divisor: a tenth.
const COOLDOWN_DIVISOR: u32 = 10;

pub(crate) fn fit_opts(steps: u32, block: u32, lr: f32, seed: u64) -> model::FitOpts {
    model::FitOpts {
        steps,
        batch_size: 1,
        block_size: block,
        lr,
        min_lr: lr / 10.0,
        warmup: (steps / WARMUP_DIVISOR).max(1),
        decay_iters: steps,
        weight_decay: LORA_WEIGHT_DECAY,
        grad_clip: 1.0,
        grad_accum: 1,
        // Held-out evidence is `score_chat`'s, measured on the served form;
        // the loop's own eval and early stop are off.
        eval_interval: 0,
        eval_batches: 0,
        seed,
        // The resume state is the checkpoint; no whole-model writes.
        checkpoint_secs: 0,
        mask_before: None,
        mask_per_line: false,
        align_to_lines: false,
        patience: 0,
        cooldown_steps: 0,
        adam: Default::default(),
    }
}

/// The content digest naming a base ([`brain_modelstore::fetch::base_digest`]:
/// the file's own, or a `transformers` directory's config and weights).
pub(crate) fn base_digest(path: &Path) -> Result<String> {
    brain_modelstore::fetch::base_digest(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

pub(crate) fn digest(path: &Path) -> Result<String> {
    brain_modelstore::fetch::file_digest(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}

pub(crate) fn utf8(path: &Path) -> Result<&str> {
    path.to_str().ok_or_else(|| Error::Backend(format!("{}: not a UTF-8 path", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fine-tune warms up over a twentieth of its steps, decays the rate
    /// over all of them, and decays no weights unless asked.
    #[test]
    fn the_recipe_warms_up_briefly_and_decays_no_weights_by_default() {
        let o = fit_opts(400, 256, 2e-4, 1);
        assert_eq!((o.warmup, o.decay_iters, o.weight_decay), (20, 400, 0.0));
        assert_eq!(fit_opts(10, 256, 2e-4, 1).warmup, 1);
    }

    /// A row holds the longest record, grows in powers of two from 64, and
    /// never passes the caller's cap - a record past it is refused by size.
    #[test]
    fn the_training_row_fits_the_longest_record_within_the_cap() {
        assert_eq!(block_for(10, None).unwrap(), 64);
        assert_eq!(block_for(65, None).unwrap(), 128);
        assert_eq!(block_for(100, Some(100)).unwrap(), 100);
        assert_eq!(block_for(70, Some(512)).unwrap(), 128);
        assert!(block_for(600, Some(512)).unwrap_err().to_string().contains("max_block"));
    }

    /// The share each group's rows make of the mix `repeats` lists.
    fn realised(groups: &[(usize, f32)], repeats: &[usize]) -> Vec<f64> {
        let rows: Vec<f64> = groups.iter().zip(repeats).map(|(&(n, _), &r)| (n * r) as f64).collect();
        let total: f64 = rows.iter().sum();
        rows.iter().map(|r| r / total).collect()
    }

    /// A replay set many times the dataset must not take every draw, and one
    /// much smaller must still be drawn its share: each group is listed as
    /// often as its share needs, to within a hundredth; several weighted
    /// groups each get their own share; no replay or no share leaves the
    /// plain union; and a share that leaves the dataset nothing is refused.
    #[test]
    fn a_replay_share_sets_how_often_each_group_is_listed() {
        // 100 target + 900 replay at a quarter: 27 x 100 = 2700 target rows beside 900.
        let large = [(100, 0.0), (900, 0.25)];
        let repeats = mix_repeats(&large).unwrap();
        assert_eq!(repeats, [27, 1]);
        // 100 target + 10 replay at a half: the replay is listed ten times.
        let scarce = [(100, 0.0), (10, 0.5)];
        assert_eq!(mix_repeats(&scarce).unwrap(), [1, 10]);
        // A ratio that rounds badly once is listed at a larger scale until it fits.
        let awkward = [(278, 0.0), (150, 0.25)];
        let repeats = mix_repeats(&awkward).unwrap();
        let shares = realised(&awkward, &repeats);
        assert!((shares[1] - 0.25).abs() <= MIX_TOLERANCE, "{repeats:?} -> {shares:?}");
        // Two weighted groups and the dataset each at their share.
        let three = [(278, 0.0), (60, 0.25), (93, 0.25)];
        let repeats = mix_repeats(&three).unwrap();
        let shares = realised(&three, &repeats);
        for (got, want) in shares.iter().zip([0.5, 0.25, 0.25]) {
            assert!((got - want).abs() <= MIX_TOLERANCE, "{repeats:?} -> {shares:?}");
        }
        // Nothing to weight is the plain union; an empty group lists nothing.
        assert_eq!(mix_repeats(&[(100, 0.0)]).unwrap(), [1]);
        assert_eq!(mix_repeats(&[(100, 0.0), (0, 0.25)]).unwrap(), [1, 1]);
        assert!(mix_repeats(&[]).unwrap().is_empty());
        for bad in [0.0, 1.0, -0.1, f32::NAN] {
            assert!(mix_repeats(&[(1, 0.0), (1, bad)]).unwrap_err().to_string().contains("replay share"));
        }
        assert!(mix_repeats(&[(1, 0.0), (1, 0.5), (1, 0.5)]).unwrap_err().to_string().contains("add up"));
    }
}

#[cfg(test)]
mod base_tier_tests {
    use super::base_tier;
    use qwen3::Dtype;

    #[test]
    fn the_narrowest_requested_base_wins() {
        assert_eq!(base_tier(false, false), Dtype::F32);
        assert_eq!(base_tier(true, false), Dtype::BF16);
        assert_eq!(base_tier(true, true), Dtype::I8);
        assert_eq!(base_tier(false, true), Dtype::I8);
    }
}
