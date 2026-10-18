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

use model::training::Capability;

use crate::{CancelToken, Device, Error, Result};

/// Where one run keeps its files, inside [`ChatFineTune::out_dir`].
const PREPARED_DIR: &str = "prepared";
const STATE_FILE: &str = "train.state";
const ADAPTER_FILE: &str = "adapter.safetensors";
const RECORD_FILE: &str = "training.json";

/// Training rows are at least this long, so a tiny dataset does not train on
/// a degenerate row.
const MIN_BLOCK: u32 = 64;

/// A LoRA fine-tune of a Qwen3 chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ChatFineTune {
    base: String,
    arch: String,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    held_out: Option<PathBuf>,
    held_out_text: Option<PathBuf>,
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
    seed: Option<u64>,
    max_block: Option<u32>,
    checkpoint_every: u32,
    eval_every: u32,
    patience: u32,
    keep_best: bool,
    evaluations_dir: Option<PathBuf>,
    cycle: u64,
    device: Device,
    bf16_base: bool,
    int8_base: bool,
    nf4_base: bool,
    thinking: bool,
    pack_answers: bool,
}

/// The architecture a [`ChatFineTune`] trains, with the tier its frozen base
/// is held at. The one place that names a trainer: `qwen3` and `qwen35` each
/// have their own checkpoint reader and LoRA trainer over the shared
/// `model::fit_controlled` loop.
enum ChatFamily {
    /// Int8 over bf16 over fp32.
    Qwen3(qwen3::Dtype),
    /// NF4 over bf16 over fp32 (the 27B trains on one card only at the first two).
    Qwen35(qwen35::base_source::BaseDtype),
}

/// A trained LoRA, whichever trainer produced it.
enum TrainedLora {
    Qwen3(qwen3::finetune::Trained),
    Qwen35(Box<qwen35::model::Qwen35>),
}

impl TrainedLora {
    fn save(&self, path: &str, card_id: &str, base_id: &str, dataset_id: Option<&str>, provenance: checkpoint::st::TrainingProvenance) -> std::io::Result<()> {
        match self {
            TrainedLora::Qwen3(t) => t.save_adapter_with_lineage(path, card_id, base_id, dataset_id, Some(provenance)),
            TrainedLora::Qwen35(m) => qwen35::lora::save_adapter_with_lineage(path, m, card_id, base_id, dataset_id, Some(provenance)),
        }
    }
}

impl ChatFamily {
    fn of(arch_id: &str, bf16: bool, int8: bool, nf4: bool) -> Result<ChatFamily> {
        match arch_id {
            "qwen3" if nf4 => Err(Error::Backend("ChatFineTune: nf4_base is a qwen35 tier; qwen3 holds its base at fp32, bf16 or int8".to_string())),
            "qwen3" => Ok(ChatFamily::Qwen3(if int8 {
                qwen3::Dtype::I8
            } else if bf16 {
                qwen3::Dtype::BF16
            } else {
                qwen3::Dtype::F32
            })),
            "qwen35" if int8 => Err(Error::Backend("ChatFineTune: int8_base is a qwen3 tier; qwen35 holds its base at fp32, bf16 or nf4".to_string())),
            "qwen35" => {
                use qwen35::base_source::BaseDtype;
                Ok(ChatFamily::Qwen35(if nf4 {
                    BaseDtype::Nf4
                } else if bf16 {
                    BaseDtype::Bf16
                } else {
                    BaseDtype::F32
                }))
            }
            other => Err(Error::Unsupported(model::training::Unsupported::new(other, Capability::ChatLoraFineTune))),
        }
    }

    /// Refuse what this family's trainer cannot do, before any file is read.
    fn refuse_unsupported(&self, run: &ChatFineTune) -> Result<()> {
        if let ChatFamily::Qwen35(_) = self {
            // Scoring a base against held-out records and writing an adapter per
            // evaluation are `qwen3::eval` and `qwen3` snapshot features.
            if run.held_out.is_some() || run.held_out_text.is_some() || run.evaluations_dir.is_some() {
                return Err(Error::Backend("ChatFineTune: qwen35 does not score held-out records or keep per-evaluation adapters yet; use .monitor() with .eval_every() for the loss curve".to_string()));
            }
        }
        Ok(())
    }

    /// The vocabulary the checkpoint at `open` was trained with.
    fn vocab(&self, open: &Path) -> Result<usize> {
        match self {
            ChatFamily::Qwen3(_) => Ok(qwen3::checkpoint_config(utf8(open)?).map_err(Error::Backend)?.vocab as usize),
            ChatFamily::Qwen35(_) => Ok(qwen35::base_source::open_base(open).map_err(Error::Backend)?.cfg.vocab as usize),
        }
    }

    /// The tier held-out scoring runs at; `None` where the family cannot score.
    fn scoring_tier(&self) -> Option<qwen3::Dtype> {
        match self {
            ChatFamily::Qwen3(tier) => Some(*tier),
            ChatFamily::Qwen35(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn train(&self, open: &Path, prepared_dir: &Path, opts: &model::FitOpts, rank: u32, alpha: f32, continue_from: Option<&Path>, control: model::FitControl<'_>) -> std::io::Result<(model::FitReport, TrainedLora)> {
        match self {
            ChatFamily::Qwen3(tier) => {
                let start = match continue_from {
                    Some(path) => qwen3::finetune::LoraStart::Continue(path.to_str().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "adapter path is not UTF-8"))?),
                    None => qwen3::finetune::LoraStart::Fresh,
                };
                let weights = open.to_str().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "checkpoint path is not UTF-8"))?;
                let (report, trained) = qwen3::finetune::finetune_lora_controlled(weights, prepared_dir, opts, rank, alpha, &start, control, *tier)?;
                Ok((report, TrainedLora::Qwen3(trained)))
            }
            ChatFamily::Qwen35(dtype) => {
                let run = qwen35::finetune::StreamedRun { dtype: *dtype, recompute: dtype.is_frozen_only(), long: qwen35::finetune::LongSeqOverrides::default() };
                let (report, trained) = qwen35::finetune::finetune_lora_controlled(open, prepared_dir, opts, rank, alpha, continue_from, &run, control)?;
                Ok((report, TrainedLora::Qwen35(Box::new(trained))))
            }
        }
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
            arch: "qwen3".to_string(),
            models_dir: None,
            dataset: None,
            held_out: None,
            held_out_text: None,
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
            seed: None,
            max_block: None,
            checkpoint_every: 0,
            eval_every: 0,
            patience: 0,
            keep_best: false,
            evaluations_dir: None,
            cycle: 0,
            device: Device::default(),
            bf16_base: false,
            int8_base: false,
            nf4_base: false,
            thinking: true,
            pack_answers: false,
        }
    }

    /// The architecture of the base (a `brain_arch` id; default `qwen3`).
    /// Looked up in the same registry as [`crate::study::DocumentStudy`]; an
    /// architecture without a chat fine-tuner is refused before anything is read.
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = id.into();
        self
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

    /// A second set of records to score base and tuned on, never trained on:
    /// held-out text of another kind than [`Self::held_out`] (a writer's own
    /// passages beside the dialogues it is asked as), reported apart in the
    /// outcome so a loss over the one is not drowned in the other.
    pub fn held_out_text(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out_text = Some(path.into());
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

    /// Write the adapter of every evaluation to `dir` as `step-<N>.safetensors`
    /// (N the steps trained), so the one exported - the best by monitoring
    /// loss, or the last - is not the only one a caller can choose from by
    /// some other measure. The directory is created. Needs
    /// [`Self::eval_every`].
    pub fn keep_evaluations(mut self, dir: impl Into<PathBuf>) -> Self {
        self.evaluations_dir = Some(dir.into());
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

    /// Seeds the fresh adapter's initialisation and the batch order. Unset, a
    /// run continuing a saved state takes that state's seed and any other
    /// draws a random one; [`ChatFineTuneOutcome::seed`] and the training
    /// record carry the seed used, and passing it reproduces the run.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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

    /// Train a record that supervises several assistant turns as ONE sequence
    /// ([`ChatSample::answers_packed`]) instead of one example per turn
    /// ([`ChatSample::answers_as_asked`]). Each turn's target is the same
    /// either way; packing renders the conversation once, so a long
    /// tool-using trajectory costs its length once instead of once per turn.
    /// A step's gradient is the mean over the supervised tokens of the rows it
    /// accumulates either way, so a token weighs the same; what changes is
    /// that a row is a whole conversation, so a step of `n` rows spans far
    /// more tokens and an epoch has fewer rows. A record with one supervised
    /// turn is unaffected.
    pub fn pack_answers(mut self, on: bool) -> Self {
        self.pack_answers = on;
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

    /// Hold the frozen base as 4-bit NormalFloat blocks (QLoRA): the 27B
    /// `qwen35` base then trains in 44 GiB instead of 80. `qwen35` only; the
    /// adapters and optimiser stay fp32. Takes precedence over
    /// [`Self::bf16_base`].
    pub fn nf4_base(mut self, on: bool) -> Self {
        self.nf4_base = on;
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
        let arch = crate::study::arch(&self.arch)?;
        arch.require(Capability::ChatLoraFineTune)?;
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ChatFineTune: no out directory; call .out_dir(path)".to_string()))?;
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let open = qwen3::store_checkpoint_path(&weights, &model_dir);
        let weights_str = utf8(&open)?;
        let family = ChatFamily::of(arch.id, self.bf16_base, self.int8_base, self.nf4_base)?;
        family.refuse_unsupported(self)?;

        // Every file is checked against the base's own template before a
        // device is claimed.
        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: self.pack_answers };
        let train_samples = asked.read(dataset)?;
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
        let held_out_text = self.held_out_text.as_deref().map(|path| asked.read(path)).transpose()?;
        let monitor = self.monitor.as_deref().map(|path| asked.read(path)).transpose()?;
        if self.eval_every == 0 && self.evaluations_dir.is_some() {
            return Err(Error::Backend("ChatFineTune: keeping the evaluations needs eval_every > 0".to_string()));
        }
        if self.eval_every == 0 && (self.patience > 0 || self.keep_best) {
            return Err(Error::Backend("ChatFineTune: selecting on the monitoring loss needs eval_every > 0".to_string()));
        }

        let (rank, alpha, parent) = lora_shape(self.continue_from.as_deref(), self.rank, self.alpha)?;

        let Mix { rows: training, replay_records, repeats } = mix_rows(train_samples, replay_samples, self.replay_share, weighted_samples)?;
        // The packed validation split is what a periodic evaluation scores:
        // the monitoring set, else the held-out set, else (unread, when
        // nothing is evaluated) the training set.
        let val = monitor.as_deref().or(held_out.as_deref()).unwrap_or(&training[..]);
        let vocab = family.vocab(&open)?;
        let prepared_dir = out_dir.join(PREPARED_DIR);
        let prepared = data::chat::prepare_chat_samples(&training, val, &tok, &tmpl, render, vocab, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        let block = block_for(prepared.longest_example, self.max_block)?;

        crate::device::resolve(&self.device)?;
        let scoring_tier = family.scoring_tier();
        let base_score = held_out.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, None, &tok, &tmpl, records, block, tier, render));
        let base_text_score = held_out_text.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, None, &tok, &tmpl, records, block, tier, render));

        if self.grad_accum == 0 {
            return Err(Error::Backend("ChatFineTune: grad_accum must be at least 1".to_string()));
        }
        let state_path = out_dir.join(STATE_FILE);
        let seed = resolve_seed(self.seed, Some(&state_path))?;
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
            ..fit_opts(self.steps, block, self.lr, seed)
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
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress { step: s.step, steps: s.steps, loss: s.loss, lr: s.lr });
            !cancel.is_cancelled()
        };
        let snapshot_card = format!("{base_id}:local:chat:step");
        let snapshots = match &self.evaluations_dir {
            Some(dir) => {
                std::fs::create_dir_all(dir).map_err(|e| Error::Backend(format!("{}: {e}", dir.display())))?;
                Some(model::AdapterSnapshots { dir, card_id: &snapshot_card, base_id: &base_id, family: arch.family, rank, alpha, targets: Vec::new() })
            }
            None => None,
        };
        let control = model::FitControl { on_step: Some(&mut on_step), state: Some(&state_path), state_every: self.checkpoint_every, identity, snapshots };
        let (report, trained) = family.train(&open, &prepared_dir, &opts, rank, alpha, self.continue_from.as_deref(), control).map_err(|e| Error::Backend(format!("training: {e}")))?;

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
            seed,
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
            base_text_score,
            tuned_text_score: None,
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
            seed,
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
        trained.save(utf8(&adapter_path)?, &adapter_id, &base_id, Some(&dataset_id), provenance)
            .map_err(|e| Error::Backend(format!("{}: saving the adapter: {e}", adapter_path.display())))?;
        drop(trained);
        outcome.adapter_digest = Some(digest(&adapter_path)?);
        let adapter_str = utf8(&adapter_path)?;
        outcome.tuned_score = held_out.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, Some(adapter_str), &tok, &tmpl, records, block, tier, render));
        outcome.tuned_text_score = held_out_text.as_ref().zip(scoring_tier).map(|(records, tier)| score_records(weights_str, Some(adapter_str), &tok, &tmpl, records, block, tier, render));
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
    /// The seed of the fresh adapter's initialisation and the batch order:
    /// the one asked for, the one a continued state was saved with, or the
    /// one drawn. Passing it to [`ChatFineTune::seed`] reproduces the run.
    pub seed: u64,
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
    /// The base on the [`ChatFineTune::held_out_text`] records.
    pub base_text_score: Option<HeldOutScore>,
    /// Base plus the new adapter on the same records.
    pub tuned_text_score: Option<HeldOutScore>,
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
            "seed": self.seed,
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
            "base_text_score": score(&self.base_text_score),
            "tuned_text_score": score(&self.tuned_text_score),
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
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let records = Asked { tok: &tok, tmpl: &tmpl, no_think: None, render: data::chat::RenderOpts::default(), pack: false }.read(held_out)?;
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

/// Per-token log-probabilities of `targets` given `tokens` under `base`
/// (optionally with `adapter` applied at run time, exactly as serving it
/// would), with every logit multiplied by `logit_scale` first: `1/T` scores
/// under the distribution a sampler at temperature `T` drew from, `1.0` the
/// plain model. `targets[i]` is the token that follows `tokens[i]`; an
/// [`crate::IGNORE_TARGET`] target scores `0.0`. Pass no `adapter` to score under the
/// frozen reference of a preference or policy-gradient objective: the same
/// base, adapter disabled.
///
/// `base` is resolved like [`ChatFineTune::from_pretrained`]'s. The scores are
/// the ones a trainer's `batch_token_logprobs` reads off its own forward, from
/// the full `logits_all` pass, so this is the always-correct oracle to compare
/// a training run against, not the fast path inside one.
pub fn score_tokens(base: &str, adapter: Option<&Path>, tokens: &[u32], targets: &[u32], logit_scale: f32, device: &Device) -> Result<Vec<f32>> {
    if tokens.is_empty() || tokens.len() != targets.len() {
        return Err(Error::MissingArgument(format!("score_tokens: need as many targets as tokens (got {} tokens, {} targets) and at least one", tokens.len(), targets.len())));
    }
    if !(logit_scale.is_finite() && logit_scale > 0.0) {
        return Err(Error::MissingArgument(format!("score_tokens: the logit scale must be finite and positive, got {logit_scale}")));
    }
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    let open = qwen3::store_checkpoint_path(&weights, &model_dir);
    let weights_str = utf8(&open)?;
    let cfg = qwen3::checkpoint_config(weights_str).map_err(Error::Backend)?;
    if let Some(&bad) = tokens.iter().chain(targets.iter().filter(|&&t| t != model::IGNORE)).find(|&&t| t >= cfg.vocab) {
        return Err(Error::Backend(format!("score_tokens: token {bad} is outside the model's vocabulary of {}", cfg.vocab)));
    }
    crate::device::resolve(device)?;
    let mut scored = qwen3::Qwen::load_inference(weights_str, 1, tokens.len() as u32);
    if let Some(path) = adapter {
        scored.attach_adapter(utf8(path)?).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
    }
    Ok(model::logprobs::token_logprobs_scaled(&scored, tokens, targets, logit_scale).expect("a language model has logits"))
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
pub(crate) struct Asked<'a> {
    pub tok: &'a dyn data::tokenizer::Tokenizer,
    pub tmpl: &'a ChatTemplate,
    pub no_think: Option<&'a str>,
    pub render: data::chat::RenderOpts,
    /// A record of several supervised answers is one packed example.
    pub pack: bool,
}

impl Asked<'_> {
    /// The samples of `path` as they are trained on, each checked to encode
    /// against the template and to supervise something. A dialogue is one
    /// example per answer, each as the model is asked for it
    /// ([`ChatSample::answers_as_asked`]), for a model trained for no-think
    /// mode and, in thinking mode, for a record whose supervised turns call
    /// tools or that supervises only its last message after tool turns
    /// (whole-record masking refuses those); any other record keeps its shape,
    /// its answers following the no-think block when there is one.
    pub(crate) fn read(&self, path: &Path) -> Result<Vec<ChatSample>> {
        let summary = crate::validate_chat_dataset(path).map_err(Error::Backend)?;
        if summary.trained_messages == 0 {
            return Err(Error::Backend(format!("{}: holds no supervised turns (`\"train\": true`)", path.display())));
        }
        let records = ChatSample::from_jsonl(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
        let mut samples = Vec::with_capacity(records.len());
        for (at, record) in records.iter().enumerate() {
            let invalid = |e: &dyn std::fmt::Display| Error::Backend(format!("{}: record {} cannot be encoded for training: {e}", path.display(), at + 1));
            // Whole-record masking is exact for plain answers in thinking mode
            // and refuses tool turns; everywhere else each answer is asked for,
            // and a record supervising only its last message has one boundary.
            if self.pack && record.messages.iter().filter(|m| m.train).count() > 1 {
                if let Some(packed) = record.answers_packed(self.tmpl, self.no_think.is_none()).map_err(|e| invalid(&e))? {
                    samples.push(packed);
                    continue;
                }
            }
            let per_answer = if self.no_think.is_some() || record.supervises_tool_calls() {
                record.answers_as_asked(self.tmpl, self.no_think.is_none()).map_err(|e| invalid(&e))?
            } else if record.supervises_only_final_turn() && record.context_holds_tool_turns() {
                record.final_turn_as_asked(self.tmpl, true).map_err(|e| invalid(&e))?.map(|asked| vec![asked])
            } else {
                None
            };
            match (per_answer, self.no_think) {
                (Some(asked), _) => samples.extend(asked),
                (None, None) => samples.push(record.clone()),
                (None, Some(block)) => samples.push(record.answering_without_thinking(block)),
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

pub(crate) fn tokenizer_and_template(model_dir: &Path) -> Result<(QwenBpe, ChatTemplate)> {
    let tmpl = ChatTemplate::from_model_dir(model_dir).map_err(|e| Error::Backend(format!("{}: {e}", model_dir.display())))?;
    let tok_path = model_dir.join("tokenizer.json");
    let tok = QwenBpe::from_file(utf8(&tok_path)?).map_err(|e| Error::Backend(format!("{}: {e}", tok_path.display())))?;
    Ok((tok, tmpl))
}

/// The training row for a dataset whose longest record is `longest` tokens:
/// exactly that long (at least [`MIN_BLOCK`]), capped at `max`. A record
/// longer than the cap is refused, since training on a row that cannot hold
/// it would supervise nothing of it. The row is not rounded up: attention
/// and every activation scale with it, so padding a 5646-token record to the
/// next power of two trains 8192 rows' worth of work for 5646 tokens of data.
pub(crate) fn block_for(longest: usize, max: Option<u32>) -> Result<u32> {
    if let Some(max) = max {
        if longest > max as usize {
            return Err(Error::Backend(format!("the longest record is {longest} tokens, past max_block {max}; raise max_block or shorten the record")));
        }
    }
    let block = u32::try_from(longest).map_err(|_| Error::Backend(format!("the longest record is {longest} tokens, past the largest supported row")))?.max(MIN_BLOCK);
    Ok(max.map_or(block, |m| block.min(m)))
}

/// The rows a fine-tune trains on, and how they were mixed.
pub(crate) struct Mix {
    /// Every row, each group listed as many times as its share needs.
    pub rows: Vec<ChatSample>,
    /// Records of the replay files and the weighted files, each counted once.
    pub replay_records: usize,
    /// How many times each group was listed, the dataset first.
    pub repeats: Vec<usize>,
}

/// The groups of the mix, the dataset first: the replay files at their share
/// when one is set, else in the dataset's own group as the plain union they
/// always were; then each weighted file.
pub(crate) fn mix_rows(mut train_samples: Vec<ChatSample>, mut replay_samples: Vec<ChatSample>, replay_share: Option<f32>, weighted_samples: Vec<(Vec<ChatSample>, f32)>) -> Result<Mix> {
    let replay_records = replay_samples.len() + weighted_samples.iter().map(|(s, _)| s.len()).sum::<usize>();
    let mut groups: Vec<(Vec<ChatSample>, f32)> = Vec::new();
    match replay_share {
        Some(share) => groups.push((std::mem::take(&mut replay_samples), share)),
        None => train_samples.append(&mut replay_samples),
    }
    groups.extend(weighted_samples);
    let shares: Vec<(usize, f32)> = std::iter::once((train_samples.len(), 0.0)).chain(groups.iter().map(|(s, share)| (s.len(), *share))).collect();
    let repeats = mix_repeats(&shares)?;
    let mut rows = Vec::new();
    for (samples, &times) in std::iter::once(&train_samples).chain(groups.iter().map(|(s, _)| s)).zip(&repeats) {
        for _ in 0..times {
            rows.extend(samples.iter().cloned());
        }
    }
    Ok(Mix { rows, replay_records, repeats })
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

/// The seed a run trains with: the one asked for, else the one the saved
/// state at `state` was written with (the state only resumes under the
/// options it was saved with, the seed among them), else a random draw.
pub(crate) fn resolve_seed(requested: Option<u64>, state: Option<&Path>) -> Result<u64> {
    if let Some(seed) = requested {
        return Ok(seed);
    }
    let saved = match state {
        Some(path) => model::train::saved_seed(path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?,
        None => None,
    };
    Ok(saved.unwrap_or_else(crate::random_seed))
}

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

    /// A row is exactly as long as the longest record (a short dataset gets
    /// the minimum row), never longer than the caller's cap, and a record
    /// past the cap is refused by size, never truncated. A power-of-two row
    /// would pad a 5646-token record to 8192 rows.
    #[test]
    fn the_training_row_is_the_longest_record_within_the_cap() {
        assert_eq!(block_for(10, None).unwrap(), MIN_BLOCK);
        assert_eq!(block_for(65, None).unwrap(), 65);
        assert_eq!(block_for(5646, None).unwrap(), 5646);
        assert_eq!(block_for(5646, Some(5646)).unwrap(), 5646);
        assert_eq!(block_for(70, Some(512)).unwrap(), 70);
        assert_eq!(block_for(10, Some(32)).unwrap(), 32, "the cap wins over the minimum row");
        let refused = block_for(5647, Some(5646)).unwrap_err().to_string();
        assert!(refused.contains("max_block") && refused.contains("5647"), "{refused}");
    }

    /// The samples `Asked` reads from a dataset of `lines`, in the mode asked.
    fn read_lines(lines: &[&str], no_think: bool) -> Result<Vec<ChatSample>> {
        read_lines_packing(lines, no_think, false)
    }

    fn read_lines_packing(lines: &[&str], no_think: bool, pack: bool) -> Result<Vec<ChatSample>> {
        let tmpl = ChatTemplate::compile(&std::fs::read_to_string(brain_testutil::testdata("qwen3/chat_template.jinja")).unwrap()).unwrap();
        let chars: String = (32u8..127).map(char::from).chain(['\n']).collect();
        let tok = data::tokenizer::CharTokenizer::from_corpus(&chars);
        // One file per call: tests run in parallel and each removes its own.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("brain-asked-{}-{call}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        let no_think_block = no_think.then(|| tmpl.no_think_block().unwrap());
        let render = data::chat::RenderOpts { keep_reasoning: no_think };
        let read = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think_block.as_deref(), render, pack }.read(&path);
        std::fs::remove_file(&path).unwrap();
        read
    }

    const ANSWER: &str = r#"{"messages":[{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true}]}"#;
    const DIALOGUE: &str = r#"{"messages":[{"role":"user","content":"q1","train":false},{"role":"assistant","content":"a1","train":true},{"role":"user","content":"q2","train":false},{"role":"assistant","content":"a2","train":true}]}"#;
    const TOOL_CONTEXT_FINAL_ANSWER: &str = r#"{"messages":[{"role":"user","content":"w","train":false},{"role":"assistant","content":"","train":false,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","content":"r","tool_call_id":"c1","train":false},{"role":"assistant","content":"done","train":true}]}"#;
    const TOOL_CONVERSATION: &str = r#"{"messages":[{"role":"user","content":"w","train":false},{"role":"assistant","content":"","train":true,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","content":"r","tool_call_id":"c1","train":false},{"role":"assistant","content":"done","train":true}]}"#;

    /// A conversation with a supervised tool call trains turn by turn in both
    /// thinking modes, while a dataset of plain answers keeps the shape it has
    /// always had in thinking mode: one example per record, encoded whole.
    #[test]
    fn tool_conversations_train_per_turn_and_plain_ones_as_before() {
        for no_think in [false, true] {
            let samples = read_lines(&[TOOL_CONVERSATION], no_think).unwrap();
            assert_eq!(samples.len(), 2, "no_think={no_think}");
            assert!(samples.iter().all(|s| s.rendered.is_some()));
        }
        let plain = read_lines(&[ANSWER], false).unwrap();
        assert_eq!(plain.len(), 1);
        assert!(plain[0].rendered.is_none(), "thinking-mode plain answers are encoded as the record");
        assert_eq!(read_lines(&[DIALOGUE], true).unwrap().len(), 2);
    }

    /// Packing turns a record of several supervised answers into one example
    /// in either thinking mode, and leaves one-answer records as they are.
    #[test]
    fn packing_makes_one_example_of_a_conversation_of_answers() {
        for no_think in [false, true] {
            assert_eq!(read_lines_packing(&[TOOL_CONVERSATION], no_think, true).unwrap().len(), 1, "no_think={no_think}");
            assert_eq!(read_lines_packing(&[DIALOGUE], no_think, true).unwrap().len(), 1, "no_think={no_think}");
            assert_eq!(read_lines_packing(&[TOOL_CONTEXT_FINAL_ANSWER], no_think, true).unwrap().len(), 1);
        }
        assert!(read_lines_packing(&[ANSWER], false, true).unwrap()[0].rendered.is_none());
    }

    /// A record that supervises only its final answer after a tool call and
    /// its result is one example in either thinking mode, not a refusal.
    #[test]
    fn a_final_answer_after_unsupervised_tool_turns_is_one_example() {
        for no_think in [false, true] {
            let samples = read_lines(&[TOOL_CONTEXT_FINAL_ANSWER], no_think).unwrap();
            assert_eq!(samples.len(), 1, "no_think={no_think}");
            assert!(samples[0].rendered.is_some());
        }
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
    use super::ChatFamily;
    use qwen35::base_source::BaseDtype;
    use qwen3::Dtype;

    fn qwen3_tier(bf16: bool, int8: bool) -> Dtype {
        match ChatFamily::of("qwen3", bf16, int8, false).unwrap() {
            ChatFamily::Qwen3(tier) => tier,
            ChatFamily::Qwen35(_) => unreachable!(),
        }
    }

    fn qwen35_tier(bf16: bool, nf4: bool) -> BaseDtype {
        match ChatFamily::of("qwen35", bf16, false, nf4).unwrap() {
            ChatFamily::Qwen35(tier) => tier,
            ChatFamily::Qwen3(_) => unreachable!(),
        }
    }

    #[test]
    fn the_narrowest_requested_base_wins() {
        assert_eq!(qwen3_tier(false, false), Dtype::F32);
        assert_eq!(qwen3_tier(true, false), Dtype::BF16);
        assert_eq!(qwen3_tier(true, true), Dtype::I8);
        assert_eq!(qwen3_tier(false, true), Dtype::I8);
        assert_eq!(qwen35_tier(false, false), BaseDtype::F32);
        assert_eq!(qwen35_tier(true, false), BaseDtype::Bf16);
        assert_eq!(qwen35_tier(true, true), BaseDtype::Nf4);
    }

    #[test]
    fn a_tier_the_architecture_lacks_is_refused() {
        assert!(ChatFamily::of("qwen3", false, false, true).is_err());
        assert!(ChatFamily::of("qwen35", false, true, false).is_err());
    }
}
