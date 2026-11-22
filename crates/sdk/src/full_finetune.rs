// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full-parameter fine-tuning and adapter
// merging of open-weight chat models for its clients. If your team needs
// expertise in consolidating what a model learned into its own weights, you
// can procure our services by sending an email to info@swedishembedded.com.

//! [`ChatFullFineTune`]: full-parameter fine-tuning of a chat model on a chat
//! dataset, and [`merge_adapter`]: folding a LoRA adapter into the weights it
//! targets. Both produce a checkpoint directory (for qwen3 in the
//! `transformers` layout: one fp32 `model.safetensors`, `config.json` and the
//! base's tokenizer files; qwen35, ouro and the other families without a
//! `transformers` writer write their brain-format weights as
//! `model.brain.safetensors`, the file the model store opens a base by, beside
//! the tokenizer files) that is a base like any other: [`crate::ChatFineTune`]
//! trains an adapter on it, `brain serve` serves it, and either can be asked
//! to fine-tune it again.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::ChatFullFineTune::from_pretrained("/models/qwen3-0.6b/model.brain.safetensors")
//!     .dataset("train.jsonl")
//!     .out_dir("runs/consolidate-1")
//!     .steps(200)
//!     .lr(1e-5)
//!     .run()?;
//! if let Some(checkpoint) = outcome.checkpoint {
//!     println!("trained base: {}", checkpoint.display());
//! }
//! # Ok(()) }
//! ```
//!
//! **What trains and where it lives.** Every weight trains, in fp32 (the one
//! precision the trainable build runs in), with the AdamW moments and the
//! master weights held in system RAM and only the weights and their gradients
//! on the card: the architecture's `model::finetune::ChatTrainer` (for qwen3
//! the same build `brain qwen3 finetune` runs) over `model::fit_controlled`,
//! the one training loop. That is two copies of the parameters on one card (a 0.6B model needs
//! about 4.4 GiB for them), which is the limit of the method: an 8B model
//! needs 61 GiB for the pair and does not fit two 24 GiB cards. The dataset,
//! the replay files and their shares are [`crate::ChatFineTune`]'s, validated
//! against the base's own template before a device is claimed.
//!
//! **Cancellation and resume.** A run whose [`crate::CancelToken`] fires stops at
//! the next step boundary and writes no checkpoint. With a
//! [`ChatFullFineTune::resume_state`] it first writes its exact training state
//! there (weights, AdamW moments, schedule position, batch order; about three
//! times the model), and the same fine-tune run again continues it; without
//! one, a cancelled full fine-tune starts over.

use std::path::{Path, PathBuf};

use crate::finetune::{base_digest, block_for, digest, fit_opts, mix_rows, resolve_seed, tokenizer_and_template, Asked, Mix};
use model::training::Capability;
use model::zero::Sharding;

use crate::{CancelToken, Device, Error, FineTuneProgress, FineTuneStatus, Result};

/// The training format the run writes before it is exported, and removes.
const TUNED_FILE: &str = "tuned.brain.safetensors";
/// The packed dataset, inside the out directory.
const PREPARED_DIR: &str = "prepared";
/// The checkpoint directory, inside the out directory.
const CHECKPOINT_DIR: &str = "checkpoint";
/// The weights file of a checkpoint directory.
const WEIGHTS_FILE: &str = "model.safetensors";
/// The training record, inside the out directory beside the checkpoint.
const RECORD_FILE: &str = "training.json";

/// AdamW weight decay of a full fine-tune unless named: none, so the trained
/// weights are pulled towards the data and nowhere else.
const FULL_WEIGHT_DECAY: f32 = 0.0;

/// A full-parameter fine-tune of a Qwen3 chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ChatFullFineTune {
    base: String,
    /// The architecture named by [`Self::arch`]; unnamed, the base's own config tells it (`qwen3` where it does not).
    arch: Option<String>,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    replay: Vec<PathBuf>,
    replay_share: Option<f32>,
    weighted_replay: Vec<(PathBuf, f32)>,
    grad_accum: u32,
    strict_updates: bool,
    exit_entropy: Option<f32>,
    stages: u32,
    native_sparse: Option<model::NativeSparse>,
    out_dir: Option<PathBuf>,
    weight_decay: f32,
    steps: u32,
    lr: f32,
    seed: Option<u64>,
    max_block: Option<u32>,
    device: Device,
    /// The caller has scoped this thread to its device ([`ChatFullFineTune::placed`]).
    placed: bool,
    thinking: bool,
    precision: model::TrainPrecision,
    resume_state: Option<PathBuf>,
    checkpoint_every: u32,
    /// This run as one rank of a data-parallel group.
    parallel: Option<crate::DataParallel>,
}

impl ChatFullFineTune {
    /// A full fine-tune of `base`: a checkpoint file, a checkpoint directory
    /// (a previous full fine-tune's or merge's, or a `transformers` one), or
    /// a `vendor/repo` reference in the model store. The tokenizer and chat
    /// template are read from the checkpoint's directory.
    pub fn from_pretrained(base: impl Into<String>) -> ChatFullFineTune {
        ChatFullFineTune {
            base: base.into(),
            arch: None,
            models_dir: None,
            dataset: None,
            replay: Vec::new(),
            replay_share: None,
            weighted_replay: Vec::new(),
            grad_accum: 1,
            strict_updates: false,
            exit_entropy: None,
            stages: 0,
            native_sparse: None,
            out_dir: None,
            weight_decay: FULL_WEIGHT_DECAY,
            steps: 100,
            lr: 1e-5,
            seed: None,
            max_block: None,
            device: Device::default(),
            placed: false,
            thinking: true,
            precision: model::TrainPrecision::default(),
            resume_state: None,
            checkpoint_every: 0,
            parallel: None,
        }
    }

    /// Exact resume through the training-state file `path`: a run stopped at a step boundary (cancelled, or its
    /// process killed after a periodic save) leaves the optimised weights, their AdamW moments, the schedule position
    /// and the batch order there, and running the same fine-tune again with the same file continues it to the
    /// checkpoint an uninterrupted run would have written. A state written by a run over other data, another base or
    /// other options is refused, never continued. Removed once the checkpoint is exported. Unset (the default), a
    /// stopped run starts over.
    /// Train as one rank of a data-parallel group ([`crate::DataParallel`]): every rank runs this fine-tune with the
    /// same settings and the same [`Self::seed`], each on its own card, trains on its own shard of the records, and
    /// every step's gradients are reduced over the group; the group's [`crate::DataParallel::sharding`] decides
    /// whether the optimizer state (and the reduced gradients) are sharded across the ranks instead of replicated.
    /// Rank 0 alone writes the checkpoint.
    pub fn data_parallel(mut self, rank: crate::DataParallel) -> Self {
        self.parallel = Some(rank);
        self
    }

    pub fn resume_state(mut self, path: impl Into<PathBuf>) -> Self {
        self.resume_state = Some(path.into());
        self
    }

    /// Also save the [`Self::resume_state`] every this-many steps (default 0: only when the run is stopped), so a
    /// process that dies loses at most this many.
    pub fn checkpoint_every(mut self, steps: u32) -> Self {
        self.checkpoint_every = steps;
        self
    }

    /// What the run stores each trainable parameter as: the format of the AdamW state (f32, bf16, int8 or logarithmic
    /// 8-bit blocks, held in host memory here), whether the planner may narrow it, and so on - one typed value, parsed
    /// once from `state=int8,sr=on,...` by [`model::TrainPrecision::parse`]. The default is fp32 everywhere. A choice
    /// the architecture does not run is refused by parameter name before anything is loaded.
    pub fn train_precision(mut self, precision: model::TrainPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// The architecture of the base (a `brain_arch` id). Unnamed, the base's own `config.json` tells it, as
    /// [`crate::ChatFineTune`]'s does, and `qwen3` is the architecture of a checkpoint with none.
    /// Looked up in the same registry as [`crate::study::DocumentStudy`]; an
    /// architecture without a chat fine-tuner is refused before anything is read.
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = Some(id.into());
        self
    }

    /// The models directory a `vendor/repo` base resolves in.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The chat dataset to train on (required).
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// Mix every record of another chat dataset into training; see
    /// [`crate::ChatFineTune::replay`].
    pub fn replay(mut self, path: impl Into<PathBuf>) -> Self {
        self.replay.push(path.into());
        self
    }

    /// The share of the draws the [`Self::replay`] files take in all; see
    /// [`crate::ChatFineTune::replay_share`].
    pub fn replay_share(mut self, share: f32) -> Self {
        self.replay_share = Some(share);
        self
    }

    /// Mix another chat dataset into training at `share` of the draws of its
    /// own; see [`crate::ChatFineTune::replay_at`].
    pub fn replay_at(mut self, path: impl Into<PathBuf>, share: f32) -> Self {
        self.weighted_replay.push((path.into(), share));
        self
    }

    /// Micro-batches summed into each optimizer step (default 1).
    /// Abort the run at the first optimiser step whose loss or gradient is not finite, instead of skipping
    /// it and going on (the default skips up to two in a row and aborts on the third). The weights are left as
    /// they were either way: a non-finite value never reaches them.
    pub fn strict_updates(mut self, strict: bool) -> Self {
        self.strict_updates = strict;
        self
    }

    /// The weight of the exit-distribution entropy in the objective of a family with exit gates (a looped decoder):
    /// `loss - beta * H(p)` over the supervised positions, which keeps the exit mass spread over the depths. Default
    /// [`FINE_TUNE_EXIT_ENTROPY`] for such a family, 0 for every other; `0` turns it off, and a non-zero weight is
    /// refused by a family without exit gates.
    pub fn exit_entropy(mut self, beta: f32) -> Self {
        self.exit_entropy = Some(beta);
        self
    }

    pub fn grad_accum(mut self, micro_batches: u32) -> Self {
        self.grad_accum = micro_batches;
        self
    }

    /// Cut the model into exactly this many pipeline stages (default 0: one stage when a device holds it, else the
    /// fewest that fit the selected cards). An architecture whose trainer lays itself out refuses a count by name.
    pub fn stages(mut self, stages: u32) -> Self {
        self.stages = stages;
        self
    }

    /// Train the global-attention layers with native sparse attention (N-LSA): each query reads its selection of
    /// `cfg` (window, sinks, clustered long-range keys) only. Architectures that declare support take it; the others
    /// are refused by name, and so is a device without workgroup-barrier kernels (use a GPU or `software`).
    pub fn native_sparse_attention(mut self, cfg: model::lsa::LsaConfig, long_range: bool) -> Self {
        self.native_sparse = Some(model::NativeSparse { cfg, long_range });
        self
    }

    /// Where the checkpoint directory and the packed dataset go (required).
    /// One directory per run.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// AdamW's decoupled weight decay (default 0).
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Optimizer steps (default 100). The first twentieth warms the rate up,
    /// and it decays to a tenth over all of them.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// The peak learning rate (default 1e-5: every weight moves, so the rate
    /// is two orders below a LoRA's).
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// The seed of the batch order (default: a random draw).
    /// [`ChatFullFineTuneOutcome::seed`] and the training record carry the
    /// seed used, and passing it reproduces the run.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The longest training row in tokens; a record past it is refused
    /// (default: exactly the longest record, at least a small minimum row).
    pub fn max_block(mut self, tokens: u32) -> Self {
        self.max_block = Some(tokens);
        self
    }

    /// The device the run is placed on (default: the automatic placement).
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Run on the device this thread is already scoped to (`gpu_core::devices::with_gpu`, or the host tier), the
    /// way a residency-managed workload is: the process owns the device selection and other models are being
    /// served from it, so the run neither resolves [`Self::device`] nor publishes a selection of its own.
    pub fn placed(mut self, on: bool) -> Self {
        self.placed = on;
        self
    }

    /// Whether the model is trained to reason before it answers (default
    /// true); `false` trains it for no-think mode, as
    /// [`crate::ChatFineTune::thinking`] does.
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// What this run asks of a card when its training rows are at most `block` tokens: the figure a scheduler
    /// reserves before the run is placed. Every parameter trains in fp32 with the AdamW moments and the master
    /// weights in host memory, so the card holds the weights and their gradients (8 bytes a parameter) and one
    /// step's activations, with the margin of [`crate::ChatFineTune::footprint`]. It reads the checkpoint's headers
    /// only; like that figure it is a model, not a measurement, and the run's own planner refuses what does not fit.
    /// [`Self::sharded_footprint`] is the same figure for one rank of a sharded group.
    ///
    /// # Errors
    /// The base is not found or is of an architecture without a chat trainer.
    pub fn footprint(&self, block: u32) -> Result<crate::TrainFootprint> {
        self.sharded_footprint(block, Sharding::Replicated, 1)
    }

    /// What one rank of a `world`-rank data-parallel group that shards `sharding` asks of its card, modelled as
    /// [`Self::footprint`] is (a model, not a measurement). Per rank, with `P` parameters and `S = P / world`:
    ///
    /// * `replicated`: the fp32 weights and gradients, `8 P`, the optimizer state in host memory (any world);
    /// * `optimizer`, `gradients`: `8 P` as well (the backward writes every gradient into the model's own buffers),
    ///   plus this rank's slice of the f32 master, both moments and the reduced gradient on the card (`16 S`; with
    ///   `moments=host` one slice buffer instead) and the two staging buffers a step packs units through;
    /// * `parameters` (a layer-streamed resident build): this rank's slices of the master, the moments and the
    ///   stored value (`12 S` plus the value's encoding), the window of gathered layers with their gradients, the
    ///   largest table gathered, and one transient gradient slice.
    ///
    /// Each adds one step's activations, the margin and the device context.
    ///
    /// # Errors
    /// The base is not found or is of an architecture without a chat trainer; a sharded stage over fewer than two
    /// ranks.
    pub fn sharded_footprint(&self, block: u32, sharding: Sharding, world: usize) -> Result<crate::TrainFootprint> {
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, _) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let chat = chat_entry(self.arch_of(&model_dir))?;
        let open = (chat.checkpoint_path)(&weights, &model_dir);
        let (footprint, _) = full_footprint(&open, block, true)?;
        let (modelled, terms) = rank_card_bytes(&footprint.trainable, footprint.activation_bytes, sharding, world, &self.precision)?;
        let device_bytes = model::memplan::train_reservation(modelled, None);
        let reason = match sharding {
            Sharding::Replicated => "full fine-tune: fp32 weights and gradients on the card, AdamW moments and master weights in host memory".to_string(),
            Sharding::Optimizer | Sharding::Gradients => format!("full fine-tune, one of {world} ranks sharding the optimizer state ({sharding:?}): fp32 weights and gradients and this rank's slice of the master and moments on the card"),
            Sharding::Parameters => format!("full fine-tune, one of {world} ranks sharding the parameters: this rank's slices and the layers it gathers on the card"),
        };
        Ok(crate::TrainFootprint {
            device_bytes,
            breakdown: format!("{terms} + activations {:.2} GiB + {}% margin + {:.2} GiB device context = {:.2} GiB", gib(footprint.activation_bytes), model::memplan::TRAIN_MARGIN_PERCENT, gib(model::memplan::TRAIN_DEVICE_CONTEXT_BYTES), gib(device_bytes)),
            base_tier: "f32".to_string(),
            reason,
        })
    }

    /// How a full fine-tune over `world` ranks on cards of `card_bytes` each (`None`: not known, nothing is refused
    /// for size) shards, and what each rank then asks of its card ([`Self::sharded_footprint`]).
    ///
    /// `asked` is taken as it is: refused when it needs more than the card holds, when the precision cannot shard
    /// that way ([`Sharding::admits`]), or when a sharded stage has fewer than two ranks. Unasked, the run takes the
    /// first of `replicated`, `gradients`, `parameters` whose rank fits the card and whose precision admits it
    /// (`optimizer` is never chosen: it holds what `gradients` holds and moves twice the gradient traffic); a single
    /// rank is replicated. When none fits, the refusal states what each needs and what the card holds.
    ///
    /// # Errors
    /// The refusals above, and those of [`Self::sharded_footprint`].
    pub fn sharding_plan(&self, block: u32, world: usize, card_bytes: Option<u64>, asked: Option<Sharding>) -> Result<ShardingPlan> {
        pick_sharding(world, card_bytes, asked, |s| s.admits(&self.precision), |s| self.sharded_footprint(block, s, world))
    }

    /// The architecture of the base in `model_dir`: the one named, else the checkpoint's own, else `qwen3`.
    /// Refuses, by name, an option the architecture's trainer does not take.
    fn refuse_options(&self, chat: &crate::chat_trainer::ChatEntry) -> Result<()> {
        if self.native_sparse.is_some() && !chat.native_sparse {
            return Err(Error::InvalidRequest("ChatFullFineTune: native sparse attention training is not available for this architecture".to_string()));
        }
        if self.stages > 0 && !chat.pipelined {
            return Err(Error::Backend(format!("{}: {} stages asked for, and this architecture's trainer lays itself out over the cards and takes no stage count", "ChatFullFineTune", self.stages)));
        }
        Ok(())
    }

    fn arch_of<'a>(&'a self, model_dir: &Path) -> &'a str {
        match self.arch.as_deref() {
            Some(named) => named,
            None => crate::study::arch_of_checkpoint(model_dir).map_or("qwen3", |a| a.id),
        }
    }

    /// Check these settings against the base's architecture before anything is trained: the base resolves, its
    /// architecture has a full-parameter chat trainer that takes every option asked for, `exit_entropy` is one the
    /// architecture can use, and the optimiser's plan (`train_precision`, `lr`, `weight_decay`, `grad_accum`, the steps
    /// and the row length) is one the trainer accepts. Only the base's config is read: no weight and no dataset, so a
    /// scheduler can refuse a run that would fail before it is queued. [`Self::run_with`] makes the same checks.
    ///
    /// # Errors
    /// The refusal [`Self::run_with`] would give for the same settings, before it reads anything else.
    pub fn check(&self) -> Result<ChatFullFineTunePlan> {
        let (_, model_dir, _) = self.resolve_base()?;
        let chat = self.chat_for(&model_dir)?;
        let block = self.max_block.unwrap_or(crate::finetune::MIN_BLOCK);
        let (_, exit_entropy) = self.plan(&chat, block, self.seed.unwrap_or_default())?;
        Ok(ChatFullFineTunePlan { arch: self.arch_of(&model_dir).to_string(), exit_entropy, block, lr: self.lr, weight_decay: self.weight_decay })
    }

    /// The base this run trains, resolved in the model store: refused by name when it is not there, or when a named
    /// architecture's trainer does not take an option asked for (before anything is read).
    fn resolve_base(&self) -> Result<(PathBuf, PathBuf, String)> {
        if let Some(named) = self.arch.as_deref() {
            self.refuse_options(&chat_entry(named)?)?;
        }
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)
    }

    /// The architecture's chat trainer for the base in `model_dir`, holding it to every option asked for.
    fn chat_for(&self, model_dir: &Path) -> Result<crate::chat_trainer::ChatEntry> {
        let chat = chat_entry(self.arch_of(model_dir))?;
        self.refuse_options(&chat)?;
        if self.grad_accum == 0 {
            return Err(Error::InvalidRequest("ChatFullFineTune: grad_accum must be at least 1".to_string()));
        }
        Ok(chat)
    }

    /// The optimiser's validated options for rows of `block` tokens, and the exit-entropy weight they carry.
    fn plan(&self, chat: &crate::chat_trainer::ChatEntry, block: u32, seed: u64) -> Result<(model::FitOpts, f32)> {
        chat.train_precision.check(&self.precision).map_err(|why| Error::InvalidRequest(format!("ChatFullFineTune: {why}")))?;
        let exit_entropy = exit_entropy_for(chat, self.exit_entropy, FINE_TUNE_EXIT_ENTROPY)?;
        let sharding = self.parallel.as_ref().map_or(model::zero::Sharding::Replicated, crate::DataParallel::sharding);
        // A parameter-sharded run builds its resident state over the group.
        let parameter_group = self.parallel.as_ref().filter(|p| p.sharding() == model::zero::Sharding::Parameters).map(crate::DataParallel::shard_group);
        let requested = model::FitOpts { grad_accum: self.grad_accum, strict_updates: self.strict_updates, sharding, parameter_group, weight_decay: self.weight_decay, precision: self.precision.clone(), stages: self.stages, native_sparse: self.native_sparse, exit_entropy, ..fit_opts(self.steps, block, self.lr, seed) };
        let opts = model::training_plan::ValidatedTrainingPlan::build(requested, &[]).map_err(|e| Error::InvalidRequest(format!("ChatFullFineTune: {e}")))?.into_opts();
        Ok((opts, exit_entropy))
    }

    /// Run to completion.
    pub fn run(&self) -> Result<ChatFullFineTuneOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step to `on_progress` and stopping at
    /// the next step boundary once `cancel` fires; a stopped run writes no
    /// checkpoint.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<ChatFullFineTuneOutcome> {
        let run_clock = std::time::Instant::now();
        // A named architecture, and every option it refuses, are refused before anything is read; an
        // architecture read from the checkpoint is held to the same refusals once the base is resolved.
        let (weights, model_dir, base_id) = self.resolve_base()?;
        let chat = self.chat_for(&model_dir)?;
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("ChatFullFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ChatFullFineTune: no out directory; call .out_dir(path)".to_string()))?;
        let open = (chat.checkpoint_path)(&weights, &model_dir);

        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: false, images: &data::chat::TextOnly::TRAINER };
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
        let Mix { rows, replay_records, .. } = mix_rows(train_samples, replay_samples, self.replay_share, weighted_samples)?;

        let vocab = (chat.vocab)(&open).map_err(Error::Backend)?;
        let prepared_dir = out_dir.join(PREPARED_DIR);
        // No evaluation runs during a full fine-tune; the validation split is
        // the rows themselves, unread.
        let prepared = data::chat::prepare_chat_samples(&rows, &rows, &tok, &tmpl, render, vocab, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        model::finetune::stage_time("sdk: read dataset and prepare it", run_clock);
        let block = block_for(prepared.longest_example, self.max_block)?;
        let set = if self.placed {
            // The process owns the selection: the run is confined to the card this thread is scoped to.
            gpu_core::devices::ComputeSet { gpus: gpu_core::devices::current_gpu().into_iter().collect(), ..gpu_core::devices::ambient_compute_set().clone() }
        } else {
            crate::device::resolve(&self.device)?
        };
        // Where this rank trains, as resolved: what the run's report says it ran on.
        let ran_on = crate::data_parallel::RankDevice::resolved(self.parallel.as_ref().map_or(0, crate::DataParallel::rank), &set.gpus);
        if let Some(parallel) = &self.parallel {
            parallel.ran_on(ran_on.clone());
        }
        // The fp32 plan below prices every weight and gradient on the cards; a resident build holds its linears in
        // their own encoding (streamed through a window, sharded over a group) and its planner checks that build
        // before anything is allocated.
        if self.precision.resident.is_none() {
            plan_before_load(&open, block, &set)?;
        }

        if self.parallel.is_some() && self.seed.is_none() {
            return Err(Error::MissingArgument("ChatFullFineTune: a data-parallel rank needs the seed every rank shares; call .seed(n) with the same n on every rank".to_string()));
        }
        let seed = resolve_seed(self.seed, self.resume_state.as_deref())?;
        let (opts, exit_entropy) = self.plan(&chat, block, seed)?;
        let digest_clock = std::time::Instant::now();
        let base_digest = base_digest(&open)?;
        model::finetune::stage_time("sdk: digest the base weights", digest_clock);
        let mut identity = serde_json::json!({
            "objective": "chat_full",
            "base": base_digest,
            "dataset": digest(dataset)?,
            "replay": self.replay.iter().map(|p| digest(p)).collect::<Result<Vec<_>>>()?,
            "replay_share_bits": self.replay_share.map(f32::to_bits),
            "weighted_replay": self.weighted_replay.iter().map(|(p, share)| Ok((digest(p)?, share.to_bits()))).collect::<Result<Vec<_>>>()?,
            "thinking": self.thinking,
        });
        // Part of the arithmetic: a state trained at another weight is not this run's. Absent at 0, so the identity
        // of a run of a family without exit gates is what it was.
        if exit_entropy > 0.0 {
            identity["exit_entropy"] = serde_json::json!(exit_entropy);
        }
        let run = FullRun { chat: &chat, open: &open, tokenizer_dir: &model_dir, prepared_dir: &prepared_dir, out_dir, opts: &opts, set: &set, precision: &self.precision, resume_state: self.resume_state.as_deref(), checkpoint_every: self.checkpoint_every, identity, parallel: self.parallel.as_ref() };
        let trained = run.train_and_export(cancel, &mut on_progress)?;
        let mut outcome = ChatFullFineTuneOutcome {
            status: trained.status,
            checkpoint: None,
            weights: None,
            weights_digest: None,
            base_digest: Some(base_digest),
            seed,
            record: None,
            steps: self.steps,
            steps_completed: trained.report.steps_completed,
            resumed_at: trained.report.resumed_at,
            initial_loss: Some(trained.report.initial_loss),
            final_loss: trained.report.final_loss,
            train_records: dataset_records,
            replay_records,
            block,
            lr: self.lr,
            exit_entropy,
            parallel: crate::data_parallel::ParallelReport::of(self.parallel.as_ref(), ran_on),
        };
        if trained.status != FineTuneStatus::Completed {
            return Ok(outcome);
        }
        outcome.weights_digest = trained.weights_digest;
        outcome.weights = trained.weights;
        outcome.checkpoint = trained.checkpoint;

        let record_path = out_dir.join(RECORD_FILE);
        std::fs::write(&record_path, serde_json::to_vec_pretty(&outcome.record_json(&base_id)).map_err(|e| Error::Backend(e.to_string()))?)
            .map_err(|e| Error::Backend(format!("{}: {e}", record_path.display())))?;
        outcome.record = Some(record_path);
        Ok(outcome)
    }
}

/// One run of a family's full-parameter trainer over a prepared dataset, then the export of what it trained: the part a
/// chat fine-tune ([`ChatFullFineTune`]) and raw-text pretraining ([`crate::Pretrain`]) share. Only the dataset each
/// prepares and the options each sets differ.
pub(crate) struct FullRun<'a> {
    pub chat: &'a crate::chat_trainer::ChatEntry,
    /// The checkpoint the trainer opens.
    pub open: &'a Path,
    /// The directory whose tokenizer files the exported checkpoint carries.
    pub tokenizer_dir: &'a Path,
    pub prepared_dir: &'a Path,
    pub out_dir: &'a Path,
    pub opts: &'a model::FitOpts,
    pub set: &'a gpu_core::devices::ComputeSet,
    pub precision: &'a model::TrainPrecision,
    pub resume_state: Option<&'a Path>,
    pub checkpoint_every: u32,
    /// What else a saved state must agree with to be continued (digests of the base and data, settings).
    pub identity: serde_json::Value,
    /// This run as one rank of a data-parallel group; only rank 0 exports.
    pub parallel: Option<&'a crate::DataParallel>,
}

/// What a [`FullRun`] produced.
pub(crate) struct FullTrained {
    pub status: FineTuneStatus,
    pub report: model::FitReport,
    pub checkpoint: Option<PathBuf>,
    pub weights: Option<PathBuf>,
    pub weights_digest: Option<String>,
}

impl FullRun<'_> {
    /// Train, reporting every step and stopping at the next boundary once `cancel` fires (the resume state is written
    /// first when there is one), then export the checkpoint directory and digest its weights. A stopped run exports
    /// nothing and leaves its state for the next call.
    pub(crate) fn train_and_export(&self, cancel: &CancelToken, on_progress: &mut dyn FnMut(&FineTuneProgress)) -> Result<FullTrained> {
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress::of(s));
            // The ranks of a group stop after the same step.
            match self.parallel {
                Some(rank) => rank.keep_going(s.step, cancel.is_cancelled()),
                None => !cancel.is_cancelled(),
            }
        };
        let control = model::FitControl {
            on_step: Some(&mut on_step),
            state: self.resume_state,
            state_every: self.checkpoint_every,
            identity: self.identity.clone(),
            grad_sync: self.parallel.map(crate::DataParallel::grad_sync),
            ..Default::default()
        };
        let tuned = self.out_dir.join(TUNED_FILE);
        let train_clock = std::time::Instant::now();
        // A sharded run's optimizer holds this rank's slice of the moments in host memory: the card is planned and
        // built with none beside the weights.
        let state = if self.opts.sharding.is_replicated() { optimizer_state_for(self.open, self.opts.block_size, self.set, self.precision)? } else { model::finetune::OptimizerState::HostOffload };
        let report = (self.chat.full)(self.open, self.prepared_dir, self.opts, &tuned, state, control).map_err(|e| match e.kind() {
            std::io::ErrorKind::InvalidData if self.resume_state.is_some_and(Path::exists) => Error::Backend(format!("the saved training state cannot be continued by this run: {e}")),
            _ => crate::error::training_error("training", e),
        })?;
        model::finetune::stage_time("sdk: train (open, upload, steps, save)", train_clock);
        let mut trained = FullTrained { status: FineTuneStatus::Completed, report, checkpoint: None, weights: None, weights_digest: None };
        if trained.report.interrupted {
            std::fs::remove_file(&tuned).ok();
            trained.status = FineTuneStatus::Cancelled;
            return Ok(trained);
        }
        // The ranks hold the same weights; rank 0 wrote them and exports them.
        if self.parallel.is_some_and(|p| p.rank() != 0) {
            return Ok(trained);
        }
        let checkpoint = self.out_dir.join(CHECKPOINT_DIR);
        let export_clock = std::time::Instant::now();
        match export_trained(self.chat, &tuned, self.tokenizer_dir, &checkpoint, None, &mut || !cancel.is_cancelled()) {
            Err(Error::Cancelled) => {
                std::fs::remove_file(&tuned).ok();
                trained.status = FineTuneStatus::Cancelled;
                return Ok(trained);
            }
            exported => exported?,
        }
        std::fs::remove_file(&tuned).ok();
        model::finetune::stage_time("sdk: export the trained file to the checkpoint directory", export_clock);
        // A family that exports a brain checkpoint writes the store's base file, one with a Hugging Face writer
        // `model.safetensors`.
        let brain_file = checkpoint.join(model::finetune::BRAIN_WEIGHTS_FILE);
        let weights = if brain_file.exists() { brain_file } else { checkpoint.join(WEIGHTS_FILE) };
        trained.weights_digest = Some(digest(&weights)?);
        trained.weights = Some(weights);
        trained.checkpoint = Some(checkpoint);
        // The run is exported; its state would only resume into the same checkpoint again.
        if let Some(state) = self.resume_state {
            std::fs::remove_file(state).ok();
            std::fs::remove_dir_all(state.with_extension("offdevice")).ok();
        }
        Ok(trained)
    }
}

/// What a full fine-tune of some settings would train with, as [`ChatFullFineTune::check`] found it.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatFullFineTunePlan {
    /// The registry id of the architecture whose trainer would run.
    pub arch: String,
    /// The weight of the exit-distribution entropy in the objective (0 for an architecture without exit gates).
    pub exit_entropy: f32,
    /// The row length the plan was checked at: the longest row allowed.
    pub block: u32,
    /// The peak learning rate.
    pub lr: f32,
    /// AdamW's weight decay.
    pub weight_decay: f32,
}

/// What one full fine-tune did. Anything not measured is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatFullFineTuneOutcome {
    /// How the run ended.
    pub status: FineTuneStatus,
    /// The checkpoint directory; `None` unless completed.
    pub checkpoint: Option<PathBuf>,
    /// The directory's weights file (`model.safetensors` or `model.brain.safetensors`); `None` unless completed.
    pub weights: Option<PathBuf>,
    /// `sha256:<hex>` of the weights file.
    pub weights_digest: Option<String>,
    /// `sha256:<hex>` naming the base the run started from.
    pub base_digest: Option<String>,
    /// The JSON training record written beside the checkpoint; `None` unless
    /// completed.
    pub record: Option<PathBuf>,
    /// The seed of the batch order: the one asked for or the one drawn.
    /// Passing it to [`ChatFullFineTune::seed`] reproduces the run.
    pub seed: u64,
    /// Optimizer steps asked for.
    pub steps: u32,
    /// Steps completed, those before a resume included.
    pub steps_completed: u32,
    /// The step this call continued a saved [`ChatFullFineTune::resume_state`] at, when it did.
    pub resumed_at: Option<u32>,
    /// The training loss estimated before the first step.
    pub initial_loss: Option<f32>,
    /// The last step's training loss.
    pub final_loss: Option<f32>,
    /// Records of the dataset.
    pub train_records: usize,
    /// Records of the replay and weighted files.
    pub replay_records: usize,
    /// The training row length, sized to the longest record.
    pub block: u32,
    /// The peak learning rate of the run.
    pub lr: f32,
    /// The exit-entropy weight the run trained with (0 for a family without exit gates).
    pub exit_entropy: f32,
    /// How the run was spread over ranks, cards and servers, as it ran ([`crate::ParallelReport`]).
    pub parallel: crate::ParallelReport,
}

impl ChatFullFineTuneOutcome {
    /// The run's report: the document the served `train` job answers with for a full fine-tune (`objective`
    /// `full_sft`) and `brain finetune` writes as `report.json`.
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({
            "status": match self.status { FineTuneStatus::Completed => "completed", FineTuneStatus::Cancelled => "cancelled" },
            "objective": "full_sft",
            "weights_sha256": self.weights_digest,
            "base_digest": self.base_digest,
            "records": self.train_records,
            "replay_records": self.replay_records,
            "block": self.block,
            "seed": self.seed,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "lr": self.lr,
            "exit_entropy": self.exit_entropy,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "parallel": self.parallel,
        })
    }

    fn record_json(&self, base_id: &str) -> serde_json::Value {
        serde_json::json!({
            "base": base_id,
            "base_digest": self.base_digest,
            "weights_digest": self.weights_digest,
            "seed": self.seed,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "train_records": self.train_records,
            "replay_records": self.replay_records,
            "block": self.block,
            "lr": self.lr,
        })
    }
}

/// The generic fine-tuners' entry points of architecture `arch`, or the
/// refusal naming it and the capability it lacks.
/// Refuse a full fine-tune that cannot fit the selected cards BEFORE the base is loaded, with the planner's
/// per-category arithmetic. Every tensor trains in fp32; the AdamW moments and the master weights live in host
/// memory ([`model::finetune::OptimizerState::HostOffload`]), so the cards hold the values, the gradients and one
/// step's activations. The activation term is calibrated on the 27B (`crate::finetune::activation_bytes`) and
/// included when the checkpoint's config names its depth and width. No selected card (a CPU run) is not checked.
pub(crate) fn plan_before_load(open: &Path, block: u32, set: &gpu_core::devices::ComputeSet) -> Result<()> {
    let cards: Vec<u64> = gpu_core::devices::gpus().iter().filter(|d| set.gpus.contains(&d.index)).map(|d| d.identity.vram_bytes).collect();
    if cards.is_empty() {
        return Ok(());
    }
    plan_on_cards(open, block, &cards)
}

/// [`plan_before_load`]'s arithmetic for cards of `cards` bytes. `open` is a checkpoint file or a checkpoint
/// directory (a `transformers` one, single-file or sharded, whose depth and width are in its `config.json`).
fn plan_on_cards(open: &Path, block: u32, cards: &[u64]) -> Result<()> {
    use model::memplan::{plan, training_demands, Authorized, Capacity, Category, Demand};
    let (footprint, numel) = full_footprint(open, block, true)?;
    let mut demands = training_demands(&footprint).map_err(|e| Error::Backend(e.to_string()))?;
    demands.push(Demand { category: Category::Weights, variants: vec![model::memplan::Variant::new(model::memplan::Lever::Base, "master weights on the host", 0, 4 * numel)] });
    // The trainer runs on the selected cards together; host memory is not bounded by the planner here.
    let capacity = Capacity { device_bytes: cards.iter().sum(), host_bytes: u64::MAX };
    plan(&demands, capacity, &Authorized::none()).map(drop).map_err(|e| Error::Backend(format!("ChatFullFineTune: refusing to load {}: {e}", open.display())))
}

/// What a full fine-tune of the checkpoint `open` asks of the device at `block` tokens, with the optimizer state on
/// the host (`state_on_host`) or beside the weights, and the element count of the model.
fn full_footprint(open: &Path, block: u32, state_on_host: bool) -> Result<(model::memplan::TrainingFootprint, u64)> {
    let reader = checkpoint::weightio::WeightReader::open_any(open.to_str().ok_or_else(|| Error::Backend(format!("{}: not valid UTF-8", open.display())))?).map_err(|e| Error::Backend(format!("{}: {e}", open.display())))?;
    let trainable: Vec<(String, usize)> = reader.names().map(|n| (n.to_string(), reader.shape(n).map_or(0, |s| s.iter().product::<u64>() as usize))).collect();
    let config = if open.is_dir() {
        let path = open.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&text).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))?
    } else {
        reader.config()
    };
    let dim = |keys: &[&str]| keys.iter().find_map(|k| config.get(*k).and_then(|v| v.as_u64()));
    let activation_bytes = match (dim(&["num_hidden_layers", "n_layers"]), dim(&["hidden_size", "d_model"])) {
        (Some(layers), Some(width)) => crate::finetune::activation_bytes(block, layers, width),
        _ => 0,
    };
    let numel: u64 = trainable.iter().map(|(_, n)| *n as u64).sum();
    let footprint = model::memplan::TrainingFootprint {
        frozen_weight_bytes: 0,
        trainable,
        precision: model::memplan::ParamPrecision::FP32,
        moment_slots: 2,
        int8_block: 256,
        optimizer_state_on_host: state_on_host,
        activation_bytes,
        recompute_activation_bytes: None,
        saved_boundaries: None,
    };
    Ok((footprint, numel))
}

/// Which sharding a full fine-tune trains with over its group, and what each rank of it asks of its card
/// ([`ChatFullFineTune::sharding_plan`]).
#[derive(Clone, Debug)]
pub struct ShardingPlan {
    pub sharding: Sharding,
    /// One rank's footprint, which every card of the group is charged.
    pub footprint: crate::TrainFootprint,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// The bytes one rank of a `world`-rank group sharding `sharding` keeps on its card for the parameters `trainable`
/// (name, elements) and `activation` bytes of one step, before the margin and the device context, with its terms
/// for a refusal to quote. See [`ChatFullFineTune::sharded_footprint`].
fn rank_card_bytes(trainable: &[(String, usize)], activation: u64, sharding: Sharding, world: usize, precision: &model::TrainPrecision) -> Result<(u64, String)> {
    let params: u64 = trainable.iter().map(|(_, n)| *n as u64).sum();
    let largest = trainable.iter().map(|(_, n)| *n as u64).max().unwrap_or(0);
    let weights_and_grads = 8 * params;
    if !sharding.is_replicated() && world < 2 {
        return Err(Error::InvalidRequest(format!("sharding {sharding:?} splits the run's state across the ranks of a data-parallel group, and this run has {world}; shard over two or more")));
    }
    let world = world.max(1) as u64;
    let slice = params.div_ceil(world);
    Ok(match sharding {
        Sharding::Replicated => (weights_and_grads + activation, format!("fp32 weights and gradients {:.2} GiB", gib(weights_and_grads))),
        Sharding::Optimizer | Sharding::Gradients => {
            // A step packs units of at most the cap (a tensor larger than the cap is its own unit) through two staging
            // buffers; the slice state is the master, both moments and the reduced gradient, or with `moments=host`
            // one slice buffer the host state is moved through.
            let unit = largest.max(params.min(model::zero::DEFAULT_UNIT_ELEMS));
            let staging = 2 * 4 * unit;
            let state = if precision.host_moments { 4 * unit.div_ceil(world) } else { 16 * slice };
            (weights_and_grads + state + staging + activation, format!("fp32 weights and gradients {:.2} GiB + this rank's optimizer slice {:.2} GiB + staging {:.2} GiB", gib(weights_and_grads), gib(state), gib(staging)))
        }
        Sharding::Parameters => {
            // Every layer is one unit and every other tensor (a table, a norm) is its own.
            let mut layers: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
            let mut other = 0u64;
            for (name, n) in trainable {
                match layer_of(name) {
                    Some(layer) => *layers.entry(layer).or_default() += *n as u64,
                    None => other = other.max(*n as u64),
                }
            }
            let layer = layers.values().copied().max().unwrap_or(0);
            let resident = precision.resident;
            let value = |n: u64| resident.map_or(2 * n, |r| r.weights.device_bytes(1, n));
            let window = resident.and_then(|r| r.window).map_or(2, |w| w as u64).max(2);
            let slices = 12 * slice + value(slice);
            let gathered = window * (value(layer) + 4 * layer) + value(other) + 4 * other;
            let transient = 4 * layer.max(other).div_ceil(world);
            (slices + gathered + transient + activation, format!("this rank's slices of master, moments and values {:.2} GiB + gathered layers and tables {:.2} GiB + a gradient slice {:.2} GiB", gib(slices), gib(gathered), gib(transient)))
        }
    })
}

/// The layer a parameter belongs to (`...layers.<n>.`), when it belongs to one.
fn layer_of(name: &str) -> Option<&str> {
    let at = name.find("layers.")?;
    let rest = &name[at + "layers.".len()..];
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    (digits > 0).then(|| &name[..at + "layers.".len() + digits])
}

/// The rule of [`ChatFullFineTune::sharding_plan`] over `admits` (whether the run's precision can shard a way) and
/// `footprint` (one rank's footprint sharded a way).
fn pick_sharding(world: usize, card: Option<u64>, asked: Option<Sharding>, admits: impl Fn(Sharding) -> std::result::Result<(), String>, footprint: impl Fn(Sharding) -> Result<crate::TrainFootprint>) -> Result<ShardingPlan> {
    let name = |s: Sharding| serde_json::to_value(s).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
    let fits = |f: &crate::TrainFootprint| card.is_none_or(|card| f.device_bytes <= card);
    let card_text = |card: u64| format!("{:.1} GiB", gib(card));
    if let Some(sharding) = asked {
        admits(sharding).map_err(|why| Error::InvalidRequest(format!("sharding {}: {why}", name(sharding))))?;
        let footprint = footprint(sharding)?;
        if let Some(card) = card.filter(|_| !fits(&footprint)) {
            return Err(Error::InvalidRequest(format!("sharding {} over {world} rank(s) needs about {:.1} GiB a card ({}: {}) and the largest card has {}", name(sharding), gib(footprint.device_bytes), footprint.reason, footprint.breakdown, card_text(card))));
        }
        return Ok(ShardingPlan { sharding, footprint });
    }
    let candidates: &[Sharding] = if world < 2 { &[Sharding::Replicated] } else { &[Sharding::Replicated, Sharding::Gradients, Sharding::Parameters] };
    let mut needs = Vec::new();
    for &sharding in candidates {
        let footprint = footprint(sharding)?;
        match admits(sharding) {
            Ok(()) if fits(&footprint) => return Ok(ShardingPlan { sharding, footprint }),
            Ok(()) => needs.push(format!("{} {:.1} GiB ({})", name(sharding), gib(footprint.device_bytes), footprint.breakdown)),
            Err(why) => needs.push(format!("{} {:.1} GiB, not with this precision ({why})", name(sharding), gib(footprint.device_bytes))),
        }
    }
    let card = card.map(card_text).unwrap_or_default();
    Err(Error::InvalidRequest(format!("no sharding over {world} rank(s) fits a card of {card}: {}; lower max_block, hold the base narrower, or train on more cards", needs.join("; "))))
}

/// Where the AdamW moments of a full fine-tune of `open` live: beside the weights and gradients when the first selected
/// card has room for all of it (the update is then a device kernel, not a round trip of every gradient and moment
/// through host memory), in host memory otherwise. A CPU run, or a card whose free memory the platform does not report,
/// keeps the moments on the host as before.
fn optimizer_state_for(open: &Path, block: u32, set: &gpu_core::devices::ComputeSet, precision: &model::TrainPrecision) -> Result<model::finetune::OptimizerState> {
    use model::finetune::OptimizerState;
    // A resident build keeps its master and moments on the host by construction; a precision that only the host
    // optimizer runs (`moments=host`, log8) needs them there.
    if precision.resident.is_some() || precision.wants_host_moments() {
        return Ok(OptimizerState::HostOffload);
    }
    let Some(card) = gpu_core::devices::gpus().iter().filter(|d| set.gpus.contains(&d.index)).map(|d| d.index).min() else { return Ok(OptimizerState::HostOffload) };
    let Some(free) = gpu_core::capacity::available_for(card) else { return Ok(OptimizerState::HostOffload) };
    let fits = state_fits_on_device(open, block, precision.policy.default, free)?;
    if gpu_core::profile::enabled() {
        eprintln!("optimizer state placement: card {card} has {free} bytes free; the moments {}", if fits { "fit beside the weights and gradients" } else { "go to host memory" });
    }
    Ok(if fits { OptimizerState::Device } else { OptimizerState::HostOffload })
}

/// Whether a full fine-tune of `open` at `block` tokens fits a card with `free` bytes with the AdamW moments beside the
/// weights and gradients: the planner's arithmetic with no lever pulled.
fn state_fits_on_device(open: &Path, block: u32, precision: model::memplan::ParamPrecision, free: u64) -> Result<bool> {
    use model::memplan::{plan, training_demands, Authorized, Capacity};
    let (footprint, _) = full_footprint(open, block, false)?;
    let footprint = model::memplan::TrainingFootprint { precision, ..footprint };
    let demands = training_demands(&footprint).map_err(|e| Error::Backend(e.to_string()))?;
    Ok(plan(&demands, Capacity { device_bytes: free, host_bytes: u64::MAX }, &Authorized::none()).is_ok())
}

pub(crate) fn chat_entry(arch: &str) -> Result<crate::chat_trainer::ChatEntry> {
    let entry = crate::study::arch(arch)?;
    entry.require(Capability::ChatFineTune)?;
    entry.chat.ok_or_else(|| Error::Unsupported(model::training::Unsupported::new(entry.id, Capability::ChatFineTune)))
}

/// Writes the brain checkpoint `trained` to `checkpoint` as a checkpoint
/// directory in the architecture's own format, with the tokenizer files of
/// `tokenizer_dir` and `adapter` folded in when one is given. `keep_going` is
/// asked before each unit of the export's work; a stopped export is
/// [`Error::Cancelled`] and leaves no file of its own.
fn export_trained(chat: &crate::chat_trainer::ChatEntry, trained: &Path, tokenizer_dir: &Path, checkpoint: &Path, adapter: Option<&Path>, keep_going: &mut dyn FnMut() -> bool) -> Result<()> {
    (chat.export)(trained, tokenizer_dir, checkpoint, adapter, keep_going).map_err(|e| if e == capability::failure::CANCELLED { Error::Cancelled } else { Error::Backend(e) })
}

/// Folds the LoRA `adapter` into the weights of `base` it targets
/// (`W + alpha/r * B A`) and writes the result to `out_dir` as a checkpoint
/// directory: the base with the adapter, in one set of weights, that needs no
/// adapter to be served. `base` is whatever [`ChatFullFineTune::from_pretrained`]
/// takes; the tokenizer files are copied from the base's directory. Returns
/// `out_dir`, with the weights file and its digest. `keep_going` is asked
/// before each unit of the fold's work (a tensor, a chunk of a copy); once it
/// says no the merge is [`Error::Cancelled`] and leaves nothing of its own in
/// `out_dir`.
pub fn merge_adapter(base: &str, adapter: &Path, out_dir: &Path, keep_going: &mut dyn FnMut() -> bool) -> Result<MergedCheckpoint> {
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    // The base's own architecture folds it (`qwen3` for a checkpoint that does not say); one without an export
    // that takes an adapter refuses by name.
    let chat = chat_entry(crate::study::arch_of_checkpoint(&model_dir).map_or("qwen3", |a| a.id))?;
    let open = (chat.checkpoint_path)(&weights, &model_dir);
    export_trained(&chat, &open, &model_dir, out_dir, Some(adapter), keep_going)?;
    let weights = out_dir.join(WEIGHTS_FILE);
    Ok(MergedCheckpoint { weights_digest: digest(&weights)?, dir: out_dir.to_path_buf(), weights })
}

/// The checkpoint directory [`merge_adapter`] wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct MergedCheckpoint {
    /// The directory.
    pub dir: PathBuf,
    /// Its `model.safetensors`.
    pub weights: PathBuf,
    /// `sha256:<hex>` of the weights file.
    pub weights_digest: String,
}

/// The exit-entropy weight of a chat fine-tune of a family with exit gates, unless the run names one: the weight the
/// entropy-regularised looped-decoder recipe (arXiv:2510.25741) keeps after its first pretraining stage.
pub const FINE_TUNE_EXIT_ENTROPY: f32 = 0.05;

/// The exit-entropy weight of a run: `asked` when given (finite, non-negative, and zero for a family without exit
/// gates), else `default` for a family with exit gates and 0 for any other.
pub(crate) fn exit_entropy_for(chat: &crate::chat_trainer::ChatEntry, asked: Option<f32>, default: f32) -> Result<f32> {
    match asked {
        Some(beta) if !(beta.is_finite() && beta >= 0.0) => Err(Error::Backend(format!("exit_entropy {beta}: the weight must be finite and non-negative"))),
        Some(beta) if beta > 0.0 && !chat.exit_gates => Err(Error::Backend(format!("exit_entropy {beta}: the {} architecture has no exit gates to regularise", chat.model))),
        Some(beta) => Ok(beta),
        None => Ok(if chat.exit_gates { default } else { 0.0 }),
    }
}

#[cfg(test)]
mod tests {
    use model::memplan::ParamPrecision;

    use super::{pick_sharding, plan_on_cards, rank_card_bytes, state_fits_on_device, Sharding};

    /// A `transformers` checkpoint directory: weights with no embedded config, and the config beside them.
    fn hf_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-sdk-full-plan-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let tensors = vec![("model.embed_tokens.weight".to_string(), vec![64, 32], vec![0.0f32; 64 * 32]), ("lm_head.weight".to_string(), vec![64, 32], vec![0.0f32; 64 * 32])];
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"num_hidden_layers": 2, "hidden_size": 32}"#).unwrap();
        dir
    }

    #[test]
    fn a_checkpoint_directory_is_planned_and_refused_only_when_it_does_not_fit() {
        let dir = hf_dir("fit");
        plan_on_cards(&dir, 64, &[1 << 40]).expect("a checkpoint directory fits a large card");
        let refused = plan_on_cards(&dir, 64, &[1024]).expect_err("nothing fits a 1 KiB card");
        assert!(refused.to_string().contains("refusing to load"), "{refused}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The moments stay beside the weights exactly when the card has room for weights, gradients, moments and
    /// activations together; otherwise they go to host memory.
    #[test]
    fn the_moments_stay_on_the_card_only_when_the_whole_state_fits_it() {
        let dir = hf_dir("state");
        let numel = 2 * 64 * 32u64;
        // Values, gradients and two moments in fp32 are 16 bytes per element.
        assert!(state_fits_on_device(&dir, 64, ParamPrecision::FP32, 1 << 40).unwrap(), "a large card holds everything");
        assert!(state_fits_on_device(&dir, 64, ParamPrecision::FP32, 16 * numel + (1 << 20)).unwrap(), "room for the state and a small activation term");
        assert!(!state_fits_on_device(&dir, 64, ParamPrecision::FP32, 12 * numel).unwrap(), "room for values, gradients and one moment is not enough");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A toy decoder's parameters: `layers` layers of `per_layer` elements and two tables of `table` elements.
    fn decoder(layers: usize, per_layer: usize, table: usize) -> Vec<(String, usize)> {
        let mut params = vec![("model.embed_tokens.weight".to_string(), table), ("lm_head.weight".to_string(), table)];
        for l in 0..layers {
            params.push((format!("model.layers.{l}.mlp.weight"), per_layer / 2));
            params.push((format!("model.layers.{l}.attn.weight"), per_layer / 2));
        }
        params
    }

    /// Replicated ranks hold the same whatever the group; the optimizer stages add only their slice of the state,
    /// which shrinks as the group grows; parameter sharding is the one that holds less than a replica on a large
    /// model, and less again on more ranks; a sharded stage needs a group.
    #[test]
    fn a_rank_holds_what_its_sharding_leaves_on_its_card() {
        let default = model::TrainPrecision::default();
        let resident = model::TrainPrecision::parse("weights=bf16").unwrap();
        let params = decoder(36, 100 << 20, 400 << 20);
        let bytes = |s: Sharding, world: usize, p: &model::TrainPrecision| rank_card_bytes(&params, 0, s, world, p).unwrap().0;
        assert_eq!(bytes(Sharding::Replicated, 1, &default), bytes(Sharding::Replicated, 8, &default));
        assert!(bytes(Sharding::Gradients, 4, &default) > bytes(Sharding::Replicated, 4, &default), "the optimizer stages keep the full weights and gradients and add a slice");
        assert!(bytes(Sharding::Gradients, 8, &default) < bytes(Sharding::Gradients, 2, &default));
        assert_eq!(bytes(Sharding::Optimizer, 4, &default), bytes(Sharding::Gradients, 4, &default));
        assert!(bytes(Sharding::Gradients, 4, &model::TrainPrecision::parse("moments=host").unwrap()) < bytes(Sharding::Gradients, 4, &default), "moments=host keeps the slice off the card");
        assert!(bytes(Sharding::Parameters, 4, &resident) < bytes(Sharding::Replicated, 4, &default), "parameter sharding holds less than a replica");
        assert!(bytes(Sharding::Parameters, 8, &resident) < bytes(Sharding::Parameters, 4, &resident));
        for s in [Sharding::Optimizer, Sharding::Gradients, Sharding::Parameters] {
            let refused = rank_card_bytes(&params, 0, s, 1, &default).unwrap_err().to_string();
            assert!(refused.contains("two or more"), "{s:?}: {refused}");
        }
    }

    fn footprint(bytes: u64) -> crate::TrainFootprint {
        crate::TrainFootprint { device_bytes: bytes, breakdown: format!("{bytes} bytes"), base_tier: "f32".into(), reason: "toy".into() }
    }

    /// Unasked, a run takes the first of replicated, gradients and parameters that fits the card and that its
    /// precision admits; asked, it takes what it asked or is refused with the numbers; with nothing fitting, the
    /// refusal states what each needs and what the card holds.
    #[test]
    fn a_full_fine_tune_shards_only_as_far_as_the_card_needs() {
        let sizes = |s: Sharding| Ok(footprint(match s {
            Sharding::Replicated => 100 << 30,
            Sharding::Optimizer | Sharding::Gradients => 80 << 30,
            Sharding::Parameters => 30 << 30,
        }));
        let any = |_: Sharding| Ok(());
        let no_resident = |s: Sharding| if s == Sharding::Parameters { Err("needs weights=bf16".to_string()) } else { Ok(()) };
        let pick = |card: Option<u64>, asked: Option<Sharding>, admits: &dyn Fn(Sharding) -> Result<(), String>| pick_sharding(4, card, asked, admits, sizes).map(|p| p.sharding);
        assert_eq!(pick(None, None, &any).unwrap(), Sharding::Replicated, "an unknown card holds a replica");
        assert_eq!(pick(Some(120 << 30), None, &any).unwrap(), Sharding::Replicated);
        assert_eq!(pick(Some(90 << 30), None, &any).unwrap(), Sharding::Gradients);
        assert_eq!(pick(Some(50 << 30), None, &any).unwrap(), Sharding::Parameters);
        let none = pick(Some(50 << 30), None, &no_resident).unwrap_err().to_string();
        assert!(none.contains("replicated 100.0 GiB") && none.contains("gradients 80.0 GiB") && none.contains("parameters 30.0 GiB") && none.contains("weights=bf16") && none.contains("50.0 GiB"), "{none}");
        let too_small = pick(Some(10 << 30), None, &any).unwrap_err().to_string();
        assert!(too_small.contains("no sharding over 4 rank(s) fits a card of 10.0 GiB"), "{too_small}");
        assert_eq!(pick(Some(120 << 30), Some(Sharding::Gradients), &any).unwrap(), Sharding::Gradients, "what is asked is taken");
        assert_eq!(pick(Some(90 << 30), Some(Sharding::Optimizer), &any).unwrap(), Sharding::Optimizer);
        let over = pick(Some(50 << 30), Some(Sharding::Gradients), &any).unwrap_err().to_string();
        assert!(over.contains("sharding gradients over 4 rank(s) needs about 80.0 GiB") && over.contains("50.0 GiB"), "{over}");
        let refused = pick(None, Some(Sharding::Parameters), &no_resident).unwrap_err().to_string();
        assert!(refused.contains("sharding parameters: needs weights=bf16"), "{refused}");
        let alone = pick_sharding(1, Some(50 << 30), None, any, sizes).unwrap_err().to_string();
        assert!(alone.contains("replicated 100.0 GiB") && !alone.contains("gradients"), "one rank is never sharded: {alone}");
    }
}
