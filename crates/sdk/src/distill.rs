// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distillation of a trained policy back into a fresh or continued adapter, for its
// clients. If your team needs expertise in knowledge distillation and consolidation of fine-tuned language models, you
// can procure our services by sending an email to info@swedishembedded.com.

//! [`DistillFineTune`]: a LoRA adapter trained toward a teacher's top-K next-token distributions on chat records, the
//! teacher being the same base on its own, with the adapter the run starts from, or with another adapter.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::DistillFineTune::from_pretrained("Qwen/Qwen3-4B")
//!     .dataset("prompts.jsonl")
//!     .teacher(brain::DistillTeacher::Adapter("adapters/specialist.safetensors".into()))
//!     .out_dir("runs/distill-1")
//!     .steps(200)
//!     .run()?;
//! println!("cross-entropy to the teacher {:?} -> {:?}", outcome.initial_loss, outcome.final_loss);
//! # Ok(()) }
//! ```
//!
//! **The objective** (`rl::objective::distill::DistillTopK`): at every supervised position of a record the teacher's
//! `top_k` most likely next tokens, renormalised, are the target of a forward KL `KL(q_K || p)`, a mean over the
//! record's teacher-scored positions; it is `top_k` weighted cross-entropy passes over one forward. The teacher scores
//! every record once before the first step (a forward of the base, or of the base with its adapter, at fp32), so no
//! second model is held while the student trains.

use std::path::PathBuf;

use model::training::Capability;
use rl::objective::distill::DistillConfig;

use crate::chat_trainer::DistillJob;
use crate::finetune::{base_digest, block_for, digest, fit_opts, lora_shape, resolve_seed, tokenizer_and_template, Asked};
use crate::{CancelToken, Device, Error, FineTuneProgress, FineTuneStatus, Result};

const STATE_FILE: &str = "train.state";
const ADAPTER_FILE: &str = "adapter.safetensors";
const RECORD_FILE: &str = "training.json";
const PREPARED_DIR: &str = "prepared";

/// The teacher's top-K kept per position when [`DistillFineTune::top_k`] is not called.
pub const DEFAULT_DISTILL_TOP_K: usize = 16;

/// Which policy of the same base teaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DistillTeacher {
    /// The base on its own.
    Base,
    /// The base with the adapter the run starts from ([`DistillFineTune::continue_from`], which it needs).
    Start,
    /// The base with this adapter.
    Adapter(PathBuf),
}

impl DistillTeacher {
    /// The teacher's name in the training record: `base`, `start` or `adapter`.
    pub fn name(&self) -> &'static str {
        match self {
            DistillTeacher::Base => "base",
            DistillTeacher::Start => "start",
            DistillTeacher::Adapter(_) => "adapter",
        }
    }
}

/// A distillation of a LoRA adapter on a chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct DistillFineTune {
    base: String,
    arch: Option<String>,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    teacher: DistillTeacher,
    top_k: usize,
    continue_from: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    adapter_id: Option<String>,
    rank: Option<u32>,
    alpha: Option<f32>,
    grad_accum: u32,
    strict_updates: bool,
    thinking: bool,
    steps: u32,
    lr: f32,
    seed: Option<u64>,
    max_block: Option<u32>,
    checkpoint_every: u32,
    resume_state: Option<PathBuf>,
    device: Device,
    placed: bool,
    base_format: Option<String>,
    /// This run as one rank of a data-parallel group ([`DistillFineTune::data_parallel`]).
    parallel: Option<crate::DataParallel>,
}

impl DistillFineTune {
    /// A distillation onto an adapter of `base`, resolved like [`crate::ChatFineTune::from_pretrained`]'s; the teacher
    /// defaults to the base.
    pub fn from_pretrained(base: impl Into<String>) -> DistillFineTune {
        DistillFineTune {
            base: base.into(),
            arch: None,
            models_dir: None,
            dataset: None,
            teacher: DistillTeacher::Base,
            top_k: DEFAULT_DISTILL_TOP_K,
            continue_from: None,
            out_dir: None,
            adapter_id: None,
            rank: None,
            alpha: None,
            grad_accum: 1,
            strict_updates: false,
            thinking: true,
            steps: 100,
            lr: crate::finetune::DEFAULT_LORA_LR,
            seed: None,
            max_block: None,
            checkpoint_every: 0,
            resume_state: None,
            device: Device::default(),
            placed: false,
            base_format: None,
            parallel: None,
        }
    }

    /// Train as one rank of a data-parallel group ([`crate::DataParallel`]), as [`crate::ChatFineTune::data_parallel`]
    /// does: every rank runs this distillation with the same settings and the same [`Self::seed`], each on its own
    /// card, trains on its own shard of the records (rank `r` of `world` trains the `n`-th record of the stream when
    /// `n % world == r`), and every step's gradients are reduced over the group, so the ranks' adapters stay
    /// byte-identical. Each rank has the teacher score every record before the first step.
    pub fn data_parallel(mut self, rank: crate::DataParallel) -> Self {
        self.parallel = Some(rank);
        self
    }

    /// The architecture of the base; unnamed, the base's own `config.json` tells it (`qwen3` where there is none).
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = Some(id.into());
        self
    }

    /// The models directory a `vendor/repo` base resolves in.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The `generic-messages-v2` records whose supervised turns the teacher scores (required).
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// The policy that teaches (default [`DistillTeacher::Base`]).
    pub fn teacher(mut self, teacher: DistillTeacher) -> Self {
        self.teacher = teacher;
        self
    }

    /// The teacher's most likely tokens kept per position (default [`DEFAULT_DISTILL_TOP_K`]).
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Continue this adapter as the student (and, with [`DistillTeacher::Start`], as the teacher).
    pub fn continue_from(mut self, adapter: impl Into<PathBuf>) -> Self {
        self.continue_from = Some(adapter.into());
        self
    }

    /// Where the prepared records, the adapter, its training record and any resume state go (required).
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// The id written on the adapter's card (default `<base id>:local:distill:latest`).
    pub fn adapter_id(mut self, id: impl Into<String>) -> Self {
        self.adapter_id = Some(id.into());
        self
    }

    /// LoRA rank of a fresh student (default 8; a continued adapter keeps its own).
    pub fn rank(mut self, rank: u32) -> Self {
        self.rank = Some(rank);
        self
    }

    /// LoRA alpha of a fresh student (default `2 * rank`).
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// Records per optimizer step (default 1).
    pub fn grad_accum(mut self, records: u32) -> Self {
        self.grad_accum = records;
        self
    }

    /// Abort at the first optimiser step whose loss or gradient is not finite.
    pub fn strict_updates(mut self, strict: bool) -> Self {
        self.strict_updates = strict;
        self
    }

    /// Render records with their reasoning (default on), as [`crate::ChatFineTune::thinking`].
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// Optimizer steps (default 100).
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// Peak learning rate.
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Seeds the fresh adapter; unset, a random one (reported).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The longest training row; a longer record is refused by name.
    pub fn max_block(mut self, tokens: u32) -> Self {
        self.max_block = Some(tokens);
        self
    }

    /// Also save the resume state every this-many steps (default 0: only when cancelled).
    pub fn checkpoint_every(mut self, steps: u32) -> Self {
        self.checkpoint_every = steps;
        self
    }

    /// Exact resume through this training-state file instead of `<out_dir>/train.state`.
    pub fn resume_state(mut self, path: impl Into<PathBuf>) -> Self {
        self.resume_state = Some(path.into());
        self
    }

    /// Hold the student's frozen base in the named format (the teacher is scored at fp32).
    pub fn base_format(mut self, spec: impl Into<String>) -> Self {
        self.base_format = Some(spec.into());
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Run on the device this thread is already scoped to; see [`crate::PreferenceFineTune::placed`].
    pub fn placed(mut self, on: bool) -> Self {
        self.placed = on;
        self
    }

    /// Run to completion.
    pub fn run(&self) -> Result<DistillFineTuneOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step and stopping at the next step boundary once `cancel` fires. A cancelled run
    /// exports no adapter and leaves its resume state.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<DistillFineTuneOutcome> {
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let arch = match self.arch.as_deref() {
            Some(named) => crate::study::arch(named)?,
            None => crate::study::arch_of_checkpoint(&model_dir).map_or_else(|| crate::study::arch("qwen3"), Ok)?,
        };
        arch.require(Capability::ChatFineTune)?;
        arch.require(Capability::TokenObjectives)?;
        let chat = arch.chat.ok_or_else(|| Error::Unsupported(model::training::Unsupported::new(arch.id, Capability::ChatFineTune)))?;
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("DistillFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("DistillFineTune: no out directory; call .out_dir(path)".to_string()))?;
        if self.top_k == 0 {
            return Err(Error::InvalidRequest("DistillFineTune: top_k must be at least 1".to_string()));
        }
        if self.grad_accum == 0 {
            return Err(Error::InvalidRequest("DistillFineTune: grad_accum must be at least 1".to_string()));
        }
        let teacher_adapter = match &self.teacher {
            DistillTeacher::Base => None,
            DistillTeacher::Start => Some(self.continue_from.clone().ok_or_else(|| Error::MissingArgument("DistillFineTune: teacher start is the adapter the run starts from; call .continue_from(adapter)".to_string()))?),
            DistillTeacher::Adapter(path) => Some(path.clone()),
        };
        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: false, images: &data::chat::TextOnly::TRAINER };
        let records = asked.read(dataset)?;
        let open = (chat.checkpoint_path)(&weights, &model_dir);
        let vocab = (chat.vocab)(&open).map_err(Error::Backend)?;
        let prepared_dir = out_dir.join(PREPARED_DIR);
        let prepared = data::chat::prepare_chat_samples(&records, &[], &tok, &tmpl, render, vocab, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        let block = block_for(prepared.longest_example, self.max_block)?;
        let base_format = match self.base_format.as_deref() {
            Some(spec) => chat.lora_bases.parse(spec).map_err(Error::Backend)?,
            None if self.placed => {
                let card_bytes = gpu_core::devices::current_gpu().and_then(|index| gpu_core::devices::gpus().iter().find(|d| d.index == index)).map(|d| d.identity.vram_bytes);
                let mut sizing = crate::ChatFineTune::from_pretrained(self.base.as_str()).arch(arch.id);
                if let Some(dir) = &self.models_dir {
                    sizing = sizing.models_dir(dir.as_str());
                }
                chat.lora_bases.parse(&sizing.footprint(block, card_bytes)?.base_tier).map_err(Error::Backend)?
            }
            None => gpu_core::select::Dtype::F32,
        };
        let base_tiers = model::ops::TierPolicy::uniform(base_format);
        let (rank, alpha, parent) = lora_shape(self.continue_from.as_deref(), self.rank, self.alpha, Default::default())?;
        let gpus: Vec<u32> = if self.placed { gpu_core::devices::current_gpu().into_iter().collect() } else { crate::device::resolve(&self.device)?.gpus };
        let ran_on = crate::data_parallel::place_rank("DistillFineTune", self.parallel.as_ref(), self.seed, &gpus)?;
        let state_path = self.resume_state.clone().unwrap_or_else(|| out_dir.join(STATE_FILE));
        let seed = resolve_seed(self.seed, Some(&state_path))?;
        // One record per row: the teacher reads whole-row logits, and the student trains one example per micro-step.
        let requested = model::FitOpts { batch_size: 1, grad_accum: self.grad_accum, strict_updates: self.strict_updates, ..fit_opts(self.steps, block, self.lr, seed) };
        let cfg = DistillConfig { seq_len: block as usize };
        let opts = model::training_plan::ValidatedTrainingPlan::build(requested, &[&cfg]).map_err(|e| Error::InvalidRequest(format!("DistillFineTune: {e}")))?.into_opts();
        let fp32 = model::ops::TierPolicy::uniform(gpu_core::select::Dtype::F32);
        let (examples, teacher_digest) = match &teacher_adapter {
            None => ((chat.teacher)(&open, &prepared_dir, &opts, self.top_k).map_err(|e| Error::Backend(format!("the teacher: {e}")))?, None),
            Some(path) => {
                let (t_rank, t_alpha, t_digest) = lora_shape(Some(path), None, None, Default::default())?;
                ((chat.adapter_teacher)(&open, &prepared_dir, &opts, self.top_k, (path, t_rank, t_alpha), &fp32).map_err(|e| Error::Backend(format!("the teacher: {e}")))?, t_digest)
            }
        };
        let base_digest = base_digest(&weights)?;
        let dataset_id = digest(dataset)?;
        let identity = serde_json::json!({
            "regime": "distill",
            "base": base_digest,
            "dataset": dataset_id,
            "parent": parent,
            "teacher": self.teacher.name(),
            "teacher_adapter": teacher_digest,
            "top_k": self.top_k,
            "rank": rank,
            "alpha_bits": alpha.to_bits(),
            "grad_accum": self.grad_accum,
            "thinking": self.thinking,
            "base_format": self.base_format,
        });
        std::fs::create_dir_all(out_dir).map_err(|e| Error::Backend(format!("{}: {e}", out_dir.display())))?;
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress::of(s));
            crate::data_parallel::keep_going(self.parallel.as_ref(), s.step, cancel)
        };
        let control = model::FitControl { on_step: Some(&mut on_step), state: Some(&state_path), state_every: self.checkpoint_every, identity, snapshots: None, publish: None, grad_sync: self.parallel.as_ref().map(crate::DataParallel::grad_sync) };
        let adapter_path = out_dir.join(ADAPTER_FILE);
        let adapter_id = self.adapter_id.clone().unwrap_or_else(|| format!("{base_id}:local:distill:latest"));
        let provenance = checkpoint::st::TrainingProvenance {
            code_revision: format!("brain {}", env!("CARGO_PKG_VERSION")),
            regime: "distill".to_string(),
            seed,
            hyperparams: serde_json::json!({ "rank": rank, "alpha": alpha, "steps": self.steps, "lr": self.lr, "block": block, "top_k": self.top_k, "teacher": self.teacher.name(), "teacher_adapter": teacher_digest, "examples": examples.len() }),
            environment: gpu_core::backend_name().to_string(),
            gate: None,
            trained_from: parent.clone(),
            base_digest: Some(base_digest.clone()),
            cycle: 0,
        };
        let example_count = examples.len();
        let job = DistillJob {
            base: &open,
            lora: model::finetune::LoraRequest { rank, alpha, continue_from: self.continue_from.as_deref(), base: &base_tiers, modules_to_save: &[] },
            opts: &opts,
            examples,
            adapter: &adapter_path,
            lineage: model::finetune::AdapterLineage { card_id: &adapter_id, base_id: &base_id, dataset_id: Some(&dataset_id), provenance: Some(provenance) },
        };
        let report = (chat.distill)(job, control).map_err(|e| Error::Backend(format!("training: {e}")))?;
        let mut outcome = DistillFineTuneOutcome {
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
            examples: example_count,
            block,
            rank,
            alpha,
            top_k: self.top_k,
            teacher: self.teacher.name(),
            teacher_adapter: teacher_digest,
            trained_from: parent,
            base_digest: Some(base_digest),
            parallel: crate::data_parallel::ParallelReport::of(self.parallel.as_ref(), ran_on),
        };
        if report.interrupted {
            outcome.status = FineTuneStatus::Cancelled;
            outcome.resume_state = Some(state_path);
            return Ok(outcome);
        }
        outcome.adapter_digest = Some(digest(&adapter_path)?);
        outcome.adapter = Some(adapter_path);
        let record_path = out_dir.join(RECORD_FILE);
        std::fs::write(&record_path, serde_json::to_vec_pretty(&outcome.record_json(&base_id, &dataset_id)).map_err(|e| Error::Backend(e.to_string()))?).map_err(|e| Error::Backend(format!("{}: {e}", record_path.display())))?;
        outcome.record = Some(record_path);
        std::fs::remove_file(&state_path).ok();
        Ok(outcome)
    }
}

/// What one distillation did. Anything not measured is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct DistillFineTuneOutcome {
    pub status: FineTuneStatus,
    pub adapter: Option<PathBuf>,
    /// `sha256:<hex>` of the adapter file.
    pub adapter_digest: Option<String>,
    pub record: Option<PathBuf>,
    pub resume_state: Option<PathBuf>,
    pub seed: u64,
    pub steps: u32,
    pub steps_completed: u32,
    pub resumed_at: Option<u32>,
    /// The student's mean cross-entropy against the teacher's renormalised top-K per teacher-scored position (its
    /// `KL(q_K || p)` plus the teacher's entropy, a constant of the run) before this call's first step, and after its
    /// last.
    pub initial_loss: Option<f32>,
    pub final_loss: Option<f32>,
    /// Records the teacher scored (one example each).
    pub examples: usize,
    pub block: u32,
    pub rank: u32,
    pub alpha: f32,
    pub top_k: usize,
    /// `base`, `start` or `adapter`.
    pub teacher: &'static str,
    /// `sha256:<hex>` of the teacher's adapter, when it has one.
    pub teacher_adapter: Option<String>,
    pub trained_from: Option<String>,
    pub base_digest: Option<String>,
    /// How the run was spread over ranks, cards and servers, as it ran ([`crate::ParallelReport`]).
    pub parallel: crate::ParallelReport,
}

impl DistillFineTuneOutcome {
    fn record_json(&self, base_id: &str, dataset_id: &str) -> serde_json::Value {
        serde_json::json!({
            "regime": "distill",
            "base": base_id,
            "dataset": dataset_id,
            "adapter_digest": self.adapter_digest,
            "seed": self.seed,
            "trained_from": self.trained_from,
            "teacher": self.teacher,
            "teacher_adapter": self.teacher_adapter,
            "top_k": self.top_k,
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "examples": self.examples,
            "block": self.block,
            "rank": self.rank,
            "alpha": self.alpha,
            "base_digest": self.base_digest,
            "parallel": self.parallel,
        })
    }
}
