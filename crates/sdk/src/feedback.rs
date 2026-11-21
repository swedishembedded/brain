// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of chat models on a product's own thumbs-up and thumbs-down feedback
// for its clients. If your team needs expertise in feedback optimization (KTO) and its evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! [`FeedbackFineTune`]: LoRA fine-tuning of a chat model by KTO on `generic-feedback-v1` judged completions, one
//! completion per example, no pair needed.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::FeedbackFineTune::from_pretrained("Qwen/Qwen3-4B")
//!     .dataset("feedback.jsonl")
//!     .out_dir("runs/feedback-1")
//!     .steps(200)
//!     .grad_accum(8)
//!     .run()?;
//! println!("desirable {:?}, undesirable {:?} (mean log-ratio to the start)", outcome.desirable_logratio, outcome.undesirable_logratio);
//! # Ok(()) }
//! ```
//!
//! **The objective** (`rl::objective::feedback`): with `r` the completion's log-probability under the policy less its
//! log-probability under the reference (the model the run starts from, frozen and scored once per example before the
//! first step), and `z` the step's estimate of the policy's divergence from the reference (the mean log-ratio of
//! mismatched pairs, each example's prompt with the next example's completion, never below 0), a desirable example costs
//! `desirable_weight * (1 - sigmoid(beta * (r - z)))` and an undesirable one `undesirable_weight * (1 - sigmoid(beta *
//! (z - r)))`. A step of `grad_accum` examples forwards each twice (once for `z`, once to train).
//!
//! Everything else is [`crate::PreferenceFineTune`]'s contract: the dataset is checked against the base's own
//! tokenizer and chat template before a device is claimed, progress and cancellation go through
//! [`crate::CancelToken`], a cancelled run resumes exactly, and the adapter's card records its provenance.

use std::path::{Path, PathBuf};

use data::chat::RenderOpts;
use data::chat_template::ChatTemplate;
use data::feedback::FeedbackSample;
use data::preference::EncodedTurn;
use data::qwen_tokenizer::QwenBpe;
use model::training::Capability;
use rl::objective::feedback::{FeedbackConfig, PackedFeedback, FEEDBACK_ROWS};

use crate::chat_trainer::FeedbackJob;
use crate::finetune::{base_digest, block_for, digest, fit_opts, lora_shape, resolve_seed, tokenizer_and_template};
use crate::{CancelToken, Device, Error, FineTuneProgress, FineTuneStatus, Result};

pub use rl::objective::feedback::{DEFAULT_KTO_BETA, DEFAULT_KTO_DESIRABLE_WEIGHT, DEFAULT_KTO_UNDESIRABLE_WEIGHT};

const STATE_FILE: &str = "train.state";
const ADAPTER_FILE: &str = "adapter.safetensors";
const RECORD_FILE: &str = "training.json";

/// A KTO LoRA fine-tune of a chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct FeedbackFineTune {
    base: String,
    arch: Option<String>,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    continue_from: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    adapter_id: Option<String>,
    rank: Option<u32>,
    alpha: Option<f32>,
    beta: f32,
    desirable_weight: f32,
    undesirable_weight: f32,
    grad_accum: u32,
    strict_updates: bool,
    keep_reasoning: bool,
    steps: u32,
    lr: f32,
    seed: Option<u64>,
    max_block: Option<u32>,
    checkpoint_every: u32,
    resume_state: Option<PathBuf>,
    cycle: u64,
    device: Device,
    placed: bool,
    base_format: Option<String>,
}

impl FeedbackFineTune {
    /// A feedback fine-tune of `base`, resolved like [`crate::ChatFineTune::from_pretrained`]'s.
    pub fn from_pretrained(base: impl Into<String>) -> FeedbackFineTune {
        FeedbackFineTune {
            base: base.into(),
            arch: None,
            models_dir: None,
            dataset: None,
            continue_from: None,
            out_dir: None,
            adapter_id: None,
            rank: None,
            alpha: None,
            beta: DEFAULT_KTO_BETA,
            desirable_weight: DEFAULT_KTO_DESIRABLE_WEIGHT,
            undesirable_weight: DEFAULT_KTO_UNDESIRABLE_WEIGHT,
            grad_accum: 1,
            strict_updates: false,
            keep_reasoning: crate::DEFAULT_DPO_KEEP_REASONING,
            steps: 100,
            lr: crate::finetune::DEFAULT_LORA_LR,
            seed: None,
            max_block: None,
            checkpoint_every: 0,
            resume_state: None,
            cycle: 0,
            device: Device::default(),
            placed: false,
            base_format: None,
        }
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

    /// The `generic-feedback-v1` examples to train on (required, at least two).
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// Continue this adapter; base plus it is then the frozen reference.
    pub fn continue_from(mut self, adapter: impl Into<PathBuf>) -> Self {
        self.continue_from = Some(adapter.into());
        self
    }

    /// Where the adapter, its training record and any resume state go (required).
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// The id written on the adapter's card (default `<base id>:local:feedback:latest`).
    pub fn adapter_id(mut self, id: impl Into<String>) -> Self {
        self.adapter_id = Some(id.into());
        self
    }

    /// LoRA rank (default 8, or a continued adapter's own).
    pub fn rank(mut self, rank: u32) -> Self {
        self.rank = Some(rank);
        self
    }

    /// LoRA alpha (default `2 * rank`, or a continued adapter's own).
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// `beta`, scaling the log-ratio's distance from the reference point (default [`DEFAULT_KTO_BETA`]).
    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// `lambda_D`, the weight of a desirable example's term (default 1). The paper balances the two weights so that
    /// `desirable_weight * desirable examples` is about `undesirable_weight * undesirable examples`.
    pub fn desirable_weight(mut self, weight: f32) -> Self {
        self.desirable_weight = weight;
        self
    }

    /// `lambda_U`, the weight of an undesirable example's term (default 1).
    pub fn undesirable_weight(mut self, weight: f32) -> Self {
        self.undesirable_weight = weight;
        self
    }

    /// Examples per optimizer step (default 1): the reference point is estimated over them.
    pub fn grad_accum(mut self, examples: u32) -> Self {
        self.grad_accum = examples;
        self
    }

    /// Abort at the first optimiser step whose loss or gradient is not finite (default: skip it, abort on the third).
    pub fn strict_updates(mut self, strict: bool) -> Self {
        self.strict_updates = strict;
        self
    }

    /// Train on the reasoning of each completion (default off); see [`crate::PreferenceFineTune::keep_reasoning`].
    pub fn keep_reasoning(mut self, on: bool) -> Self {
        self.keep_reasoning = on;
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

    /// Seeds the fresh adapter and the example order; unset, a random one (reported).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The longest training row; a longer example is refused by name.
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

    /// The caller's loop iteration, recorded in the adapter's provenance (default 0).
    pub fn cycle(mut self, cycle: u64) -> Self {
        self.cycle = cycle;
        self
    }

    /// Hold the frozen base in the named format; see [`crate::PreferenceFineTune::base_format`].
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
    pub fn run(&self) -> Result<FeedbackFineTuneOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step and stopping at the next step boundary once `cancel` fires. A cancelled run
    /// exports no adapter and leaves its resume state.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<FeedbackFineTuneOutcome> {
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let arch = match self.arch.as_deref() {
            Some(named) => crate::study::arch(named)?,
            None => crate::study::arch_of_checkpoint(&model_dir).map_or_else(|| crate::study::arch("qwen3"), Ok)?,
        };
        arch.require(Capability::ChatFineTune)?;
        arch.require(Capability::TokenObjectives)?;
        let chat = arch.chat.ok_or_else(|| Error::Unsupported(model::training::Unsupported::new(arch.id, Capability::ChatFineTune)))?;
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("FeedbackFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("FeedbackFineTune: no out directory; call .out_dir(path)".to_string()))?;
        if self.grad_accum == 0 {
            return Err(Error::InvalidRequest("FeedbackFineTune: grad_accum must be at least 1".to_string()));
        }
        let config = |seq_len: usize| FeedbackConfig { beta: self.beta, desirable_weight: self.desirable_weight, undesirable_weight: self.undesirable_weight, seq_len };
        config(2).validate().map_err(|why| Error::InvalidRequest(format!("FeedbackFineTune: {why}")))?;
        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let render = RenderOpts { keep_reasoning: self.keep_reasoning };
        let examples = check_feedback(dataset, &tok, &tmpl, self.max_block, render).map_err(Error::Backend)?;
        let longest = examples.iter().map(|e| e.own.ids.len().max(e.mismatched.ids.len())).max().unwrap_or(0);
        let block = block_for(longest, self.max_block)?;
        let base_format = match self.base_format.as_deref() {
            Some(spec) => chat.lora_bases.parse(spec).map_err(Error::Backend)?,
            None if self.placed => {
                let card_bytes = gpu_core::devices::current_gpu().and_then(|index| gpu_core::devices::gpus().iter().find(|d| d.index == index)).map(|d| d.identity.vram_bytes);
                let mut sizing = crate::ChatFineTune::from_pretrained(self.base.as_str()).arch(arch.id);
                if let Some(dir) = &self.models_dir {
                    sizing = sizing.models_dir(dir.as_str());
                }
                chat.lora_bases.parse(&sizing.footprint(block.saturating_mul(FEEDBACK_ROWS), card_bytes)?.base_tier).map_err(Error::Backend)?
            }
            None => gpu_core::select::Dtype::F32,
        };
        let base_tiers = model::ops::TierPolicy::uniform(base_format);
        let (rank, alpha, parent) = lora_shape(self.continue_from.as_deref(), self.rank, self.alpha, Default::default())?;
        let seq_len = block as usize;
        let packed = examples
            .iter()
            .enumerate()
            .map(|(i, e)| PackedFeedback::from_masked(seq_len, (&e.own.ids, &e.own.mask), (&e.mismatched.ids, &e.mismatched.mask), e.desirable).map_err(|why| Error::Backend(format!("{}: record {}: {why}", dataset.display(), i + 1))))
            .collect::<Result<Vec<_>>>()?;
        if !self.placed {
            crate::device::resolve(&self.device)?;
        }
        let state_path = self.resume_state.clone().unwrap_or_else(|| out_dir.join(STATE_FILE));
        let seed = resolve_seed(self.seed, Some(&state_path))?;
        let requested = model::FitOpts { batch_size: FEEDBACK_ROWS, grad_accum: self.grad_accum, strict_updates: self.strict_updates, ..fit_opts(self.steps, block, self.lr, seed) };
        let cfg = config(seq_len);
        let opts = model::training_plan::ValidatedTrainingPlan::build(requested, &[&cfg]).map_err(|e| Error::InvalidRequest(format!("FeedbackFineTune: {e}")))?.into_opts();
        let base_digest = base_digest(&weights)?;
        let dataset_id = digest(dataset)?;
        let identity = serde_json::json!({
            "regime": "feedback",
            "base": base_digest,
            "dataset": dataset_id,
            "parent": parent,
            "rank": rank,
            "alpha_bits": alpha.to_bits(),
            "beta_bits": self.beta.to_bits(),
            "desirable_weight_bits": self.desirable_weight.to_bits(),
            "undesirable_weight_bits": self.undesirable_weight.to_bits(),
            "reference": { "base_digest": base_digest, "adapter_digest": parent, "base_format": self.base_format },
            "grad_accum": self.grad_accum,
            "keep_reasoning": self.keep_reasoning,
        });
        std::fs::create_dir_all(out_dir).map_err(|e| Error::Backend(format!("{}: {e}", out_dir.display())))?;
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress::of(s));
            !cancel.is_cancelled()
        };
        let control = model::FitControl { on_step: Some(&mut on_step), state: Some(&state_path), state_every: self.checkpoint_every, identity, snapshots: None, grad_sync: None };
        let adapter_path = out_dir.join(ADAPTER_FILE);
        let adapter_id = self.adapter_id.clone().unwrap_or_else(|| format!("{base_id}:local:feedback:latest"));
        let (desirable, undesirable) = (examples.iter().filter(|e| e.desirable).count(), examples.iter().filter(|e| !e.desirable).count());
        let provenance = checkpoint::st::TrainingProvenance {
            code_revision: format!("brain {}", env!("CARGO_PKG_VERSION")),
            regime: "feedback".to_string(),
            seed,
            hyperparams: serde_json::json!({
                "rank": rank,
                "alpha": alpha,
                "beta": self.beta,
                "desirable_weight": self.desirable_weight,
                "undesirable_weight": self.undesirable_weight,
                "grad_accum": self.grad_accum,
                "keep_reasoning": self.keep_reasoning,
                "steps": self.steps,
                "lr": self.lr,
                "block": block,
                "desirable_examples": desirable,
                "undesirable_examples": undesirable,
                "reference": { "base_digest": base_digest, "adapter_digest": parent },
            }),
            environment: gpu_core::backend_name().to_string(),
            gate: None,
            trained_from: parent.clone(),
            base_digest: Some(base_digest.clone()),
            cycle: self.cycle,
        };
        let job = FeedbackJob {
            base: &weights,
            lora: model::finetune::LoraRequest { rank, alpha, continue_from: self.continue_from.as_deref(), base: &base_tiers, modules_to_save: &[] },
            opts: &opts,
            cfg,
            examples: packed,
            adapter: &adapter_path,
            lineage: model::finetune::AdapterLineage { card_id: &adapter_id, base_id: &base_id, dataset_id: Some(&dataset_id), provenance: Some(provenance) },
        };
        let (report, logratios) = (chat.feedback)(job, control).map_err(|e| Error::Backend(format!("training: {e}")))?;
        let mean_of = |want: bool| {
            let picked: Vec<f64> = examples.iter().zip(&logratios).filter(|(e, _)| e.desirable == want).filter_map(|(_, r)| r.map(f64::from)).collect();
            (!picked.is_empty()).then(|| (picked.iter().sum::<f64>() / picked.len() as f64) as f32)
        };
        let mut outcome = FeedbackFineTuneOutcome {
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
            desirable_examples: desirable,
            undesirable_examples: undesirable,
            block,
            rank,
            alpha,
            beta: self.beta,
            desirable_weight: self.desirable_weight,
            undesirable_weight: self.undesirable_weight,
            trained_from: parent,
            base_digest: Some(base_digest),
            desirable_logratio: mean_of(true),
            undesirable_logratio: mean_of(false),
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

/// What one feedback fine-tune did. Anything not measured is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackFineTuneOutcome {
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
    /// The KTO loss estimated before this call's first step.
    pub initial_loss: Option<f32>,
    pub final_loss: Option<f32>,
    pub desirable_examples: usize,
    pub undesirable_examples: usize,
    /// The training row length, sized to the longest example (or its mismatched rendering).
    pub block: u32,
    pub rank: u32,
    pub alpha: f32,
    pub beta: f32,
    pub desirable_weight: f32,
    pub undesirable_weight: f32,
    pub trained_from: Option<String>,
    pub base_digest: Option<String>,
    /// The mean over the desirable training examples the run drew of `log pi(y | x) - log ref(y | x)` after the run
    /// (it should be above zero), and the same over the undesirable ones (below); `None` when the run did not complete
    /// or drew none. An example no step drew has no reference score, which is measured when a step first draws it.
    pub desirable_logratio: Option<f32>,
    pub undesirable_logratio: Option<f32>,
}

impl FeedbackFineTuneOutcome {
    fn record_json(&self, base_id: &str, dataset_id: &str) -> serde_json::Value {
        serde_json::json!({
            "regime": "feedback",
            "base": base_id,
            "dataset": dataset_id,
            "adapter_digest": self.adapter_digest,
            "seed": self.seed,
            "trained_from": self.trained_from,
            "reference": { "base_digest": self.base_digest, "adapter_digest": self.trained_from },
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "desirable_examples": self.desirable_examples,
            "undesirable_examples": self.undesirable_examples,
            "block": self.block,
            "rank": self.rank,
            "alpha": self.alpha,
            "beta": self.beta,
            "desirable_weight": self.desirable_weight,
            "undesirable_weight": self.undesirable_weight,
            "desirable_logratio": self.desirable_logratio,
            "undesirable_logratio": self.undesirable_logratio,
        })
    }
}

/// One example rendered for training: its own conversation, its prompt with the next example's completion, and the
/// judgement.
struct EncodedFeedback {
    own: EncodedTurn,
    mismatched: EncodedTurn,
    desirable: bool,
}

/// Parse `path` and render every example and its mismatched pair (its prompt, the next example's completion) through
/// the base's template, refusing a record that does not render, has no supervised token or is past `max_block`, and
/// a dataset of fewer than two examples (the reference point is estimated from mismatched pairs).
fn check_feedback(path: &Path, tok: &QwenBpe, tmpl: &ChatTemplate, max_block: Option<u32>, render: RenderOpts) -> std::result::Result<Vec<EncodedFeedback>, String> {
    let examples = FeedbackSample::from_jsonl(path).map_err(|e| e.to_string())?;
    if examples.len() < 2 {
        return Err(format!("{}: {} example(s); the reference point is estimated from mismatched pairs, which needs at least two", path.display(), examples.len()));
    }
    let mut out = Vec::with_capacity(examples.len());
    for (index, example) in examples.iter().enumerate() {
        let record = || format!("{}: record {}", path.display(), index + 1);
        let own = example.encode_with(tok, tmpl, render).map_err(|e| format!("{} cannot be rendered for training: {e}", record()))?;
        let next = &examples[(index + 1) % examples.len()];
        let (ids, mask) = example.with_completion_of(next).encode_candidate(tok, tmpl, render).map_err(|e| format!("{}: its prompt with record {}'s completion cannot be rendered: {e}", record(), (index + 1) % examples.len() + 1))?;
        let mismatched = EncodedTurn { ids, mask };
        for (name, turn) in [("completion", &own), ("mismatched completion", &mismatched)] {
            if !turn.mask.iter().any(|m| *m) {
                return Err(format!("{}: the {name} renders to no supervised token", record()));
            }
            if let Some(max) = max_block.filter(|&max| turn.ids.len() > max as usize) {
                return Err(format!("{}: the {name} is {} tokens, past max_block {max}; raise max_block or shorten the record", record(), turn.ids.len()));
            }
        }
        out.push(EncodedFeedback { own, mismatched, desirable: example.desirable });
    }
    Ok(out)
}
