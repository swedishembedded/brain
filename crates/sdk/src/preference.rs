// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements preference fine-tuning of chat models on a
// product's own reviewed answers for its clients. If your team needs
// expertise in preference optimization (DPO) and its evaluation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! [`PreferenceFineTune`]: LoRA fine-tuning of a Qwen3 chat model by direct
//! preference optimization (DPO) on `generic-preference-v1` pairs, and
//! [`score_preference`]: the held-out measurement of how much more the tuned
//! model prefers the chosen answers than the base does.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::PreferenceFineTune::from_pretrained("/models/qwen3-0.6b/model.brain.safetensors")
//!     .dataset("pairs.jsonl")
//!     .held_out("held_out_pairs.jsonl")
//!     .out_dir("runs/prefs-1")
//!     .steps(200)
//!     .run()?;
//! if let Some(score) = outcome.held_out_score {
//!     println!("held-out accuracy {:?}, mean margin {:?}", score.accuracy, score.mean_margin);
//! }
//! # Ok(()) }
//! ```
//!
//! **The objective.** Standard DPO: per pair, `-log sigmoid(beta * ((log
//! pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))))`,
//! each log-probability summed over the candidate's own assistant-turn tokens
//! (the positions [`crate::ChatFineTune`] would supervise). The reference is
//! the model the run starts from - the base, or the base plus the adapter
//! passed to [`PreferenceFineTune::continue_from`] - frozen: it is scored once
//! per pair before the first step and never again, so no second model is held
//! during training. It trains through `rl::objective::dpo::Dpo` over
//! `model::fit_controlled`, the one training loop, one pair per step.
//!
//! Everything else is [`crate::ChatFineTune`]'s contract: the dataset is
//! validated against the base's own tokenizer and chat template
//! ([`crate::validate_preference_dataset_for`]) before a device is claimed,
//! progress and cancellation go through [`crate::CancelToken`], a cancelled
//! run resumes exactly, and the exported adapter's card records its training
//! provenance, including the base's digest.

use std::path::{Path, PathBuf};

use data::chat_template::ChatTemplate;
use data::preference::EncodedTurn;
use data::qwen_tokenizer::QwenBpe;

use crate::finetune::{block_for, digest, fit_opts, lora_shape, tokenizer_and_template, utf8};
use crate::{CancelToken, Device, Error, FineTuneProgress, FineTuneStatus, Result};

const STATE_FILE: &str = "train.state";
const ADAPTER_FILE: &str = "adapter.safetensors";
const RECORD_FILE: &str = "training.json";

/// The DPO temperature used when [`PreferenceFineTune::beta`] is not called:
/// the conventional DPO default.
pub const DEFAULT_DPO_BETA: f32 = 0.1;

/// A DPO LoRA fine-tune of a Qwen3 chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct PreferenceFineTune {
    base: String,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    held_out: Option<PathBuf>,
    continue_from: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    adapter_id: Option<String>,
    rank: Option<u32>,
    alpha: Option<f32>,
    beta: f32,
    nll_weight: f32,
    grad_accum: u32,
    keep_reasoning: bool,
    steps: u32,
    lr: f32,
    seed: u64,
    max_block: Option<u32>,
    checkpoint_every: u32,
    cycle: u64,
    device: Device,
}

impl PreferenceFineTune {
    /// A preference fine-tune of `base`, resolved like
    /// [`crate::ChatFineTune::from_pretrained`]'s; the tokenizer and chat
    /// template are read from the checkpoint's directory.
    pub fn from_pretrained(base: impl Into<String>) -> PreferenceFineTune {
        PreferenceFineTune {
            base: base.into(),
            models_dir: None,
            dataset: None,
            held_out: None,
            continue_from: None,
            out_dir: None,
            adapter_id: None,
            rank: None,
            alpha: None,
            beta: DEFAULT_DPO_BETA,
            nll_weight: 0.0,
            grad_accum: 1,
            keep_reasoning: false,
            steps: 100,
            lr: 3e-4,
            seed: 1337,
            max_block: None,
            checkpoint_every: 0,
            cycle: 0,
            device: Device::default(),
        }
    }

    /// The models directory a `vendor/repo` base resolves in.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The `generic-preference-v1` pairs to train on (required).
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// Pairs to score the tuned adapter on, never trained on. Without it the
    /// outcome's held-out score is `None`.
    pub fn held_out(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out = Some(path.into());
        self
    }

    /// Continue training this adapter instead of starting a fresh one. Base
    /// plus this adapter is then the frozen reference; its rank and alpha are
    /// the run's, and its digest is recorded as the new adapter's parent.
    pub fn continue_from(mut self, adapter: impl Into<PathBuf>) -> Self {
        self.continue_from = Some(adapter.into());
        self
    }

    /// Where the adapter, its training record and any resume state go
    /// (required). One directory per run.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// The id written on the adapter's card (default `<base id>:local:dpo:latest`).
    pub fn adapter_id(mut self, id: impl Into<String>) -> Self {
        self.adapter_id = Some(id.into());
        self
    }

    /// LoRA rank (default 8, or a continued adapter's own; a different one is
    /// refused).
    pub fn rank(mut self, rank: u32) -> Self {
        self.rank = Some(rank);
        self
    }

    /// LoRA alpha (default `2 * rank`, or a continued adapter's own).
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    /// The DPO temperature `beta` scaling the reference-normalized margin
    /// inside the sigmoid (default [`DEFAULT_DPO_BETA`]).
    pub fn beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Weight of an anchor on the chosen answer: its mean negative
    /// log-likelihood is added to the DPO loss (default 0, plain DPO). DPO
    /// alone can lower the probability of both answers while widening the
    /// margin between them; the anchor keeps the chosen one from falling.
    pub fn nll_weight(mut self, weight: f32) -> Self {
        self.nll_weight = weight;
        self
    }

    /// Pairs summed into each optimizer step (default 1), the effective
    /// batch size. One pair per update is a noisy gradient for a preference
    /// objective.
    pub fn grad_accum(mut self, pairs: u32) -> Self {
        self.grad_accum = pairs;
        self
    }

    /// Train on the reasoning of each candidate (default off). A reasoning
    /// model's template drops a closed think block from the assistant turns it
    /// renders as history, which is how every candidate is rendered here; with
    /// this set the block is kept, so a model that is asked from an empty
    /// closed block trains on exactly that form. See
    /// [`crate::ChatFineTune::keep_reasoning`].
    pub fn keep_reasoning(mut self, on: bool) -> Self {
        self.keep_reasoning = on;
        self
    }

    /// Optimizer steps, one pair each (default 100). The first fifth warms
    /// the rate up, and it decays over all of them.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }

    /// Peak learning rate (default 3e-4); it decays to a tenth of this.
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Seeds the fresh adapter's initialisation and the pair order (default 1337).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The longest training row allowed. Rows are sized to the longest
    /// candidate; a pair with a longer one is refused by name rather than
    /// trained on truncated.
    pub fn max_block(mut self, tokens: u32) -> Self {
        self.max_block = Some(tokens);
        self
    }

    /// Also save the resume state every this-many steps (default 0: only when
    /// cancelled).
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

    /// Run to completion.
    pub fn run(&self) -> Result<PreferenceFineTuneOutcome> {
        self.run_with(&CancelToken::default(), |_| {})
    }

    /// Run, reporting every optimizer step to `on_progress` (its loss is
    /// that step's DPO loss) and stopping at the next step boundary once
    /// `cancel` fires. A cancelled run exports no adapter; it leaves its
    /// resume state, and running the same fine-tune again continues it.
    pub fn run_with(&self, cancel: &CancelToken, mut on_progress: impl FnMut(&FineTuneProgress)) -> Result<PreferenceFineTuneOutcome> {
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("PreferenceFineTune: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("PreferenceFineTune: no out directory; call .out_dir(path)".to_string()))?;
        if !(self.nll_weight.is_finite() && self.nll_weight >= 0.0) {
            return Err(Error::Backend(format!("PreferenceFineTune: nll_weight must be zero or positive, got {}", self.nll_weight)));
        }
        if self.grad_accum == 0 {
            return Err(Error::Backend("PreferenceFineTune: grad_accum must be at least 1".to_string()));
        }
        if !(self.beta.is_finite() && self.beta > 0.0) {
            return Err(Error::Backend(format!("PreferenceFineTune: beta must be a positive number, got {}", self.beta)));
        }
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, base_id) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let weights_str = utf8(&weights)?;

        // Every file is checked against the base's own template before a
        // device is claimed.
        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let render = data::chat::RenderOpts { keep_reasoning: self.keep_reasoning };
        let train = encode_checked(dataset, &tok, &tmpl, self.max_block, render)?;
        let held_out = self.held_out.as_deref().map(|path| encode_checked(path, &tok, &tmpl, None, render)).transpose()?;
        let (rank, alpha, parent) = lora_shape(self.continue_from.as_deref(), self.rank, self.alpha)?;
        let block = block_for(longest(&train), self.max_block)?;
        let seq_len = block as usize;
        let packed = train
            .iter()
            .enumerate()
            .map(|(i, (c, r))| rl::objective::dpo::PackedPair::from_masked(seq_len, (&c.ids, &c.mask), (&r.ids, &r.mask)).map_err(|e| Error::Backend(format!("{}: record {}: {e}", dataset.display(), i + 1))))
            .collect::<Result<Vec<_>>>()?;

        crate::device::resolve(&self.device)?;

        // Both rows of a pair share one forward.
        let opts = model::FitOpts { batch_size: 2, grad_accum: self.grad_accum, ..fit_opts(self.steps, block, self.lr, self.seed) };
        let base_digest = digest(&weights)?;
        let dataset_id = digest(dataset)?;
        let identity = serde_json::json!({
            "regime": "dpo",
            "base": base_digest,
            "dataset": dataset_id,
            "parent": parent,
            "rank": rank,
            "alpha_bits": alpha.to_bits(),
            "beta_bits": self.beta.to_bits(),
            "nll_weight_bits": self.nll_weight.to_bits(),
            "grad_accum": self.grad_accum,
            "keep_reasoning": self.keep_reasoning,
        });
        let state_path = out_dir.join(STATE_FILE);
        std::fs::create_dir_all(out_dir).map_err(|e| Error::Backend(format!("{}: {e}", out_dir.display())))?;
        let mut on_step = |s: &model::StepReport| {
            on_progress(&FineTuneProgress { step: s.step, steps: s.steps, loss: s.loss, lr: s.lr });
            !cancel.is_cancelled()
        };
        let control = model::FitControl { on_step: Some(&mut on_step), state: Some(&state_path), state_every: self.checkpoint_every, identity };
        let start = match &self.continue_from {
            Some(path) => qwen3::finetune::LoraStart::Continue(utf8(path)?),
            None => qwen3::finetune::LoraStart::Fresh,
        };
        let policy = lora_model(weights_str, rank, alpha, &opts, &start)?;
        let objective = rl::objective::dpo::Dpo::from_dataset(rl::objective::dpo::DpoConfig { beta: self.beta, seq_len, nll_weight: self.nll_weight }, packed);
        let fit_err = |e: std::io::Error| Error::Backend(format!("training: {e}"));
        let (report, trained) = match policy {
            qwen3::finetune::Trained::Single(m) => {
                let (r, m) = model::fit_controlled(m, objective, &opts, None, control).map_err(fit_err)?;
                (r, qwen3::finetune::Trained::Single(m))
            }
            qwen3::finetune::Trained::Pipeline(m) => {
                let (r, m) = model::fit_controlled(m, objective, &opts, None, control).map_err(fit_err)?;
                (r, qwen3::finetune::Trained::Pipeline(m))
            }
        };

        let mut outcome = PreferenceFineTuneOutcome {
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
            train_pairs: train.len(),
            block,
            rank,
            alpha,
            beta: self.beta,
            nll_weight: self.nll_weight,
            trained_from: parent.clone(),
            base_digest: Some(base_digest.clone()),
            train_score: None,
            held_out_score: None,
        };
        if report.interrupted {
            outcome.status = FineTuneStatus::Cancelled;
            outcome.resume_state = Some(state_path);
            return Ok(outcome);
        }

        let adapter_path = out_dir.join(ADAPTER_FILE);
        let adapter_id = self.adapter_id.clone().unwrap_or_else(|| format!("{base_id}:local:dpo:latest"));
        let provenance = checkpoint::st::TrainingProvenance {
            code_revision: format!("brain {}", env!("CARGO_PKG_VERSION")),
            regime: "dpo".to_string(),
            seed: self.seed,
            hyperparams: serde_json::json!({
                "rank": rank,
                "alpha": alpha,
                "beta": self.beta,
                "nll_weight": self.nll_weight,
                "grad_accum": self.grad_accum,
                "keep_reasoning": self.keep_reasoning,
                "steps": self.steps,
                "lr": self.lr,
                "block": block,
                "train_pairs": outcome.train_pairs,
                "reference": { "base_digest": base_digest, "adapter_digest": parent },
            }),
            environment: gpu_core::backend_name().to_string(),
            gate: None,
            trained_from: parent,
            base_digest: Some(base_digest),
            cycle: self.cycle,
        };
        trained.save_adapter_with_lineage(utf8(&adapter_path)?, &adapter_id, &base_id, Some(&dataset_id), Some(provenance))
            .map_err(|e| Error::Backend(format!("{}: saving the adapter: {e}", adapter_path.display())))?;
        drop(trained);
        outcome.adapter_digest = Some(digest(&adapter_path)?);

        let reference = self.continue_from.as_deref().map(utf8).transpose()?;
        let adapter_str = utf8(&adapter_path)?;
        outcome.train_score = Some(score_pairs(weights_str, reference, adapter_str, &train));
        outcome.held_out_score = held_out.as_ref().map(|pairs| score_pairs(weights_str, reference, adapter_str, pairs));
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

/// What one preference fine-tune did. Anything not measured is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceFineTuneOutcome {
    pub status: FineTuneStatus,
    /// The adapter file; `None` unless completed.
    pub adapter: Option<PathBuf>,
    /// `sha256:<hex>` of the adapter file.
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
    /// The DPO loss estimated before this call's first step. On a fresh or
    /// continued start the policy is the reference, so this is `ln 2`.
    pub initial_loss: Option<f32>,
    /// The last step's DPO loss; `None` when this call ran no step.
    pub final_loss: Option<f32>,
    pub train_pairs: usize,
    /// The training row length, sized to the longest candidate.
    pub block: u32,
    pub rank: u32,
    pub alpha: f32,
    pub beta: f32,
    pub nll_weight: f32,
    /// The digest of the adapter this run continued - also the reference's
    /// adapter - if it continued one.
    pub trained_from: Option<String>,
    /// `sha256:<hex>` of the base checkpoint file the reference (and the
    /// adapter) is built on - the digest the adapter's card records as
    /// `TrainingProvenance::base_digest`. `None` when the run did not compute
    /// it.
    pub base_digest: Option<String>,
    /// The tuned adapter against the run's reference on the training pairs;
    /// `None` unless completed.
    pub train_score: Option<PreferenceScore>,
    /// The same on the held-out pairs; `None` unless completed with a
    /// held-out set.
    pub held_out_score: Option<PreferenceScore>,
}

impl PreferenceFineTuneOutcome {
    fn record_json(&self, base_id: &str, dataset_id: &str) -> serde_json::Value {
        let score = |s: &Option<PreferenceScore>| s.as_ref().map(|s| serde_json::json!({ "accuracy": s.accuracy, "mean_margin": s.mean_margin, "pairs": s.pairs, "skipped": s.skipped }));
        serde_json::json!({
            "regime": "dpo",
            "base": base_id,
            "dataset": dataset_id,
            "adapter_digest": self.adapter_digest,
            "trained_from": self.trained_from,
            "reference": { "base_digest": self.base_digest, "adapter_digest": self.trained_from },
            "steps": self.steps,
            "steps_completed": self.steps_completed,
            "resumed_at": self.resumed_at,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "train_pairs": self.train_pairs,
            "block": self.block,
            "rank": self.rank,
            "alpha": self.alpha,
            "beta": self.beta,
            "nll_weight": self.nll_weight,
            "train_score": score(&self.train_score),
            "held_out_score": score(&self.held_out_score),
        })
    }
}

/// How much more a tuned model prefers each pair's chosen answer over its
/// rejected one than its reference does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreferenceScore {
    /// Fraction of scored pairs whose margin is positive; `None` when no
    /// pair was scored.
    pub accuracy: Option<f32>,
    /// Mean over scored pairs of `(log pi(chosen) - log ref(chosen)) - (log
    /// pi(rejected) - log ref(rejected))`, in nats, each log-probability
    /// summed over the candidate's assistant-turn tokens (no `beta`); `None`
    /// when no pair was scored.
    pub mean_margin: Option<f32>,
    /// Pairs scored.
    pub pairs: usize,
    /// Pairs skipped because a candidate does not fit the model's context.
    pub skipped: usize,
}

/// Score `base` with `adapter` folded in - exactly as serving folds it -
/// against `base` alone on the `generic-preference-v1` pairs in `dataset`.
/// `base` is resolved like [`PreferenceFineTune::from_pretrained`]'s; the
/// tokenizer and template come from its directory, and the file is
/// validated against them first.
pub fn score_preference(base: &str, adapter: &Path, dataset: &Path) -> Result<PreferenceScore> {
    let store_root = loader::model_dir::resolve(None);
    let (weights, model_dir, _) = loader::model_dir::resolve_base(base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    let weights_str = utf8(&weights)?;
    checkpoint::weightio::WeightReader::open(weights_str).map_err(|e| Error::Backend(format!("{weights_str}: {e}")))?;
    checkpoint::st::read_card(utf8(adapter)?).map_err(|e| Error::Backend(format!("{}: {e}", adapter.display())))?;
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let pairs = encode_checked(dataset, &tok, &tmpl, None, data::chat::RenderOpts::default())?;
    Ok(score_pairs(weights_str, None, utf8(adapter)?, &pairs))
}

/// The policy `weights` + `policy` against the reference `weights` +
/// `reference` (or the base alone), on a scoring row that fits the longest
/// candidate.
fn score_pairs(weights: &str, reference: Option<&str>, policy: &str, pairs: &[(EncodedTurn, EncodedTurn)]) -> PreferenceScore {
    let block = block_for(longest(pairs), None).expect("no cap, so every length fits");
    let m = qwen3::eval::preference_margins(weights, reference, Some(policy), pairs, block);
    let n = m.margins.len();
    let scored = (n > 0).then_some(n as f64);
    PreferenceScore {
        accuracy: scored.map(|n| (m.margins.iter().filter(|&&x| x > 0.0).count() as f64 / n) as f32),
        mean_margin: scored.map(|n| (m.margins.iter().sum::<f64>() / n) as f32),
        pairs: n,
        skipped: m.skipped,
    }
}

fn encode_checked(path: &Path, tok: &QwenBpe, tmpl: &ChatTemplate, max_block: Option<u32>, render: data::chat::RenderOpts) -> Result<Vec<(EncodedTurn, EncodedTurn)>> {
    let (summary, pairs) = crate::preference_dataset::check_pairs(path, tok, tmpl, max_block, render).map_err(Error::Backend)?;
    if summary.pairs == 0 {
        return Err(Error::Backend(format!("{}: holds no preference pairs", path.display())));
    }
    Ok(pairs)
}

fn longest(pairs: &[(EncodedTurn, EncodedTurn)]) -> usize {
    pairs.iter().map(|(c, r)| c.ids.len().max(r.ids.len())).max().unwrap_or(0)
}

/// The trainable LoRA policy at `opts`' two rows and context: the same
/// start (`qwen3::finetune::lora_start`) and footprint-checked placement
/// `qwen3::finetune::finetune_lora_controlled` builds, with the adapter's
/// moments kept on the device so the run can be resumed exactly.
fn lora_model(weights: &str, rank: u32, alpha: f32, opts: &model::FitOpts, start: &qwen3::finetune::LoraStart<'_>) -> Result<qwen3::finetune::Trained> {
    let (cfg, init) = qwen3::finetune::lora_start(weights, rank, alpha, opts.seed, start).map_err(|e| Error::Backend(format!("{weights}: {e}")))?;
    let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
    std::env::remove_var("BRAIN_OFFLOAD_ADAM");
    let built = qwen3::finetune::build_trainer(cfg, opts, &init, gpu_core::select::Dtype::F32);
    if let Some(v) = prev_off {
        std::env::set_var("BRAIN_OFFLOAD_ADAM", v);
    }
    built.map_err(|e| Error::Backend(e.to_string()))
}
