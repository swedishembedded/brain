// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained models to
// the compute a deployment has, by removing what they need least, for its
// clients. If your team needs expertise in structured pruning and model
// compression then you can procure our services by sending an email to
// info@swedishembedded.com.

//! [`ModelPruner`]: a chat model with whole channels removed, ranked by what
//! removing each costs, written as a checkpoint any brain command serves.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::ModelPruner::from_pretrained("Qwen/Qwen3-0.6B")
//!     .dataset("domain.jsonl")
//!     .out_dir("runs/prune-1")
//!     .keep(0.75)
//!     .run()?;
//! println!("pruned checkpoint: {}", outcome.checkpoint.display());
//! # Ok(()) }
//! ```
//!
//! **What runs.** The dataset is split and prepared as [`crate::ModelOptimizer`]
//! does. On the architecture's trainable build every channel of the dimensions
//! the model declares prunable (qwen3: each layer's MLP hidden channels) is
//! ranked by the estimated loss increase of removing it on the calibration
//! records ([`modelopt::channel_importance`]); the most important share is kept,
//! the original and the plan are measured on the held-out records (the plan by
//! zeroing what it drops, which computes exactly what the narrower model
//! computes), and the narrower model is written in the family's own checkpoint
//! format with the base's tokenizer. The original and the pruned checkpoint
//! are then measured through the serving engine where the family has a scoring
//! path. With [`ModelPruner::recover`] a LoRA adapter is distilled over the
//! pruned checkpoint from the original. `prune.json` holds the outcome.

use std::path::PathBuf;

use gpu_core::select::Dtype;
use modelopt::{PruneReport, PruneWidth};
use serde::Serialize;

use crate::optimize::{distill_recovery, prepare, DeployedLoss, OptimizerProgress, Recovery, RecoveryRequest, WorkloadSpec};
use crate::{Device, Error, Result};

/// The narrower model before export, inside the out directory.
const PRUNED_FILE: &str = "pruned.brain.safetensors";
/// The exported checkpoint directory, inside the out directory.
const CHECKPOINT_DIR: &str = "checkpoint";
/// The report, inside the out directory.
const REPORT_FILE: &str = "prune.json";

/// Structured pruning of a chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ModelPruner {
    base: String,
    arch: Option<String>,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    held_out: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    width: PruneWidth,
    seed: Option<u64>,
    max_block: Option<u32>,
    device: Device,
    thinking: bool,
    recover_steps: u32,
    recover_lr: f32,
    recover_rank: u32,
}

/// What a pruning run decided and measured.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelPrunerOutcome {
    /// The ranking's decisions and the trainable build's measurements.
    pub report: PruneReport,
    /// The pruned checkpoint directory: weights, config and the base's tokenizer.
    pub checkpoint: PathBuf,
    /// The original and the pruned checkpoint, measured as deployed.
    pub deployed: Vec<DeployedLoss>,
    /// Where [`Self::deployed`] ran: the family's serving engine, or its trainer build.
    pub measured_on: &'static str,
    pub recovery: Option<Recovery>,
    pub calibration_records: usize,
    pub held_out_records: usize,
    pub seed: u64,
    pub report_path: PathBuf,
}

impl ModelPruner {
    /// A pruning of `base`, resolved as [`crate::ChatFullFineTune::from_pretrained`] resolves it.
    pub fn from_pretrained(base: impl Into<String>) -> ModelPruner {
        ModelPruner {
            base: base.into(),
            arch: None,
            models_dir: None,
            dataset: None,
            held_out: None,
            out_dir: None,
            width: PruneWidth::Fraction(0.75),
            seed: None,
            max_block: None,
            device: Device::default(),
            thinking: true,
            recover_steps: 0,
            recover_lr: 1e-4,
            recover_rank: 16,
        }
    }

    /// The architecture of the base (a `brain_arch` id); by default the one the
    /// checkpoint's `config.json` names.
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = Some(id.into());
        self
    }

    /// The models directory a `vendor/repo` base resolves in.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The chat dataset the channels are ranked and the result measured on;
    /// a quarter of its records is held out unless [`Self::held_out`] is given.
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// A second chat dataset to measure on; the whole of [`Self::dataset`] then ranks.
    pub fn held_out(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out = Some(path.into());
        self
    }

    /// Where the prepared dataset, the checkpoint and `prune.json` are written.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// Keep this share of every prunable dimension's channels (default 0.75).
    pub fn keep(mut self, fraction: f64) -> Self {
        self.width = PruneWidth::Fraction(fraction);
        self
    }

    /// Keep exactly this many channels of every prunable dimension.
    pub fn keep_channels(mut self, channels: usize) -> Self {
        self.width = PruneWidth::Channels(channels);
        self
    }

    /// The seed of the calibration/held-out split; drawn at random when unset and reported.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The longest row, in tokens; longer records are refused.
    pub fn max_block(mut self, tokens: u32) -> Self {
        self.max_block = Some(tokens);
        self
    }

    /// The devices the run uses.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Whether the chat template renders the reasoning block (default on).
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// After pruning, distill a LoRA adapter over the pruned checkpoint for
    /// `steps` steps, the original model as the teacher (default 0: none).
    pub fn recover(mut self, steps: u32) -> Self {
        self.recover_steps = steps;
        self
    }

    /// The recovery adapter's learning rate (default 1e-4).
    pub fn recover_lr(mut self, lr: f32) -> Self {
        self.recover_lr = lr;
        self
    }

    /// The recovery adapter's rank, its alpha equal to it (default 16).
    pub fn recover_rank(mut self, rank: u32) -> Self {
        self.recover_rank = rank;
        self
    }

    /// Prune, export, measure, write the report.
    pub fn run(&self) -> Result<ModelPrunerOutcome> {
        self.run_with(|_| {})
    }

    /// [`Self::run`], reporting each stage to `on_progress`.
    pub fn run_with(&self, mut on_progress: impl FnMut(&OptimizerProgress)) -> Result<ModelPrunerOutcome> {
        let work = prepare(&WorkloadSpec {
            who: "ModelPruner",
            base: &self.base,
            arch: self.arch.as_deref(),
            models_dir: self.models_dir.as_deref(),
            dataset: self.dataset.as_deref(),
            held_out: self.held_out.as_deref(),
            out_dir: self.out_dir.as_deref(),
            seed: self.seed,
            max_block: self.max_block,
            device: &self.device,
            thinking: self.thinking,
        })?;
        let chat = work.chat;
        let pruned_file = work.out_dir.join(PRUNED_FILE);
        let job = crate::chat_trainer::PruneJob { base: &work.open, dir: &work.prepared_dir, opts: &work.opts, width: self.width, out: &pruned_file };
        let report = (chat.prune)(job, &mut |p| on_progress(&OptimizerProgress::Search(p))).map_err(|e| Error::Backend(format!("pruning: {e}")))?;
        let checkpoint = work.out_dir.join(CHECKPOINT_DIR);
        (chat.export)(&pruned_file, &work.model_dir, &checkpoint, None).map_err(|e| Error::Backend(format!("exporting the pruned checkpoint: {e}")))?;
        std::fs::remove_file(&pruned_file).map_err(|e| Error::Backend(format!("{}: {e}", pruned_file.display())))?;
        // The pruned directory resolves like any base the caller names, so it opens the way they would open it.
        let checkpoint_ref = checkpoint.to_str().ok_or_else(|| Error::Backend(format!("{}: not a UTF-8 path", checkpoint.display())))?;
        let (weights, dir, _) = loader::model_dir::resolve_base(checkpoint_ref, None).map_err(Error::ModelNotFound)?;
        let pruned_open = (chat.checkpoint_path)(&weights, &dir);

        let f32 = model::ops::TierPolicy::uniform(Dtype::F32);
        let mut measured_on = crate::optimize::ON_ENGINE;
        let mut deployed = Vec::new();
        for (label, open) in [("original", &work.open), ("pruned", &pruned_open)] {
            on_progress(&OptimizerProgress::Deployed { label: label.to_string() });
            let served = (chat.served)(open, &f32, None, &work.prepared_dir, &work.opts);
            if served.is_none() {
                measured_on = crate::optimize::ON_TRAINER;
            }
            let loss = served.unwrap_or_else(|| (chat.measure)(open, &work.prepared_dir, &work.opts, &f32, None)).map_err(|e| Error::Backend(format!("measuring the {label} model: {e}")))?;
            let bytes = if label == "original" { report.params_before } else { report.params_after } as u64 * 4;
            deployed.push(DeployedLoss { label: label.to_string(), bytes, loss });
        }
        let recovery = if self.recover_steps > 0 {
            Some(distill_recovery(
                &work,
                &RecoveryRequest { student: &pruned_open, base_tiers: &f32, steps: self.recover_steps, lr: self.recover_lr, rank: self.recover_rank, regime: "pruning-recovery distillation" },
                &mut on_progress,
            )?)
        } else {
            None
        };
        let report_path = work.out_dir.join(REPORT_FILE);
        let outcome = ModelPrunerOutcome { report, checkpoint, deployed, measured_on, recovery, calibration_records: work.calibration_records, held_out_records: work.held_out_records, seed: work.seed, report_path: report_path.clone() };
        crate::optimize::write_json(&report_path, &outcome)?;
        Ok(outcome)
    }
}
