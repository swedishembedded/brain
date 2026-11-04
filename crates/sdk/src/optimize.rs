// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fitting open-weight language
// models into the memory a deployment has, by measurement, for its clients. If
// your team needs expertise in mixed-precision model compression then you can
// procure our services by sending an email to info@swedishembedded.com.

//! [`ModelOptimizer`]: the storage tier of every linear of a chat model,
//! searched within a byte budget and accepted only by measurement.
//!
//! ```no_run
//! # fn main() -> brain::Result<()> {
//! let outcome = brain::ModelOptimizer::from_pretrained("Qwen/Qwen3-0.6B")
//!     .dataset("held-out-domain.jsonl")
//!     .out_dir("runs/optimize-1")
//!     .run()?;
//! println!("--weight-format {}", outcome.report.policy);
//! assert!(outcome.accepted);
//! # Ok(()) }
//! ```
//!
//! **What runs.** The dataset's records are split into a calibration set and a
//! held-out set that share no record (or the held-out set is a second file).
//! The architecture's trainer builds the base trainable in fp32 and the search
//! ([`modelopt::search_weights`]) ranks every linear and candidate tier by its
//! estimated loss increase on the calibration set, solves for the tiers that
//! fit the budget, and measures the plan against the original and every
//! uniform tier that fits, with the weights simulated by the deployed codec.
//! Then the plan and those uniform tiers are each measured again with the base
//! built at them on the deployed kernels (the trainer's frozen-base build, the
//! kernels serving uses), which is what [`ModelOptimizerOutcome::accepted`]
//! is decided on. The report is written to `optimize.json` in the out
//! directory.
//!
//! The candidate tiers default to every frozen-base tier the architecture's
//! trainer holds, so every candidate can be measured on its deployed kernels;
//! the budget defaults to the bytes of uniform int8.

use std::path::{Path, PathBuf};

use gpu_core::select::Dtype;
use modelopt::{Budget, Report};
use serde::Serialize;

use crate::finetune::{block_for, fit_opts, resolve_seed, tokenizer_and_template, Asked};
use crate::{Device, Error, Result};

/// The prepared dataset, inside the out directory.
const PREPARED_DIR: &str = "prepared";
/// The report, inside the out directory.
const REPORT_FILE: &str = "optimize.json";
/// The share of the dataset's records held out when no held-out file is named.
const HELD_OUT_SHARE: f64 = 0.25;
/// Rows per batch of the search and the measurements.
const ROWS: u32 = 4;

/// The weight precision search of a chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ModelOptimizer {
    base: String,
    arch: String,
    models_dir: Option<String>,
    dataset: Option<PathBuf>,
    held_out: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    budget: BudgetSpec,
    tiers: Option<String>,
    seed: Option<u64>,
    max_block: Option<u32>,
    device: Device,
    thinking: bool,
}

/// The budget as the caller stated it, resolved against the model once its sites are known.
#[derive(Clone, Debug, PartialEq)]
enum BudgetSpec {
    Uniform(String),
    BitsPerWeight(f64),
    Bytes(u64),
}

impl BudgetSpec {
    fn resolve(&self) -> Result<Budget> {
        match self {
            BudgetSpec::Uniform(name) => {
                let tier = model::policy::parse_weight_format(name).ok().and_then(|p| p.uniform_dtype()).ok_or_else(|| Error::Backend(format!("ModelOptimizer: budget tier {name:?} is not a weight format")))?;
                Ok(Budget::Uniform(tier))
            }
            BudgetSpec::BitsPerWeight(bits) => Ok(Budget::BitsPerWeight(*bits)),
            BudgetSpec::Bytes(bytes) => Ok(Budget::Bytes(*bytes)),
        }
    }
}

/// A default candidate the device would hold at a wider tier.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExcludedTier {
    pub tier: &'static str,
    pub held_as: &'static str,
}

/// One measurement on the deployed kernels.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DeployedLoss {
    /// `"plan"`, `"original"` or a uniform tier's name.
    pub label: String,
    pub bytes: u64,
    pub loss: f64,
}

/// What a search decided and measured.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelOptimizerOutcome {
    /// The search: the plan, its `--weight-format` policy and the simulated measurements.
    pub report: Report,
    /// The plan, the original and each uniform tier that fits the budget, measured on the deployed kernels.
    pub deployed: Vec<DeployedLoss>,
    /// Whether the plan measured lower held-out loss on the deployed kernels than every uniform tier that fits the
    /// budget: the condition for deploying it instead of one of them.
    pub accepted: bool,
    /// Default candidates left out because the device would hold them at a wider tier, so neither their bytes nor
    /// their loss would be what the search assumed.
    pub excluded_tiers: Vec<ExcludedTier>,
    pub calibration_records: usize,
    pub held_out_records: usize,
    pub seed: u64,
    /// Where the report was written.
    pub report_path: PathBuf,
}

impl ModelOptimizer {
    /// A search over `base`: a checkpoint file, a checkpoint directory or a
    /// `vendor/repo` reference in the model store, as
    /// [`crate::ChatFullFineTune::from_pretrained`] takes.
    pub fn from_pretrained(base: impl Into<String>) -> ModelOptimizer {
        ModelOptimizer {
            base: base.into(),
            arch: "qwen3".to_string(),
            models_dir: None,
            dataset: None,
            held_out: None,
            out_dir: None,
            budget: BudgetSpec::Uniform("int8".to_string()),
            tiers: None,
            seed: None,
            max_block: None,
            device: Device::default(),
            thinking: true,
        }
    }

    /// The architecture of the base (a `brain_arch` id; default `qwen3`).
    pub fn arch(mut self, id: impl Into<String>) -> Self {
        self.arch = id.into();
        self
    }

    /// The models directory a `vendor/repo` base resolves in.
    pub fn models_dir(mut self, dir: impl Into<String>) -> Self {
        self.models_dir = Some(dir.into());
        self
    }

    /// The chat dataset (JSONL, the fine-tuners' format) representative of
    /// what the model will serve. Without [`Self::held_out`], a quarter of
    /// its records are held out to measure the result.
    pub fn dataset(mut self, path: impl Into<PathBuf>) -> Self {
        self.dataset = Some(path.into());
        self
    }

    /// A second chat dataset to measure the result on; the whole of
    /// [`Self::dataset`] then calibrates.
    pub fn held_out(mut self, path: impl Into<PathBuf>) -> Self {
        self.held_out = Some(path.into());
        self
    }

    /// Where the prepared dataset and `optimize.json` are written.
    pub fn out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    /// The budget: the bytes every searched linear would take at one weight
    /// format (`int8`, the default, or `bf16`, `nf4`, `q4`, ...).
    pub fn budget_like(mut self, format: impl Into<String>) -> Self {
        self.budget = BudgetSpec::Uniform(format.into());
        self
    }

    /// The budget: mean stored bits per searched weight, scales included
    /// (uniform group-scaled int8 is 9).
    pub fn budget_bits_per_weight(mut self, bits: f64) -> Self {
        self.budget = BudgetSpec::BitsPerWeight(bits);
        self
    }

    /// The budget: bytes over every searched linear, packed words and scales.
    pub fn budget_bytes(mut self, bytes: u64) -> Self {
        self.budget = BudgetSpec::Bytes(bytes);
        self
    }

    /// The candidate weight formats, comma separated (`"bf16,int8,nf4"`;
    /// default: every frozen-base format the architecture's trainer holds and
    /// the device holds as itself). A format the trainer or the device cannot
    /// hold is refused before anything is loaded: it could not be measured.
    pub fn tiers(mut self, formats: impl Into<String>) -> Self {
        self.tiers = Some(formats.into());
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

    /// The devices the search runs on.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Whether the chat template renders the reasoning block (default on).
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// Search, measure on the deployed kernels, write the report.
    pub fn run(&self) -> Result<ModelOptimizerOutcome> {
        let chat = crate::full_finetune::chat_entry(&self.arch)?;
        let dataset = self.dataset.as_deref().ok_or_else(|| Error::MissingArgument("ModelOptimizer: no dataset; call .dataset(path)".to_string()))?;
        let out_dir = self.out_dir.as_deref().ok_or_else(|| Error::MissingArgument("ModelOptimizer: no out directory; call .out_dir(path)".to_string()))?;
        let set = crate::device::resolve(&self.device)?;
        let (tiers, excluded_tiers) = candidates(self.tiers.as_deref(), chat.lora_bases, &gpu_core::Gpu::new(&[]).caps().numeric)?;
        let store_root = loader::model_dir::resolve(self.models_dir.as_deref());
        let (weights, model_dir, _) = loader::model_dir::resolve_base(&self.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
        let open = (chat.checkpoint_path)(&weights, &model_dir);

        let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: false, images: &data::chat::TextOnly::TRAINER };
        let seed = resolve_seed(self.seed, None)?;
        let (calibration, held_out) = match &self.held_out {
            Some(path) => (asked.read(dataset)?, asked.read(path)?),
            None => split(asked.read(dataset)?, seed)?,
        };
        let vocab = (chat.vocab)(&open).map_err(Error::Backend)?;
        let prepared_dir = out_dir.join(PREPARED_DIR);
        let prepared = data::chat::prepare_chat_samples(&calibration, &held_out, &tok, &tmpl, render, vocab, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
        let block = block_for(prepared.longest_example, self.max_block)?;
        crate::full_finetune::plan_before_load(&open, block, &set)?;
        let opts = model::FitOpts { batch_size: ROWS, ..fit_opts(1, block, 0.0, seed) };

        let report = (chat.search)(&open, &prepared_dir, &opts, &tiers, self.budget.resolve()?).map_err(|e| Error::Backend(format!("searching: {e}")))?;
        let measure = |label: String, bytes: u64, policy: &model::ops::TierPolicy| -> Result<DeployedLoss> {
            let loss = (chat.measure)(&open, &prepared_dir, &opts, policy).map_err(|e| Error::Backend(format!("measuring {label}: {e}")))?;
            Ok(DeployedLoss { label, bytes, loss })
        };
        let plan_policy = model::ops::TierPolicy::parse(&report.policy).map_err(Error::Backend)?;
        let mut deployed = vec![measure("plan".into(), report.plan_bytes, &plan_policy)?, measure("original".into(), report.f32_bytes, &model::ops::TierPolicy::uniform(Dtype::F32))?];
        for b in &report.baselines {
            let tier = Dtype::from_name(b.tier).ok_or_else(|| Error::Backend(format!("unknown tier {}", b.tier)))?;
            deployed.push(measure(b.tier.to_string(), b.bytes, &model::ops::TierPolicy::uniform(tier))?);
        }
        let plan_loss = deployed[0].loss;
        let accepted = deployed.iter().filter(|d| report.baselines.iter().any(|b| b.tier == d.label)).all(|d| plan_loss < d.loss);

        let report_path = out_dir.join(REPORT_FILE);
        let outcome = ModelOptimizerOutcome { report, deployed, accepted, excluded_tiers, calibration_records: calibration.len(), held_out_records: held_out.len(), seed, report_path: report_path.clone() };
        write_report(&report_path, &outcome)?;
        Ok(outcome)
    }
}

/// The candidate tiers: the caller's, each of which the trainer must hold (so it can be measured on its deployed
/// kernels) and the device must hold as itself, else an error naming it; or by default every tier the trainer holds,
/// less those the device would widen, which are returned with the tier they would become.
fn candidates(asked: Option<&str>, trainer: model::policy::TrainBases, numeric: &gpu_core::NumericSupport) -> Result<(Vec<Dtype>, Vec<ExcludedTier>)> {
    match asked {
        Some(list) => {
            let mut tiers = Vec::new();
            for spec in list.split(',').map(str::trim) {
                let tier = trainer.parse(spec).map_err(|why| Error::Backend(format!("ModelOptimizer: candidate {spec:?} cannot be measured on its deployed kernels: {why}")))?;
                let held = tier.promote(numeric);
                if held != tier {
                    return Err(Error::Backend(format!("ModelOptimizer: candidate {}: this device holds it as {}", tier.name(), held.name())));
                }
                tiers.push(tier);
            }
            Ok((tiers, Vec::new()))
        }
        None => {
            let (kept, widened): (Vec<Dtype>, Vec<Dtype>) = trainer.bases.iter().partition(|&&t| t.promote(numeric) == t);
            Ok((kept, widened.into_iter().map(|t| ExcludedTier { tier: t.name(), held_as: t.promote(numeric).name() }).collect()))
        }
    }
}

/// The records split into calibration and held-out sets that share none: a
/// seeded shuffle, then [`HELD_OUT_SHARE`] of it (at least one record) held out.
fn split(mut records: Vec<data::chat::ChatSample>, seed: u64) -> Result<(Vec<data::chat::ChatSample>, Vec<data::chat::ChatSample>)> {
    if records.len() < 2 {
        return Err(Error::Backend(format!("ModelOptimizer: {} record(s) cannot be split into a calibration and a held-out set; name a held-out file", records.len())));
    }
    let mut rng = data::rng::Rng::new(seed);
    for i in (1..records.len()).rev() {
        records.swap(i, rng.gen_range_inclusive(0, i as i64) as usize);
    }
    let held = ((records.len() as f64 * HELD_OUT_SHARE).round() as usize).clamp(1, records.len() - 1);
    let held_out = records.split_off(records.len() - held);
    Ok((records, held_out))
}

fn write_report(path: &Path, outcome: &ModelOptimizerOutcome) -> Result<()> {
    let json = serde_json::to_string_pretty(outcome).map_err(|e| Error::Backend(format!("serializing the report: {e}")))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).and_then(|()| std::fs::rename(&tmp, path)).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}
