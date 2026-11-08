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
//! Then the plan and those uniform tiers are each measured again as deployed:
//! through the family's serving engine where it has a scoring path (its
//! kernels, int8 activation quantization included), else on the trainer's
//! frozen-base build ([`ModelOptimizerOutcome::measured_on`] says which). That
//! measurement alone decides [`ModelOptimizerOutcome::accepted`]. The report is
//! written to `optimize.json` in the out directory.
//!
//! **Recovery.** With [`ModelOptimizer::recover`], a LoRA adapter is then
//! distilled over the base held at the plan: the original model's top-K
//! distributions on the calibration records are the teacher, the quantized
//! base plus the adapter the student. The adapter is measured on the trainer
//! build that trained it, and served beside the policy as an overlay.
//!
//! The candidate tiers default to every tier the architecture's trainer holds a
//! frozen base in (so each can be measured) that its serving engine holds per
//! linear (so an accepted plan can be served) and the device holds as itself;
//! the budget defaults to halfway between the two smallest candidates' uniform
//! sizes, so the plan has bytes to place and a uniform deployment to beat.

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
/// The recovery adapter, inside the out directory.
const RECOVERY_FILE: &str = "recovery.safetensors";
/// Teacher distribution entries kept per position for recovery.
const TEACHER_TOP_K: usize = 16;
/// Where the measurements that decide acceptance ran.
pub(crate) const ON_ENGINE: &str = "serving engine";
pub(crate) const ON_TRAINER: &str = "trainer frozen-base build";

/// The weight precision search of a chat model. See this module's doc.
#[derive(Clone, Debug)]
pub struct ModelOptimizer {
    base: String,
    arch: Option<String>,
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
    recover_steps: u32,
    recover_lr: f32,
    recover_rank: u32,
    latency_rows: u32,
    latency_limit: Option<LatencySpec>,
    frontier_points: usize,
}

/// The latency limit as the caller stated it.
#[derive(Clone, Debug, PartialEq)]
enum LatencySpec {
    Like(String),
    Seconds(f64),
}

/// The budget as the caller stated it, resolved against the model once its sites are known.
#[derive(Clone, Debug, PartialEq)]
enum BudgetSpec {
    Midway,
    Uniform(String),
    BitsPerWeight(f64),
    Bytes(u64),
}

impl BudgetSpec {
    fn resolve(&self) -> Result<Budget> {
        match self {
            BudgetSpec::Midway => Ok(Budget::Midway),
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

/// Where a [`ModelOptimizer`] run is.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizerProgress {
    /// The search on the trainable build: calibration, estimates, simulated measurements.
    Search(modelopt::Progress),
    /// A plan (`"plan"`, `"original"`, a uniform format) is about to be built and measured as deployed.
    Deployed { label: String },
    /// The recovery teacher's distributions are being collected.
    Teacher,
    /// A recovery distillation step finished.
    Recovering { step: u32, steps: u32, loss: f32 },
}

/// A LoRA adapter distilled over the base held at the plan.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Recovery {
    pub adapter: PathBuf,
    pub steps: u32,
    pub rank: u32,
    /// The trainer build's held-out loss at the plan, without and with the adapter: the same build, so the two
    /// compare directly.
    pub plan_loss: f64,
    pub recovered_loss: f64,
}

impl Recovery {
    /// Whether the adapter lowered the held-out loss of the plan.
    pub fn improves(&self) -> bool {
        self.recovered_loss < self.plan_loss
    }
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
    /// Where the measurements in [`Self::deployed`] ran: the family's serving engine, or its trainer's frozen-base
    /// build when it has no engine scoring path.
    pub measured_on: &'static str,
    /// The recovery adapter, when [`ModelOptimizer::recover`] asked for one.
    pub recovery: Option<Recovery>,
    /// The frontier's plans ([`Report::frontier`], same order) measured as deployed.
    pub frontier: Vec<DeployedLoss>,
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
            arch: None,
            models_dir: None,
            dataset: None,
            held_out: None,
            out_dir: None,
            budget: BudgetSpec::Midway,
            tiers: None,
            seed: None,
            max_block: None,
            device: Device::default(),
            thinking: true,
            recover_steps: 0,
            recover_lr: 1e-4,
            recover_rank: 16,
            latency_rows: 1,
            latency_limit: None,
            frontier_points: 0,
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
    /// format (`int8`, `bf16`, `nf4`, `q4`, ...). The default is halfway
    /// between the two smallest candidates' uniform sizes.
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

    /// The candidate weight formats, comma separated (`"bf16,int8"`; default:
    /// every format the architecture's trainer holds a frozen base in, its
    /// engine serves per linear, and the device holds as itself). A format
    /// failing any of the three is refused before anything is loaded.
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

    /// The rows each site's product is timed at on the device (default 1,
    /// single-stream decode). Every plan's seconds are reported.
    pub fn latency_rows(mut self, rows: u32) -> Self {
        self.latency_rows = rows.max(1);
        self
    }

    /// Limit the plan's summed product seconds to those of every linear at
    /// one format (`int8`, `bf16`, ...): no slower than that uniform deployment.
    pub fn max_latency_like(mut self, format: impl Into<String>) -> Self {
        self.latency_limit = Some(LatencySpec::Like(format.into()));
        self
    }

    /// Limit the plan's summed product seconds.
    pub fn max_latency_seconds(mut self, seconds: f64) -> Self {
        self.latency_limit = Some(LatencySpec::Seconds(seconds));
        self
    }

    /// Also solve `points` plans at budgets spread across the candidates'
    /// range, each measured simulated and as deployed (default 0: none): the
    /// trade-off between bytes and loss, in [`ModelOptimizerOutcome::frontier`].
    pub fn frontier(mut self, points: usize) -> Self {
        self.frontier_points = points;
        self
    }

    /// After the search, distill a LoRA adapter over the base held at the plan
    /// for `steps` steps, the original model as the teacher (default 0: none).
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

    /// Whether the chat template renders the reasoning block (default on).
    pub fn thinking(mut self, on: bool) -> Self {
        self.thinking = on;
        self
    }

    /// Search, measure on the deployed kernels, write the report.
    pub fn run(&self) -> Result<ModelOptimizerOutcome> {
        self.run_with(|_| {})
    }

    /// [`Self::run`], reporting each stage to `on_progress`.
    pub fn run_with(&self, mut on_progress: impl FnMut(&OptimizerProgress)) -> Result<ModelOptimizerOutcome> {
        let work = prepare(&WorkloadSpec {
            who: "ModelOptimizer",
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
        let (tiers, excluded_tiers) = candidates(self.tiers.as_deref(), work.chat.lora_bases.model, work.chat.served_weights, &gpu_core::Gpu::new(&[]).caps().numeric)?;
        let Workload { chat, open, prepared_dir, opts, seed, calibration_records, held_out_records, .. } = &work;
        let (chat, open, prepared_dir, opts, seed) = (*chat, open.clone(), prepared_dir.clone(), opts.clone(), *seed);

        let limit = match &self.latency_limit {
            None => None,
            Some(LatencySpec::Seconds(s)) if s.is_finite() && *s > 0.0 => Some(modelopt::LatencyLimit::Seconds(*s)),
            Some(LatencySpec::Seconds(s)) => return Err(Error::Backend(format!("ModelOptimizer: a latency limit must be finite and positive, got {s}"))),
            Some(LatencySpec::Like(name)) => {
                let tier = model::policy::parse_weight_format(name).ok().and_then(|p| p.uniform_dtype()).ok_or_else(|| Error::Backend(format!("ModelOptimizer: latency tier {name:?} is not a weight format")))?;
                Some(modelopt::LatencyLimit::Uniform(tier))
            }
        };
        let timer = DeviceTimer::new();
        let time = |rows: u32, n: usize, k: usize, tier: Dtype| timer.time(rows, n, k, tier);
        let timing = modelopt::Timing { rows: self.latency_rows, limit, time: &time };
        let job = crate::chat_trainer::SearchJob { base: &open, dir: &prepared_dir, opts: &opts, tiers: &tiers, budget: self.budget.resolve()?, timing: Some(timing), frontier_points: self.frontier_points };
        let report = (chat.search)(job, &mut |p| on_progress(&OptimizerProgress::Search(p))).map_err(|e| Error::Backend(format!("searching: {e}")))?;
        let plan_policy = model::ops::TierPolicy::parse(&report.policy).map_err(Error::Backend)?;
        let mut measured_on = ON_ENGINE;
        let mut deployed = Vec::new();
        let mut plans: Vec<(String, u64, model::ops::TierPolicy)> = vec![("plan".into(), report.plan_bytes, plan_policy.clone()), ("original".into(), report.f32_bytes, model::ops::TierPolicy::uniform(Dtype::F32))];
        // A baseline holds the searched linears at its tier and nothing else (not the embedding or a tied head),
        // the same tensors the plan places, so the two compare.
        for b in &report.baselines {
            plans.push((b.tier.to_string(), b.bytes, model::ops::TierPolicy::parse(&b.policy).map_err(Error::Backend)?));
        }
        let first_frontier = plans.len();
        for (i, p) in report.frontier.iter().enumerate() {
            plans.push((format!("frontier {}", i + 1), p.plan_bytes, model::ops::TierPolicy::parse(&p.policy).map_err(Error::Backend)?));
        }
        for (label, bytes, policy) in plans {
            on_progress(&OptimizerProgress::Deployed { label: label.clone() });
            let served = (chat.served)(&open, &policy, None, &prepared_dir, &opts);
            if served.is_none() {
                measured_on = ON_TRAINER;
            }
            let loss = served.unwrap_or_else(|| (chat.measure)(&open, &prepared_dir, &opts, &policy, None)).map_err(|e| Error::Backend(format!("measuring {label}: {e}")))?;
            deployed.push(DeployedLoss { label, bytes, loss });
        }
        let frontier = deployed.split_off(first_frontier);
        let plan_loss = deployed[0].loss;
        let accepted = deployed.iter().filter(|d| report.baselines.iter().any(|b| b.tier == d.label)).all(|d| plan_loss < d.loss);
        let recovery = if self.recover_steps > 0 { Some(distill_recovery(
                &work,
                &RecoveryRequest { student: &work.open, base_tiers: &plan_policy, steps: self.recover_steps, lr: self.recover_lr, rank: self.recover_rank, regime: "quantization-recovery distillation" },
                &mut on_progress,
            )?) } else { None };

        let report_path = work.out_dir.join(REPORT_FILE);
        let outcome = ModelOptimizerOutcome { report, deployed, accepted, measured_on, recovery, frontier, excluded_tiers, calibration_records: *calibration_records, held_out_records: *held_out_records, seed, report_path: report_path.clone() };
        write_json(&report_path, &outcome)?;
        Ok(outcome)
    }
}

/// A recovery distillation: a LoRA adapter over `student` held at
/// `base_tiers`, trained toward the workload's base (the teacher) on the
/// calibration records.
pub(crate) struct RecoveryRequest<'a> {
    pub student: &'a Path,
    pub base_tiers: &'a model::ops::TierPolicy,
    pub steps: u32,
    pub lr: f32,
    pub rank: u32,
    /// What the adapter's provenance calls the run.
    pub regime: &'static str,
}

/// Distill the recovery adapter, then measure the student without and with it
/// on the trainer build that trained it.
pub(crate) fn distill_recovery(work: &Workload, req: &RecoveryRequest<'_>, on_progress: &mut dyn FnMut(&OptimizerProgress)) -> Result<Recovery> {
    let (chat, prepared_dir, opts) = (&work.chat, work.prepared_dir.as_path(), &work.opts);
    on_progress(&OptimizerProgress::Teacher);
    // The teacher reads whole-row logits (`Model::logits_all`), which a model built for one row gives.
    let one_row = model::FitOpts { batch_size: 1, ..opts.clone() };
    let examples = (chat.teacher)(&work.open, prepared_dir, &one_row, TEACHER_TOP_K).map_err(|e| Error::Backend(format!("the teacher: {e}")))?;
    let student = model::FitOpts { batch_size: 1, ..fit_opts(req.steps, opts.block_size, req.lr, opts.seed) };
    let adapter = work.out_dir.join(RECOVERY_FILE);
    let base_id = work.base_id.as_str();
    let card_id = format!("{base_id}:local:recovery:latest");
    let provenance = checkpoint::st::TrainingProvenance {
        code_revision: format!("brain {}", env!("CARGO_PKG_VERSION")),
        regime: req.regime.to_string(),
        seed: student.seed,
        hyperparams: serde_json::json!({ "rank": req.rank, "steps": req.steps, "lr": req.lr, "top_k": TEACHER_TOP_K, "weight_format": req.base_tiers.describe(), "examples": examples.len() }),
        environment: gpu_core::backend_name().to_string(),
        gate: None,
        trained_from: None,
        base_digest: Some(crate::finetune::base_digest(req.student)?),
        cycle: 0,
    };
    let mut on_step = |s: &model::StepReport| {
        on_progress(&OptimizerProgress::Recovering { step: s.step, steps: s.steps, loss: s.loss });
        true
    };
    let job = crate::chat_trainer::DistillJob {
        base: req.student,
        lora: model::finetune::LoraRequest { rank: req.rank, alpha: req.rank as f32, continue_from: None, base: req.base_tiers, modules_to_save: &[] },
        opts: &student,
        examples,
        adapter: &adapter,
        lineage: model::finetune::AdapterLineage { card_id: &card_id, base_id, dataset_id: None, provenance: Some(provenance) },
    };
    (chat.distill)(job, model::FitControl { on_step: Some(&mut on_step), ..Default::default() }).map_err(|e| Error::Backend(format!("recovery: {e}")))?;
    on_progress(&OptimizerProgress::Deployed { label: "the student with the recovery adapter".into() });
    let plan_loss = (chat.measure)(req.student, prepared_dir, opts, req.base_tiers, None).map_err(|e| Error::Backend(format!("measuring the student: {e}")))?;
    let recovered_loss = (chat.measure)(req.student, prepared_dir, opts, req.base_tiers, Some((&adapter, req.rank, req.rank as f32))).map_err(|e| Error::Backend(format!("measuring the recovery adapter: {e}")))?;
    Ok(Recovery { adapter, steps: req.steps, rank: req.rank, plan_loss, recovered_loss })
}

/// The candidate tiers. Each must be one the family serves per linear (so an accepted plan can be served, and is
/// measured on that serving engine) and one the device holds as itself (so its bytes and loss are what the search
/// assumed). The caller's list is refused by the first that is not; the default is every served tier the device holds,
/// with those it would widen returned with what it would hold.
fn candidates(asked: Option<&str>, family: &str, served: &[Dtype], numeric: &gpu_core::NumericSupport) -> Result<(Vec<Dtype>, Vec<ExcludedTier>)> {
    let names = |tiers: &[Dtype]| tiers.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ");
    if served.is_empty() {
        return Err(Error::Backend(format!("ModelOptimizer: {family} serves no per-linear weight format, so no searched plan could be served")));
    }
    match asked {
        Some(list) => {
            let mut tiers = Vec::new();
            for spec in list.split(',').map(str::trim) {
                let tier = model::policy::parse_weight_format(spec).ok().and_then(|p| p.uniform_dtype()).ok_or_else(|| Error::Backend(format!("ModelOptimizer: candidate {spec:?} is not a weight format")))?;
                if !served.contains(&tier) {
                    return Err(Error::Backend(format!("ModelOptimizer: candidate {}: {family} does not serve it per linear; it serves {}", tier.name(), names(served))));
                }
                let held = tier.promote(numeric);
                if held != tier {
                    return Err(Error::Backend(format!("ModelOptimizer: candidate {}: this device holds it as {}", tier.name(), held.name())));
                }
                tiers.push(tier);
            }
            Ok((tiers, Vec::new()))
        }
        None => {
            let (kept, widened): (Vec<Dtype>, Vec<Dtype>) = served.iter().partition(|&&t| t.promote(numeric) == t);
            Ok((kept, widened.into_iter().map(|t| ExcludedTier { tier: t.name(), held_as: t.promote(numeric).name() }).collect()))
        }
    }
}

/// What a compression run is asked to read, before anything is resolved.
pub(crate) struct WorkloadSpec<'a> {
    /// The public type's name, for errors.
    pub who: &'static str,
    pub base: &'a str,
    /// `None`: read from the checkpoint.
    pub arch: Option<&'a str>,
    pub models_dir: Option<&'a str>,
    pub dataset: Option<&'a Path>,
    pub held_out: Option<&'a Path>,
    pub out_dir: Option<&'a Path>,
    pub seed: Option<u64>,
    pub max_block: Option<u32>,
    pub device: &'a Device,
    pub thinking: bool,
}

/// A base and a chat dataset prepared for it: the calibration records as the
/// prepared train split, the held-out records as its validation split.
pub(crate) struct Workload {
    pub chat: crate::chat_trainer::ChatEntry,
    /// The checkpoint the family's trainer opens, and the directory holding its tokenizer.
    pub open: PathBuf,
    pub model_dir: PathBuf,
    pub base_id: String,
    pub out_dir: PathBuf,
    pub prepared_dir: PathBuf,
    /// Rows of [`ROWS`] records, the row the longest record.
    pub opts: model::FitOpts,
    pub seed: u64,
    pub calibration_records: usize,
    pub held_out_records: usize,
}

/// The registered architecture a brain-format checkpoint names in its own
/// config (`model`), through the registry's family aliases.
fn brain_checkpoint_arch(weights: &Path) -> Option<&'static str> {
    let reader = checkpoint::weightio::WeightReader::open(weights.to_str()?).ok()?;
    let config = reader.config();
    brain_arch::by_family(config["model"].as_str()?).map(|a| a.id)
}

/// Resolve the base, read and split the dataset ([`split`], or the held-out
/// file), and prepare both splits for the base's tokenizer and template;
/// refuses a run whose base does not fit the devices before anything loads.
pub(crate) fn prepare(spec: &WorkloadSpec<'_>) -> Result<Workload> {
    let dataset = spec.dataset.ok_or_else(|| Error::MissingArgument(format!("{}: no dataset; call .dataset(path)", spec.who)))?;
    let out_dir = spec.out_dir.ok_or_else(|| Error::MissingArgument(format!("{}: no out directory; call .out_dir(path)", spec.who)))?;
    let set = crate::device::resolve(spec.device)?;
    let store_root = loader::model_dir::resolve(spec.models_dir);
    let (weights, model_dir, base_id) = loader::model_dir::resolve_base(spec.base, store_root.as_deref()).map_err(Error::ModelNotFound)?;
    // The architecture is the checkpoint's own (its config.json, through the registry) unless the caller names one.
    let arch = match spec.arch {
        Some(id) => id.to_string(),
        None => crate::study::arch_of_checkpoint(&model_dir)
            .map(|entry| entry.id)
            .or_else(|| brain_checkpoint_arch(&weights))
            .map(str::to_string)
            .ok_or_else(|| Error::UnsupportedArchitecture(format!("{}: cannot tell the architecture of {} from a config.json beside it or its own config; name it with .arch(id)", spec.who, spec.base)))?,
    };
    let chat = crate::full_finetune::chat_entry(&arch)?;
    let open = (chat.checkpoint_path)(&weights, &model_dir);
    let (tok, tmpl) = tokenizer_and_template(&model_dir)?;
    let no_think = if spec.thinking { None } else { tmpl.no_think_block() };
    let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
    let asked = Asked { tok: &tok, tmpl: &tmpl, no_think: no_think.as_deref(), render, pack: false, images: &data::chat::TextOnly::TRAINER };
    let seed = resolve_seed(spec.seed, None)?;
    let (calibration, held_out) = match spec.held_out {
        Some(path) => (asked.read(dataset)?, asked.read(path)?),
        None => split(asked.read(dataset)?, seed)?,
    };
    let vocab = (chat.vocab)(&open).map_err(Error::Backend)?;
    let prepared_dir = out_dir.join(PREPARED_DIR);
    let prepared = data::chat::prepare_chat_samples(&calibration, &held_out, &tok, &tmpl, render, vocab, &prepared_dir).map_err(|e| Error::Backend(format!("preparing the dataset: {e}")))?;
    let block = block_for(prepared.longest_example, spec.max_block)?;
    crate::full_finetune::plan_before_load(&open, block, &set)?;
    let opts = model::FitOpts { batch_size: ROWS, ..fit_opts(1, block, 0.0, seed) };
    Ok(Workload { chat, open, model_dir, base_id, out_dir: out_dir.to_path_buf(), prepared_dir, opts, seed, calibration_records: calibration.len(), held_out_records: held_out.len() })
}

/// Times one product on the device the search runs on, through the ops façade
/// a model's linears go through ([`model::probe::gemm`]: the selector's
/// kernel for the tier, the fastest of repeated runs), the device first
/// brought to its operating point once.
struct DeviceTimer {
    ops: std::cell::OnceCell<model::ops::Ops>,
}

impl DeviceTimer {
    fn new() -> DeviceTimer {
        DeviceTimer { ops: std::cell::OnceCell::new() }
    }

    fn time(&self, rows: u32, n: usize, k: usize, tier: Dtype) -> std::result::Result<f64, String> {
        let shape = (rows, u32::try_from(n).map_err(|_| "n too large")?, u32::try_from(k).map_err(|_| "k too large")?);
        let plan = model::probe::Plan { shapes: vec![shape], budget: std::time::Duration::from_millis(250), reps: 16, warmup: std::time::Duration::ZERO };
        let ops = match self.ops.get() {
            Some(ops) => ops,
            None => {
                let ops = model::ops::Ops::new(gpu_core::Gpu::new(model::ops::training_kernel_list()))?;
                // Only the operating point matters here, not how the clock ramped to it.
                let _ramp = model::probe::warm_up(&ops, &model::probe::Plan { warmup: std::time::Duration::from_secs(2), ..plan.clone() });
                self.ops.get_or_init(|| ops)
            }
        };
        model::probe::gemm(ops, tier, &plan).map(|t| t.best_seconds).map_err(|e| e.to_string())
    }
}

/// The records split into calibration and held-out sets that share none: a
/// seeded shuffle, then [`HELD_OUT_SHARE`] of it (at least one record) held out.
fn split(mut records: Vec<data::chat::ChatSample>, seed: u64) -> Result<(Vec<data::chat::ChatSample>, Vec<data::chat::ChatSample>)> {
    if records.len() < 2 {
        return Err(Error::Backend(format!("{} record(s) cannot be split into a calibration and a held-out set; name a held-out file", records.len())));
    }
    let mut rng = data::rng::Rng::new(seed);
    for i in (1..records.len()).rev() {
        records.swap(i, rng.gen_range_inclusive(0, i as i64) as usize);
    }
    let held = ((records.len() as f64 * HELD_OUT_SHARE).round() as usize).clamp(1, records.len() - 1);
    let held_out = records.split_off(records.len() - held);
    Ok((records, held_out))
}

/// `value` as pretty JSON at `path`, through a temporary file and a rename.
pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let json = serde_json::to_string_pretty(value).map_err(|e| Error::Backend(format!("serializing the report: {e}")))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).and_then(|()| std::fs::rename(&tmp, path)).map_err(|e| Error::Backend(format!("{}: {e}", path.display())))
}
