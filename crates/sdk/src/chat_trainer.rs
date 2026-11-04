// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning open-weight language
// models of any architecture through one training stack, for its clients. If
// your team needs expertise in wiring a new model family into full-parameter
// and preference training then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The monomorphised entry points of the generic chat fine-tuners.
//!
//! [`ChatFullFineTune`](crate::ChatFullFineTune) and
//! [`PreferenceFineTune`](crate::PreferenceFineTune) are written once; what
//! differs per architecture is `model::finetune::ChatTrainer`. A table row
//! cannot be generic, so [`ChatEntry::of`] instantiates the generic functions
//! below for one trainer and keeps them as function pointers, the way
//! [`crate::study::ArchEntry`] already does for studies. Nothing here names an
//! architecture.

use std::path::{Path, PathBuf};

use data::preference::EncodedTurn;
use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use modelopt::{Budget, MeasureFit, Progress, PruneFit, PruneReport, PruneWidth, Report, SearchFit, Timing};
use rl::fit::{DistillFit, TeacherFit};
use rl::objective::distill::{DistillConfig, DistillExample};
use model::finetune::{AdapterLineage, ChatTrainer, LoraRequest, OptimizerState, PreferenceMargins};
use model::{FitControl, FitOpts, FitReport};
use rl::fit::DpoFit;
use rl::objective::dpo::{DpoConfig, PackedPair};

/// One preference fine-tune of a LoRA adapter, as the generic trainer takes it.
pub(crate) struct PreferenceJob<'a> {
    pub base: &'a Path,
    pub lora: LoraRequest<'a>,
    pub opts: &'a FitOpts,
    pub dpo: DpoConfig,
    pub pairs: Vec<PackedPair>,
    /// Where the trained adapter is written (unless the run is interrupted).
    pub adapter: &'a Path,
    pub lineage: AdapterLineage<'a>,
    /// Pair sets to score on the resident model, before the first step (their
    /// reference is the starting policy) and after the last; empty when the
    /// family scores the exported adapter instead.
    pub score_sets: Vec<Vec<PackedPair>>,
}

/// What a preference run measured on the resident model: one margin list per
/// [`PreferenceJob::score_sets`] entry.
pub(crate) type ResidentMargins = Vec<Vec<f64>>;

type FullFn = for<'c> fn(&Path, &Path, &FitOpts, &Path, OptimizerState, FitControl<'c>) -> std::io::Result<FitReport>;
type PreferenceFn = for<'j, 'c> fn(PreferenceJob<'j>, FitControl<'c>) -> std::io::Result<(FitReport, ResidentMargins)>;
type MarginsFn = fn(&Path, Option<&Path>, &Path, &[(EncodedTurn, EncodedTurn)], u32) -> Option<PreferenceMargins>;
type SearchFn = for<'j> fn(SearchJob<'j>, &mut dyn FnMut(Progress)) -> std::io::Result<Report>;

/// One weight precision search of a base, on a prepared dataset.
pub(crate) struct SearchJob<'a> {
    pub base: &'a Path,
    /// The prepared dataset: its train split calibrates, its validation split measures.
    pub dir: &'a Path,
    pub opts: &'a FitOpts,
    pub tiers: &'a [Dtype],
    pub budget: Budget,
    pub timing: Option<Timing<'a>>,
    pub frontier_points: usize,
}
/// An adapter to apply: its file, rank and alpha.
pub(crate) type AdapterArg<'a> = Option<(&'a Path, u32, f32)>;
type MeasureFn = fn(&Path, &Path, &FitOpts, &TierPolicy, AdapterArg<'_>) -> std::io::Result<f64>;
type ServedFn = fn(&Path, &TierPolicy, Option<&Path>, &Path, &FitOpts) -> Option<std::io::Result<f64>>;
type PruneFn = for<'j> fn(PruneJob<'j>, &mut dyn FnMut(Progress)) -> std::io::Result<PruneReport>;

/// One structured pruning of a base, on a prepared dataset, the narrower
/// model written to `out`.
pub(crate) struct PruneJob<'a> {
    pub base: &'a Path,
    pub dir: &'a Path,
    pub opts: &'a FitOpts,
    pub width: PruneWidth,
    pub out: &'a Path,
}

type TeacherFn = fn(&Path, &Path, &FitOpts, usize) -> std::io::Result<Vec<DistillExample>>;
type DistillFn = for<'j, 'c> fn(DistillJob<'j>, FitControl<'c>) -> std::io::Result<FitReport>;

/// One distillation of a LoRA adapter over a frozen base held at `lora.base`.
pub(crate) struct DistillJob<'a> {
    pub base: &'a Path,
    pub lora: LoraRequest<'a>,
    /// Rows of one example each, `block_size` the longest example.
    pub opts: &'a FitOpts,
    pub examples: Vec<DistillExample>,
    /// Where the trained adapter is written (unless the run is interrupted).
    pub adapter: &'a Path,
    pub lineage: AdapterLineage<'a>,
}

/// What the SDK's full-parameter and preference fine-tuners need of one
/// architecture, resolved once per table row.
#[derive(Clone, Copy)]
pub(crate) struct ChatEntry {
    pub checkpoint_path: fn(&Path, &Path) -> PathBuf,
    pub vocab: fn(&Path) -> Result<usize, String>,
    pub full: FullFn,
    pub export: fn(&Path, &Path, &Path, Option<&Path>) -> Result<(), String>,
    pub preference: PreferenceFn,
    pub margins: MarginsFn,
    /// The frozen-base formats the family's LoRA trainer holds.
    pub lora_bases: model::policy::TrainBases,
    /// The family scores the exported adapter ([`Self::margins`]) rather than the resident model.
    pub scores_exported_pairs: bool,
    /// The weight formats the family's serving engine holds per linear.
    pub served_weights: &'static [Dtype],
    /// The weight precision search over the trainable build of a base, on a prepared dataset.
    pub search: SearchFn,
    /// The held-out loss of a base built with its linears at a tier policy, on the deployed kernels, with an
    /// adapter applied when one is given.
    pub measure: MeasureFn,
    /// The held-out loss of a base at a tier policy through the family's serving engine, where it has one.
    pub served: ServedFn,
    /// Structured pruning of the base's prunable dimensions, the narrower model saved.
    pub prune: PruneFn,
    /// The top-K distributions the base (fp32, trainable build) gives a prepared dataset's train split.
    pub teacher: TeacherFn,
    /// A LoRA adapter distilled from teacher examples over a frozen base at a tier policy.
    pub distill: DistillFn,
}

impl ChatEntry {
    pub(crate) const fn of<T: ChatTrainer>() -> ChatEntry {
        ChatEntry {
            checkpoint_path: T::checkpoint_path,
            vocab: T::vocab,
            full: full::<T>,
            export: T::export_checkpoint,
            preference: preference::<T>,
            margins: T::preference_margins,
            lora_bases: T::LORA_BASES,
            scores_exported_pairs: T::SCORES_EXPORTED_PAIRS,
            served_weights: T::SERVED_WEIGHTS,
            search: search::<T>,
            measure: measure::<T>,
            served: model::finetune::served_loss::<T>,
            prune: prune::<T>,
            teacher: teacher::<T>,
            distill: distill::<T>,
        }
    }
}

fn full<T: ChatTrainer>(base: &Path, dir: &Path, opts: &FitOpts, out: &Path, state: OptimizerState, control: FitControl<'_>) -> std::io::Result<FitReport> {
    model::finetune::fit_full::<T>(base, dir, opts, out, state, control)
}

fn preference<T: ChatTrainer>(job: PreferenceJob<'_>, control: FitControl<'_>) -> std::io::Result<(FitReport, ResidentMargins)> {
    let mut fit = DpoFit::new(job.dpo, job.pairs, job.score_sets, job.opts, control);
    let (report, trained) = T::train_lora(job.base, &job.lora, job.opts, &mut fit)?;
    if !report.interrupted {
        T::save_adapter(&trained, job.adapter, &job.lineage).map_err(|e| std::io::Error::new(e.kind(), format!("{}: saving the adapter: {e}", job.adapter.display())))?;
    }
    Ok((report, fit.into_margins()))
}

/// The search runs on the full-parameter build (the estimate needs every
/// linear's gradient), with the optimizer state on the host since no step is
/// taken.
fn search<T: ChatTrainer>(job: SearchJob<'_>, on_progress: &mut dyn FnMut(Progress)) -> std::io::Result<Report> {
    let SearchJob { base, dir, opts, tiers, budget, timing, frontier_points } = job;
    let mut fit = SearchFit { dir, opts, tiers, budget, on_progress, timing, frontier_points, report: None };
    T::train_full(base, opts, OptimizerState::HostOffload, &mut fit)?;
    fit.report.ok_or_else(|| std::io::Error::other("the weight search did not run"))
}

/// The base built as a LoRA trainer over a frozen base at `tiers`: the family's
/// own per-leaf build on its deployed kernels. A fresh adapter is a zero delta,
/// so the loss measured is the base's at those tiers; `adapter` continues a
/// trained one instead, so the loss is the base's with it.
fn measure<T: ChatTrainer>(base: &Path, dir: &Path, opts: &FitOpts, tiers: &TierPolicy, adapter: AdapterArg<'_>) -> std::io::Result<f64> {
    let mut fit = MeasureFit { dir, opts, loss: None };
    let (continue_from, rank, alpha) = adapter.map_or((None, 1, 1.0), |(path, rank, alpha)| (Some(path), rank, alpha));
    T::train_lora(base, &LoraRequest { rank, alpha, continue_from, base: tiers }, opts, &mut fit)?;
    fit.loss.ok_or_else(|| std::io::Error::other("the measurement did not run"))
}

/// Pruning ranks channels by gradients, so it runs on the full-parameter
/// build; no step is taken, so its optimizer state stays on the host.
fn prune<T: ChatTrainer>(job: PruneJob<'_>, on_progress: &mut dyn FnMut(Progress)) -> std::io::Result<PruneReport> {
    let PruneJob { base, dir, opts, width, out } = job;
    let mut fit = PruneFit { dir, opts, width, out, on_progress, report: None };
    T::train_full(base, opts, OptimizerState::HostOffload, &mut fit)?;
    fit.report.ok_or_else(|| std::io::Error::other("the pruning did not run"))
}

/// The teacher is the base itself, built trainable in fp32 (the build every
/// family offers); no step is taken, so its optimizer state stays on the host.
fn teacher<T: ChatTrainer>(base: &Path, dir: &Path, opts: &FitOpts, k: usize) -> std::io::Result<Vec<DistillExample>> {
    let mut fit = TeacherFit { dir, opts, k, examples: Vec::new() };
    T::train_full(base, opts, OptimizerState::HostOffload, &mut fit)?;
    Ok(fit.examples)
}

fn distill<T: ChatTrainer>(job: DistillJob<'_>, control: FitControl<'_>) -> std::io::Result<FitReport> {
    let mut fit = DistillFit::new(DistillConfig { seq_len: job.opts.block_size as usize }, job.examples, job.opts, control);
    let (report, trained) = T::train_lora(job.base, &job.lora, job.opts, &mut fit)?;
    if !report.interrupted {
        T::save_adapter(&trained, job.adapter, &job.lineage).map_err(|e| std::io::Error::new(e.kind(), format!("{}: saving the adapter: {e}", job.adapter.display())))?;
    }
    Ok(report)
}
