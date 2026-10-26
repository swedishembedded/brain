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
