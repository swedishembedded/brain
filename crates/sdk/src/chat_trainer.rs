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
use model::finetune::{AdapterLineage, ChatTrainer, LoraRequest, ModelFit, OptimizerState, PreferenceMargins};
use model::{FitControl, FitOpts, FitReport, Model};
use rl::objective::dpo::{Dpo, DpoConfig, PackedPair};

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

type FullFn = for<'c> fn(&Path, &Path, &FitOpts, &Path, FitControl<'c>) -> std::io::Result<FitReport>;
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
            scores_exported_pairs: T::SCORES_EXPORTED_PAIRS,
        }
    }
}

fn full<T: ChatTrainer>(base: &Path, dir: &Path, opts: &FitOpts, out: &Path, control: FitControl<'_>) -> std::io::Result<FitReport> {
    model::finetune::fit_full::<T>(base, dir, opts, out, OptimizerState::HostOffload, control)
}

/// DPO over whichever model the trainer builds: the objective is made per
/// concrete type from the packed pairs the job holds, and the pair sets to
/// score are measured on that model before and after.
struct DpoFit<'a, 'c> {
    dpo: DpoConfig,
    pairs: Option<Vec<PackedPair>>,
    score_sets: Vec<Vec<PackedPair>>,
    margins: ResidentMargins,
    opts: &'a FitOpts,
    control: Option<FitControl<'c>>,
}

impl ModelFit for DpoFit<'_, '_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let (Some(pairs), Some(control)) = (self.pairs.take(), self.control.take()) else {
            return Err(std::io::Error::other("a preference fit trains one model"));
        };
        // The starting policy is the reference, as it is for the training objective.
        for set in &mut self.score_sets {
            set.iter_mut().for_each(|pair| pair.score_reference(&model));
        }
        let (report, model) = model::fit_controlled(model, Dpo::from_dataset(self.dpo, pairs), self.opts, None, control)?;
        if !report.interrupted {
            self.margins = self.score_sets.iter().map(|set| set.iter().map(|pair| pair.margin(&model)).collect()).collect();
        }
        Ok((report, model))
    }
}

fn preference<T: ChatTrainer>(job: PreferenceJob<'_>, control: FitControl<'_>) -> std::io::Result<(FitReport, ResidentMargins)> {
    let mut fit = DpoFit { dpo: job.dpo, pairs: Some(job.pairs), score_sets: job.score_sets, margins: Vec::new(), opts: job.opts, control: Some(control) };
    let (report, trained) = T::train_lora(job.base, &job.lora, job.opts, &mut fit)?;
    if !report.interrupted {
        T::save_adapter(&trained, job.adapter, &job.lineage).map_err(|e| std::io::Error::new(e.kind(), format!("{}: saving the adapter: {e}", job.adapter.display())))?;
    }
    Ok((report, fit.margins))
}
