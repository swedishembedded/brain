// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for fine-tuning open-weight language
// models of any architecture through one training stack, for its clients. If
// your team needs expertise in wiring a new model family into full-parameter,
// adapter and preference training then you can procure our services by sending
// an email to info@swedishembedded.com.

//! The seam between the generic fine-tuners and a model family.
//!
//! A fine-tuner (full-parameter, LoRA, preference) is written once, over
//! [`Model`]: it owns the data, the objective, the schedule and the loop
//! ([`crate::fit_controlled`]). A model family contributes only what it alone
//! knows, by implementing [`ChatTrainer`]:
//!
//! * how to open a checkpoint and build the model for training (placement,
//!   frozen-base tier, adapter layout), and
//! * how to write what training produced (an adapter, a checkpoint directory).
//!
//! The family hands the model it built to a [`ModelFit`] visitor instead of
//! returning it, so a family whose trainable model has more than one concrete
//! type (one card or a pipeline of cards) needs no enum the generic code would
//! have to name: the visitor is instantiated once per type. Adding a family is
//! one [`ChatTrainer`] impl; nothing here, in `rl` or in the SDK names it.
//!
//! The optimizer state of a full fine-tune is chosen by [`OptimizerState`], where
//! its moments live, and [`crate::FitOpts::precision`], the format of everything a
//! run stores: both are passed to the model's builder, which refuses a choice it
//! does not run.

use std::path::{Path, PathBuf};

use gpu_core::select::Dtype;

use crate::train::{causal_lm, load_dataset_with_itos, FitControl, FitOpts, FitReport};
use crate::Model;

/// Where the AdamW moments of a full-parameter run live.
///
/// The precision of the moments is the run's [`crate::FitOpts::precision`]; fp32 unless a run says otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerState {
    /// On the device beside the weights and gradients.
    Device,
    /// In system RAM, only weights and gradients on the device.
    HostOffload,
}

impl OptimizerState {
    /// Run `build` (a model's construction) with this state selected.
    pub fn scope<R>(self, build: impl FnOnce() -> R) -> R {
        crate::offload::with_adam_offloaded(self == OptimizerState::HostOffload, build)
    }
}

/// A training run over whichever concrete [`Model`] a family builds.
pub trait ModelFit {
    /// Train `model` and hand it back with the run's report.
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)>;
}

/// Wall-clock stage timer printed under `BRAIN_PROFILE`, for callers that do not depend on `gpu-core`.
pub use gpu_core::profile::stage_time;

/// Next-token cross-entropy over a prepared dataset: the supervised fine-tune.
pub struct CausalLmFit<'a, 'c> {
    /// The prepared dataset directory ([`data::chat::prepare_chat_samples`]).
    pub dir: &'a Path,
    pub opts: &'a FitOpts,
    /// Where the finished (and best, when selecting) model is written, if anywhere.
    pub out: Option<&'a Path>,
    /// Consumed by the one run this value performs.
    pub control: Option<FitControl<'c>>,
}

impl ModelFit for CausalLmFit<'_, '_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let control = self.control.take().ok_or_else(|| std::io::Error::other("a CausalLmFit trains one model"))?;
        let (train, val, bcfg, _vocab, itos) = load_dataset_with_itos(self.dir, self.opts)?;
        crate::fit_controlled(model, causal_lm::<M>(train, val, bcfg, itos), self.opts, self.out, control)
    }
}

/// A full-parameter fine-tune of the checkpoint at `base` on the prepared
/// dataset in `dir`, written to `out`: every weight trains in fp32 with the
/// moments where `state` says and in the format `opts.precision` names. Always a fresh start from `base`; a run stopped
/// by `control` writes nothing (the optimizer state a resume needs is three
/// times the model), so the report says it was interrupted and `out` is left
/// absent.
pub fn fit_full<T: ChatTrainer>(base: &Path, dir: &Path, opts: &FitOpts, out: &Path, state: OptimizerState, control: FitControl<'_>) -> std::io::Result<FitReport> {
    T::TRAIN_PRECISION.check(&opts.precision).map_err(|why| std::io::Error::new(std::io::ErrorKind::Unsupported, why))?;
    let mut fit = CausalLmFit { dir, opts, out: Some(out), control: Some(control) };
    let resident = opts.precision.resident_config(Some(opts.seed)).map_err(std::io::Error::other)?;
    // The host half of the run (decoding the base, the optimizer update, every transfer) works next to the card.
    let near = gpu_core::devices::selected_device().and_then(|card| gpu_core::devices::host_cores_near(card.index)).unwrap_or_default();
    if gpu_core::profile::enabled() {
        eprintln!("host threads: {} cores next to the card", near.len());
    }
    let (report, _) = backend_cpu::par::with_cores(&near, || match &resident {
        // f32 layers stream through the plain build: the fit loop hands the request to `Model::set_resident`.
        Some(resident) if resident.value == crate::resident_ft::ResidentValue::F32 => T::train_full(base, opts, state, &mut fit),
        Some(resident) => T::train_full_resident(base, opts, resident, &mut fit),
        None => T::train_full(base, opts, state, &mut fit),
    })?;
    if report.interrupted {
        // fit_controlled does not write an interrupted run's model; make sure
        // a periodic save did not leave a partial one under the final name.
        let _ = std::fs::remove_file(out);
    }
    Ok(report)
}

/// A resident low-precision full-parameter fine-tune of the checkpoint at `base` on the prepared dataset in
/// `dir`, written to `out` as the family's brain checkpoint (from the master copies): the linears train as
/// bf16 or FP8 weights resident on the device with their master and optimizer state off it
/// ([`crate::resident_ft`]), the rest in fp32. Hands the trained model back so the caller can export it in the
/// chosen format. Like [`fit_full`], always a fresh start from `base` (resume is through `control`'s state).
pub fn fit_full_resident<T: ChatTrainer>(
    base: &Path,
    dir: &Path,
    opts: &FitOpts,
    out: &Path,
    resident: &crate::resident_ft::ResidentFtConfig,
    control: FitControl<'_>,
) -> std::io::Result<(FitReport, T::Trained)> {
    let mut fit = CausalLmFit { dir, opts, out: Some(out), control: Some(control) };
    let (report, trained) = T::train_full_resident(base, opts, resident, &mut fit)?;
    if report.interrupted {
        let _ = std::fs::remove_file(out);
    }
    Ok((report, trained))
}

/// Where a LoRA adapter starts and its shape.
#[derive(Clone, Copy, Debug)]
pub struct LoraRequest<'a> {
    pub rank: u32,
    pub alpha: f32,
    /// An adapter file whose factors training continues; `None`: a fresh
    /// zero-delta adapter drawn from the run's seed.
    pub continue_from: Option<&'a Path>,
    /// How the frozen base is held on the device; one of the trainer's
    /// [`ChatTrainer::LORA_BASES`]. The generic fine-tuners refuse any other
    /// before a checkpoint is opened ([`LoraRequest::checked`]).
    pub base: Dtype,
}

impl LoraRequest<'_> {
    /// The base format, if `T` can train over it.
    pub fn checked<T: ChatTrainer>(&self) -> std::io::Result<Dtype> {
        T::LORA_BASES.check(self.base).map_err(|why| std::io::Error::new(std::io::ErrorKind::Unsupported, why))
    }
}

/// What an adapter file records about where it came from.
pub struct AdapterLineage<'a> {
    pub card_id: &'a str,
    pub base_id: &'a str,
    pub dataset_id: Option<&'a str>,
    pub provenance: Option<checkpoint::st::TrainingProvenance>,
}

/// A model family's fine-tuning entry points. See the module docs.
pub trait ChatTrainer {
    /// What a run hands back (the model, or a wrapper that can save it).
    type Trained;

    /// The frozen-base formats [`Self::train_lora`] can hold the base in.
    const LORA_BASES: crate::policy::TrainBases;

    /// What of [`crate::TrainPrecision`] this family's full fine-tune runs. A request outside it is refused by
    /// [`fit_full`] before anything is opened, naming the family and the key.
    const TRAIN_PRECISION: crate::TrainPrecisionSupport;

    /// The weights file or directory to open for the base the caller named:
    /// `weights` is the resolved file, `model_dir` the directory it lives in.
    fn checkpoint_path(weights: &Path, model_dir: &Path) -> PathBuf;

    /// Whether [`Self::preference_margins`] scores pairs after the run from the
    /// exported adapter. A family that does not is scored on the resident
    /// model instead (reference = the model the run starts from).
    const SCORES_EXPORTED_PAIRS: bool = false;

    /// The vocabulary of the checkpoint at `open`.
    fn vocab(open: &Path) -> Result<usize, String>;

    /// Train every parameter of the checkpoint at `base` through `fit`.
    fn train_full(base: &Path, opts: &FitOpts, state: OptimizerState, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Self::Trained)>;

    /// Train every parameter of the checkpoint at `base` with its linears resident in low precision
    /// ([`crate::resident_ft`]). A family that has not wired the resident state refuses by name.
    fn train_full_resident(_base: &Path, _opts: &FitOpts, _resident: &crate::resident_ft::ResidentFtConfig, _fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Self::Trained)> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "this model family has no resident low-precision full fine-tune"))
    }

    /// Train a LoRA adapter over the frozen checkpoint at `base`.
    fn train_lora(base: &Path, lora: &LoraRequest<'_>, opts: &FitOpts, fit: &mut impl ModelFit) -> std::io::Result<(FitReport, Self::Trained)>;

    /// Write the trained adapter (never the frozen base) to `path`.
    fn save_adapter(trained: &Self::Trained, path: &Path, lineage: &AdapterLineage<'_>) -> std::io::Result<()>;

    /// Write the trained model, a brain-format file at `trained`, as a
    /// checkpoint directory at `out` the family's loaders and `brain serve`
    /// open as a base, with the tokenizer files of `tokenizer_dir`; `adapter`
    /// is folded in when given.
    fn export_checkpoint(trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>) -> Result<(), String>;

    /// Per-pair preference margins of the adapter at `policy` against
    /// `reference` (or the base), `None` when the family does not score
    /// preference pairs.
    fn preference_margins(
        _weights: &Path,
        _reference: Option<&Path>,
        _policy: &Path,
        _pairs: &[(data::preference::EncodedTurn, data::preference::EncodedTurn)],
        _block: u32,
    ) -> Option<PreferenceMargins> {
        None
    }
}

/// Per-pair preference margins of a policy against a reference.
#[derive(Debug, Clone, PartialEq)]
pub struct PreferenceMargins {
    /// `(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log
    /// ref(rejected))` per scored pair, in nats, each log-probability summed
    /// over that candidate's supervised tokens only.
    pub margins: Vec<f64>,
    /// Pairs not scored because a candidate does not fit the scoring row.
    pub skipped: usize,
}

/// The [`ChatTrainer::export_checkpoint`] of a family with no Hugging Face
/// writer: a checkpoint directory holding the brain-format weights as
/// `model.safetensors` and the tokenizer files of `tokenizer_dir`. An adapter
/// is not folded in.
pub fn export_brain_checkpoint(family: &str, trained: &Path, tokenizer_dir: &Path, out: &Path, adapter: Option<&Path>) -> Result<(), String> {
    if adapter.is_some() {
        return Err(format!("{family} does not fold an adapter into a checkpoint yet"));
    }
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let copy = |from: &Path, to: &Path| std::fs::copy(from, to).map(drop).map_err(|e| format!("{}: {e}", from.display()));
    copy(trained, &out.join("model.safetensors"))?;
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"] {
        let from = tokenizer_dir.join(f);
        if from.is_file() {
            copy(&from, &out.join(f))?;
        }
    }
    Ok(())
}
