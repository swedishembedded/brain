// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What every forecaster's `finetune` action shares: the request (a universe
//! of OHLCV CSVs, windowing, optimizer, adapter), its validation before any
//! weight loads, and the gate's verdict. The loop itself is the model's own.
//!
//! The run fine-tunes on the past and promotes ONLY if the fine-tune beats the
//! base on a held-out, embargoed split: a weekly update has to earn its place
//! instead of chasing noise.
//!
//! Swedish Embedded AB implements gated fine-tuning pipelines for forecasting
//! models for its clients. If your team needs expertise in walk-forward
//! validation and leak-safe time-series training then you can procure our
//! services by sending an email to info@swedishembedded.com.

use capability::{ActionSpec, Invocation, ParamSpec, ParamType};
use forecast::finetune::FinetuneReport;
use forecast::train_data::{Series, SplitConfig};
use serde_json::json;

use crate::universe;

/// Bars of history one training window spans when the caller names none.
const DEFAULT_CONTEXT: i64 = 180;
/// Bars one training window forecasts when the caller names none.
const DEFAULT_HORIZON: i64 = 5;
/// Epochs when the caller names none.
const DEFAULT_EPOCHS: i64 = 8;
/// Learning rate when the caller names none.
const DEFAULT_LR: f64 = 4e-5;
/// Largest window, embargo, epoch or batch count a run may ask for.
const MAX_COUNT: f64 = 1_000_000.0;
/// Largest learning rate a run may ask for.
const MAX_LR: f64 = 1.0;

/// The `finetune` schema shared by every forecaster: the model adds its own
/// summary and the `activations` choice its trainer offers.
pub fn finetune_spec(summary: &str, default_save: &str) -> ActionSpec {
    ActionSpec::new("finetune", summary)
        .training()
        .streaming()
        .param(ParamSpec::new("data", ParamType::Str, "the universe: a directory of <TICKER>.csv files (Date,open,high,low,close,volume), every one validated before anything loads").required().host_resolved())
        .param(ParamSpec::new("holdout_data", ParamType::Str, "a second directory of names never trained on, to measure whether the fine-tune generalizes to unseen instruments").host_resolved())
        .param(ParamSpec::new("save", ParamType::Str, "where to write the promoted checkpoint (.safetensors); nothing is written when the gate keeps the base").default(json!(default_save)).host_resolved())
        .param(ParamSpec::new("context", ParamType::Int, "bars of history per training window").min(2.0).max(MAX_COUNT).default(json!(DEFAULT_CONTEXT)))
        .param(ParamSpec::new("horizon", ParamType::Int, "bars each training window forecasts").min(1.0).max(MAX_COUNT).default(json!(DEFAULT_HORIZON)))
        .param(ParamSpec::new("epochs", ParamType::Int, "passes over the training windows").min(1.0).max(capability::bounds::TRAIN_STEPS).default(json!(DEFAULT_EPOCHS)))
        .param(ParamSpec::new("lr", ParamType::Float, "AdamW learning rate").min(0.0).max(MAX_LR).default(json!(DEFAULT_LR)))
        .param(ParamSpec::new("lora", ParamType::Int, "LoRA adapter rank; 0 fine-tunes every weight").min(0.0).max(capability::bounds::LORA_RANK).default(json!(0)))
        .param(ParamSpec::new("embargo", ParamType::Int, "bars cleared either side of each calendar cut so no window straddles it (default: the horizon)").min(0.0).max(MAX_COUNT))
        .param(ParamSpec::new("batch", ParamType::Int, "windows per optimizer step").min(1.0).max(MAX_COUNT).default(json!(1)))
        .param(ParamSpec::new("skip_invalid", ParamType::Bool, "train on the files that parse instead of refusing the run, still listing every rejection with its file and line").default(json!(false)))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the LoRA adapters' initial factors (omit for random; the outcome echoes it; a full fine-tune draws nothing)").min(0.0))
}

/// A parsed `finetune` invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct FinetuneRequest {
    pub data: String,
    pub holdout_data: Option<String>,
    pub save: String,
    pub context: usize,
    pub horizon: usize,
    pub epochs: u32,
    pub lr: f32,
    /// `0` fine-tunes every weight.
    pub lora_rank: usize,
    pub embargo: usize,
    pub batch: u32,
    pub skip_invalid: bool,
    pub seed: u64,
    /// The trainer's `activations` choice, as named; the model parses it.
    pub activations: Option<String>,
}

impl FinetuneRequest {
    pub fn parse(inv: &Invocation, default_save: &str) -> Result<FinetuneRequest, String> {
        let data = inv.get_str("data").filter(|d| !d.is_empty()).ok_or("finetune: missing required param 'data'")?;
        let horizon = inv.get_i64("horizon").unwrap_or(DEFAULT_HORIZON).max(1) as usize;
        let context = inv.get_i64("context").unwrap_or(DEFAULT_CONTEXT).max(0) as usize;
        // A window shorter than two bars has no next token to predict, so every
        // window would be dropped by the tokenizer and the run would train on
        // nothing; a zero horizon leaves the gate scoring an empty future.
        if context < 2 {
            return Err(format!("finetune: context {context}, horizon {horizon}: need context >= 2 and horizon >= 1"));
        }
        Ok(FinetuneRequest {
            data,
            holdout_data: inv.get_str("holdout_data").filter(|d| !d.is_empty()),
            save: inv.get_str("save").filter(|s| !s.is_empty()).unwrap_or_else(|| default_save.to_string()),
            context,
            horizon,
            epochs: inv.get_i64("epochs").unwrap_or(DEFAULT_EPOCHS).max(1) as u32,
            lr: inv.get_f64("lr").unwrap_or(DEFAULT_LR) as f32,
            lora_rank: inv.get_i64("lora").unwrap_or(0).max(0) as usize,
            embargo: inv.get_i64("embargo").map_or(horizon, |e| e.max(0) as usize),
            batch: inv.get_i64("batch").unwrap_or(1).max(1) as u32,
            skip_invalid: inv.get_bool("skip_invalid").unwrap_or(false),
            seed: inv.seed()?.unwrap_or_else(seed::random_seed),
            activations: inv.get_str("activations"),
        })
    }

    /// The temporal split: the first 70% of each series' calendar trains, the
    /// next 15% validates, with the embargo cleared either side of each cut.
    pub fn split(&self) -> SplitConfig {
        SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo: self.embargo }
    }
}

/// The universes a run reads, validated.
pub struct Universes {
    pub series: Vec<Series>,
    pub holdout: Option<Vec<Series>>,
    /// What loading said about the data: rejected files with their lines, the
    /// loaded/rejected totals.
    pub notes: Vec<String>,
}

/// Read and validate the universes FIRST: a typo in `data` or one bad row
/// deserves an answer in milliseconds, not after several hundred MB of
/// checkpoint has loaded (and, for `holdout_data`, not after the whole
/// fine-tune has already run).
pub fn load_universes(req: &FinetuneRequest) -> Result<Universes, String> {
    let min_rows = req.context + req.horizon;
    let train = universe::load_accepted(&req.data, "finetune data", min_rows, req.skip_invalid)?;
    let mut notes = train.notes;
    if train.series.len() < 2 {
        return Err(format!("finetune: {}: {} usable series, need >= 2 to fine-tune a universe", req.data, train.series.len()));
    }
    let holdout = match &req.holdout_data {
        Some(dir) => {
            let h = universe::load_accepted(dir, "finetune holdout_data", min_rows, req.skip_invalid)?;
            notes.extend(h.notes);
            Some(h.series)
        }
        None => None,
    };
    Ok(Universes { series: train.series, holdout, notes })
}

/// Refuse a report that is not a verdict.
///
/// Zero steps means the embargoed split left no training window at all, and
/// printing "KEEP BASE" for it would read as "the fine-tune was tried and
/// lost" when nothing was ever trained. Nor is an empty VALIDATION slice a
/// verdict: the split clears an embargo band either side of each calendar cut,
/// so a universe whose series are short relative to the embargo can train and
/// still have nothing to be judged on.
pub fn require_verdict(req: &FinetuneRequest, series: usize, rep: &FinetuneReport) -> Result<(), String> {
    if rep.steps == 0 {
        return Err(format!(
            "finetune: 0 training steps - the {series} series in {} yielded no TRAIN windows after the embargoed temporal split (train_frac 0.7, embargo {} bars). Supply longer series, or shorten context/horizon/embargo.",
            req.data, req.embargo
        ));
    }
    if !rep.base_val.is_finite() || !rep.ft_val.is_finite() {
        return Err(format!(
            "finetune: trained {} steps but the embargoed split left no VALIDATION window (val_frac 0.15, embargo {} bars, batch {}), so there is no held-out verdict to report. Supply longer series, or shorten embargo/batch.",
            rep.steps, req.embargo, req.batch
        ));
    }
    Ok(())
}

/// The gate line: the held-out loss before and after, and the decision.
pub fn gate_line(rep: &FinetuneReport) -> String {
    format!(
        "gate (INCLUDED names, held-out future): base_val {:.4} -> ft_val {:.4}  ({} steps)  =>  {}",
        rep.base_val,
        rep.ft_val,
        rep.steps,
        if rep.promoted { "PROMOTE (fine-tune beats base out-of-sample)" } else { "KEEP BASE (no held-out improvement)" }
    )
}

/// The generalization line: the loss on names the fine-tune never saw.
pub fn generalization_line(names: usize, base: f32, tuned: f32) -> String {
    format!(
        "held-out NAMES ({names} names, never fine-tuned): base {base:.4} -> ft {tuned:.4}  =>  {}",
        if tuned < base { "GENERALIZES (more accurate on unseen instruments)" } else { "no gain on unseen names" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(steps: u32, base: f32, ft: f32) -> FinetuneReport {
        FinetuneReport { promoted: forecast::finetune::promoted(base, ft), base_val: base, ft_val: ft, steps, lora_seed: None }
    }

    #[test]
    fn the_request_defaults_the_embargo_to_the_horizon_and_echoes_a_given_seed() {
        let inv = Invocation::new().set("data", json!("/d")).set("horizon", json!(9)).set("seed", json!(4));
        let r = FinetuneRequest::parse(&inv, "x.safetensors").unwrap();
        assert_eq!((r.embargo, r.seed, r.context, r.save.as_str()), (9, 4, 180, "x.safetensors"));
        assert_eq!(r.split().embargo, 9);
        let explicit = FinetuneRequest::parse(&inv.set("embargo", json!(0)), "x").unwrap();
        assert_eq!(explicit.embargo, 0, "an explicit zero embargo is honoured, not replaced by the horizon");
    }

    #[test]
    fn a_window_too_short_to_hold_a_next_bar_or_a_missing_universe_is_refused() {
        let e = FinetuneRequest::parse(&Invocation::new().set("data", json!("/d")).set("context", json!(1)), "x").unwrap_err();
        assert!(e.contains("context >= 2"), "{e}");
        let e = FinetuneRequest::parse(&Invocation::new(), "x").unwrap_err();
        assert!(e.contains("missing required param 'data'"), "{e}");
    }

    #[test]
    fn zero_steps_and_an_empty_validation_slice_are_errors_not_verdicts() {
        let req = FinetuneRequest::parse(&Invocation::new().set("data", json!("/d")), "x").unwrap();
        assert!(require_verdict(&req, 3, &report(0, 1.0, f32::NAN)).unwrap_err().contains("0 training steps"));
        assert!(require_verdict(&req, 3, &report(5, f32::NAN, f32::NAN)).unwrap_err().contains("no VALIDATION window"));
        assert!(require_verdict(&req, 3, &report(5, 1.0, 0.9)).is_ok());
    }

    #[test]
    fn the_gate_line_names_the_decision() {
        assert!(gate_line(&report(5, 1.0, 0.9)).contains("PROMOTE"));
        assert!(gate_line(&report(5, 1.0, 1.1)).contains("KEEP BASE"));
        assert!(generalization_line(4, 1.0, 0.9).contains("GENERALIZES"));
        assert!(generalization_line(4, 1.0, 1.0).contains("no gain"));
    }
}
