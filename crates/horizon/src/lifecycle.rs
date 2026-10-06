// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `train`, `eval` and `calibrate` actions of the timeline capability: the
//! life cycle of a model around `predict` ([`crate::caps`]).
//!
//! Swedish Embedded AB implements the whole path from a client's records to a
//! checked, calibrated risk model that can be served. If your team needs
//! expertise in training, validating and calibrating time-to-event models on
//! your own data you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Inputs are `timeline-v1` blobs (every line validated at entry; an error
//! names the line). Nothing here knows the domain.
//!
//! - `train` fits a model (or, with `members` above one, an ensemble:
//!   [`crate::ensemble`]) on `dataset`, early-stopping on `held_out`. It is
//!   long-running: it reports progress every evaluation interval, polls the
//!   invocation's cancel token after every optimiser step and, cancelled or
//!   failed, leaves NO model directory behind: the directory is written beside
//!   its final name and renamed into place only when training is complete.
//!   An existing directory is refused before any training starts.
//! - `eval` judges a saved model on a dataset it was neither trained,
//!   early-stopped nor calibrated on (`TimelineModel::evaluate`), as JSON.
//! - `calibrate` fits calibrators (logistic by default, or Venn-Abers) on a
//!   validation set
//!   ([`crate::saved::Saved::calibrate`]'s arithmetic). Through the CLI it
//!   writes `calibration.json` into the model directory, or a calibrated copy
//!   into `out`, refusing to overwrite without `force`; on a served surface it
//!   only returns the calibration, since the served directory is the host's.

use std::collections::BTreeSet;
use std::path::Path;

use capability::{
    ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress,
};
use serde_json::{json, Value};

use crate::calibration::{self, Calibration, MIN_EVENTS};
use crate::config::Mixer;
use crate::ensemble::{Ensemble, Kind, Loaded};
use crate::evaluation::{evaluate, EvaluationSpec};
use crate::fit::{
    self, Hooks, Report, StepProgress, TrainError, TrainSpec, DEFAULT_BATCH, DEFAULT_EVAL_INTERVAL, DEFAULT_PATIENCE,
    DEFAULT_STEPS,
};
use crate::saved::{parse_jsonl, Saved};
use crate::timeline::Subject;

/// The host's directory a served `train` writes its model to.
pub const TRAIN_DIR_VAR: &str = "BRAIN_HORIZON_TRAIN_DIR";

/// The `train` action's schema.
pub fn train_spec() -> ActionSpec {
    ActionSpec::new(
        "train",
        "train a timeline model (or an ensemble) on a timeline-v1 dataset, early-stopping on held-out subjects; the model directory appears only when training completes",
    )
    .streaming()
    .param(
        ParamSpec::new(
            "out",
            ParamType::Str,
            "directory the finished model is written to (must not exist); nothing is left there if training is cancelled or fails",
        )
        .required()
        .host_env(TRAIN_DIR_VAR),
    )
    .param(ParamSpec::new("force", ParamType::Bool, "replace an existing model directory at 'out' (after training completes)").default(json!(false)).host_resolved())
    .param(ParamSpec::new("codes", ParamType::Str, "comma-separated outcome codes; default: every event code that occurs after a subject's entry").default(json!("")))
    .param(ParamSpec::new("absorbing", ParamType::Str, "comma-separated codes (a subset) that end follow-up for every code, e.g. deaths").default(json!("")))
    .param(ParamSpec::new("knots", ParamType::Str, "comma-separated hazard piece boundaries after entry, starting at 0 and ascending, in the data's unit; default: the model's own (0 to 22 years)").default(json!("")))
    .param(ParamSpec::new("steps", ParamType::Int, "optimiser steps at most (early stopping usually ends sooner)").default(json!(DEFAULT_STEPS)).min(1.0).max(1_000_000.0))
    .param(ParamSpec::new("batch", ParamType::Int, "subjects per batch").default(json!(DEFAULT_BATCH)).min(1.0).max(4096.0))
    .param(ParamSpec::new("seed", ParamType::Int, "seed of the initial weights, the batches and the masks (member i of an ensemble uses seed + i)").default(json!(1)).min(0.0))
    .param(ParamSpec::new("eval_interval", ParamType::Int, "steps between held-out evaluations (and progress reports)").default(json!(DEFAULT_EVAL_INTERVAL)).min(1.0))
    .param(ParamSpec::new("patience", ParamType::Int, "held-out evaluations without improvement before training stops").default(json!(DEFAULT_PATIENCE)).min(1.0))
    .param(ParamSpec::new("next_events", ParamType::Str, "comma-separated event codes to also model as 'which happens first, and when' (a self-supervised signal); default none").default(json!("")))
    .param(ParamSpec::new("next_weight", ParamType::Float, "weight of the next-event objective against the outcome codes").default(json!(0.5)).min(0.0))
    .param(
        ParamSpec::new(
            "mixer",
            ParamType::Enum(vec!["attention".into(), "gated-delta-net".into(), "hybrid".into()]),
            "read the visit history through a stack of blocks mixed this way; default: the single-state encoder",
        ),
    )
    .param(ParamSpec::new("blocks", ParamType::Int, "residual blocks of the mixer stack").default(json!(2)).min(1.0).max(64.0))
    .param(ParamSpec::new("visits", ParamType::Int, "most recent visits read one by one with a mixer (0: no mixer stack, one set; with a mixer and none given: 4)").default(json!(0)).min(0.0).max(64.0))
    .param(ParamSpec::new("forecasts", ParamType::Int, "future measurements per subject to train a value-forecast head on (0: no head)").default(json!(0)).min(0.0).max(64.0))
    .param(ParamSpec::new("members", ParamType::Int, "1 trains one model; 2 or more train an ensemble whose spread is the uncertainty").default(json!(1)).min(1.0).max(64.0))
    .param(ParamSpec::new("ensemble", ParamType::Enum(vec!["seeded".into(), "bootstrap".into()]), "how ensemble members differ: another seed each, or subjects resampled with replacement by group").default(json!("seeded")))
    .input(BlobSpec::new("dataset", Media::Text, "training subjects: timeline-v1, one per line").required())
    .input(BlobSpec::new("held_out", Media::Text, "early-stopping subjects: timeline-v1, none of them in the training set").required())
    .output(BlobSpec::new("report", Media::Text, "JSON: the kind, and per member its seed, steps, losses, held-out event NLL, parameters and weights digest"))
}

/// The `eval` action's schema.
pub fn eval_spec() -> ActionSpec {
    ActionSpec::new(
        "eval",
        "judge a saved model on held-out subjects: per outcome and horizon Uno C, time-dependent AUC, IPCW Brier, integrated Brier, calibration and event NLL",
    )
    .param(
        ParamSpec::new("weights", ParamType::Str, "directory of a saved timeline model (not an ensemble: evaluate a member)")
            .required()
            .host_env(crate::caps::DIR_VAR),
    )
    .param(ParamSpec::new("horizons", ParamType::Str, "comma-separated horizons after entry, each within the model's knots").default(json!(crate::caps::DEFAULT_TIMES)))
    .param(ParamSpec::new("min_events", ParamType::Int, "a (code, horizon) with fewer events by then is left out and listed under 'absent'").default(json!(MIN_EVENTS)).min(1.0))
    .input(BlobSpec::new("dataset", Media::Text, "held-out subjects: timeline-v1, never trained, early-stopped or calibrated on").required())
    .output(BlobSpec::new("evaluation", Media::Text, "JSON: {subjects, event_nll, results: [{code, horizon, events, uno_c, auc, brier, integrated_brier, calibration, intervals?}], absent}"))
}

/// The `calibrate` action's schema.
pub fn calibrate_spec() -> ActionSpec {
    ActionSpec::new(
        "calibrate",
        "fit calibrators for the given horizons on validation subjects the model was neither trained nor early-stopped on",
    )
    .param(
        ParamSpec::new("weights", ParamType::Str, "directory of a saved timeline model (not an ensemble)")
            .required()
            .host_env(crate::caps::DIR_VAR),
    )
    .param(ParamSpec::new("horizons", ParamType::Str, "comma-separated horizons after entry to calibrate, each within the model's knots").required())
    .param(ParamSpec::new("kind", ParamType::Str, "logistic (intercept, and slope where the data show it is not one; no interval) or venn_abers (isotonic with its interval)").default(json!(calibration::Kind::default().name())))
    .param(ParamSpec::new("min_events", ParamType::Int, "a (code, horizon) with fewer validation events by then (and as many still event-free) is not calibrated; default 100 for logistic, 500 for venn_abers").min(1.0))
    .param(ParamSpec::new("out", ParamType::Str, "write a calibrated copy of the model here instead of calibration.json into 'weights' (must not exist)").host_resolved())
    .param(ParamSpec::new("force", ParamType::Bool, "overwrite an existing calibration.json (or the directory at 'out')").default(json!(false)).host_resolved())
    .input(BlobSpec::new("validation", Media::Text, "validation subjects: timeline-v1").required())
    .output(BlobSpec::new("calibration", Media::Text, "the calibration.json content: bound to the weights' SHA-256, per (code, horizon) the events and the calibrator"))
}

/// `outcome` with every top-level field of the JSON object `value` as an
/// output, so `--json` prints the object itself.
fn with_fields(mut outcome: Outcome, value: &Value) -> Outcome {
    for (key, field) in value.as_object().into_iter().flatten() {
        outcome = outcome.set(key, field.clone());
    }
    outcome
}

/// Replace the file `path` with `bytes` atomically: written to a sibling and
/// renamed over it, so a reader sees the old file or the whole new one.
fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{}: not a usable file name", path.display()))?;
    let tmp = path.with_file_name(format!(".{name}.new-{}", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })
}

fn missing(what: &str) -> String {
    format!("horizon: missing required {what}")
}

/// The subjects of the input blob `name`; an error names the input, and the
/// line of a bad one.
fn subjects_of(inv: &Invocation, name: &str) -> Result<Vec<Subject>, String> {
    let blob = inv.get_blob(name).ok_or_else(|| missing(&format!("input '{name}'")))?;
    let text = std::str::from_utf8(&blob.bytes).map_err(|e| format!("horizon: {name} is not UTF-8: {e}"))?;
    let subjects = parse_jsonl(text).map_err(|e| format!("horizon: {name} {e}"))?;
    if subjects.is_empty() {
        return Err(format!("horizon: {name} holds no subjects"));
    }
    Ok(subjects)
}

fn list(text: Option<String>) -> Vec<String> {
    text.unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(String::from)
        .collect()
}

fn numbers(name: &str, text: &str) -> Result<Vec<f64>, String> {
    list(Some(text.to_string()))
        .iter()
        .map(|t| t.parse::<f64>().map_err(|e| format!("horizon: {name} '{t}': {e}")))
        .collect()
}

fn horizons(inv: &Invocation, required: bool) -> Result<Vec<f64>, String> {
    let text = inv.get_str("horizons").unwrap_or_default();
    let parsed = numbers("horizons", &text)?;
    if parsed.is_empty() && required {
        return Err(missing("param 'horizons'"));
    }
    Ok(parsed)
}

fn int_param(inv: &Invocation, name: &str, default: i64) -> i64 {
    inv.get_i64(name).unwrap_or(default)
}

fn count(inv: &Invocation, name: &str, default: i64) -> Result<u32, String> {
    u32::try_from(int_param(inv, name, default)).map_err(|_| format!("horizon: {name} is out of range"))
}

/// The codes a dataset's outcomes are, when the caller names none: every event
/// code that occurs after a subject's entry, in order.
fn outcome_codes(subjects: &[Subject]) -> Vec<String> {
    let seen: BTreeSet<&str> = subjects
        .iter()
        .flat_map(|s| s.events.iter().filter(|e| e.t > s.entry).map(|e| e.code.as_str()))
        .collect();
    seen.into_iter().map(String::from).collect()
}

/// The settings of a `train` invocation over its training subjects.
fn train_settings(inv: &Invocation, train: &[Subject]) -> Result<TrainSpec, String> {
    let mut codes = list(inv.get_str("codes"));
    if codes.is_empty() {
        codes = outcome_codes(train);
    }
    if codes.is_empty() {
        return Err("horizon: no outcome codes: the dataset has no event after any subject's entry; name them with 'codes'".into());
    }
    let mut spec = TrainSpec::new(codes, list(inv.get_str("absorbing")))
        .steps(count(inv, "steps", i64::from(DEFAULT_STEPS))?)
        .batch(count(inv, "batch", i64::from(DEFAULT_BATCH))?)
        .seed(u64::try_from(int_param(inv, "seed", 1)).map_err(|_| "horizon: seed is out of range".to_string())?)
        .eval_interval(count(inv, "eval_interval", i64::from(DEFAULT_EVAL_INTERVAL))?)
        .patience(count(inv, "patience", i64::from(DEFAULT_PATIENCE))?);
    let knots = numbers("knots", &inv.get_str("knots").unwrap_or_default())?;
    if !knots.is_empty() {
        spec = spec.knots(knots.iter().map(|&k| k as f32).collect());
    }
    let next = list(inv.get_str("next_events"));
    if !next.is_empty() {
        spec = spec.next_events(next, inv.get_f64("next_weight").unwrap_or(0.5) as f32);
    }
    let forecasts = count(inv, "forecasts", 0)?;
    if forecasts > 0 {
        spec = spec.forecasts(forecasts, 0.5);
    }
    let visits = count(inv, "visits", 0)?;
    match inv.get_str("mixer").as_deref() {
        Some(name) => {
            let mixer = match name {
                "attention" => Mixer::Attention,
                "gated-delta-net" => Mixer::GatedDeltaNet,
                "hybrid" => Mixer::Hybrid,
                other => return Err(format!("horizon: unknown mixer '{other}'")),
            };
            spec = spec.visits(if visits == 0 { 4 } else { visits }).mixer(mixer, count(inv, "blocks", 2)?);
        }
        None if visits > 0 => spec = spec.visits(visits),
        None => {}
    }
    Ok(spec)
}

fn member_report(report: &Report, seed: u64, digest: &str) -> Value {
    json!({
        "seed": seed,
        "steps": report.steps,
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "held_out_event_nll": report.held_out_event_nll,
        "parameters": report.parameters,
        "truncated_tokens": report.truncated_tokens,
        "weights_sha256": digest,
    })
}

fn cancelled_or(e: TrainError) -> String {
    match e {
        TrainError::Cancelled => "cancelled".to_string(),
        TrainError::Failed(why) => format!("horizon: {why}"),
    }
}

fn describe(p: &StepProgress, member: Option<(usize, usize)>) -> Progress {
    let (step, total, label) = match member {
        Some((i, n)) => (i as u32 * p.steps + p.step, n as u32 * p.steps, format!("member {}/{n} ", i + 1)),
        None => (p.step, p.steps, String::new()),
    };
    let nll = p.held_out_event_nll.map_or("-".to_string(), |v| format!("{v:.4}"));
    Progress {
        event: Some(json!({
            "member": member.map(|(i, _)| i),
            "step": p.step,
            "steps": p.steps,
            "loss": p.loss,
            "held_out_event_nll": p.held_out_event_nll,
        })),
        ..Progress::step(step, total, format!("{label}step {} loss {:.4} held-out event NLL {nll}", p.step, p.loss))
    }
}

/// `train`: see the module docs.
pub fn train(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let out = inv.get_str("out").ok_or_else(|| missing("param 'out'"))?;
    let force = inv.get_bool("force").unwrap_or(false);
    // Fail before hours of training, not after.
    if Path::new(&out).exists() && !force {
        return Err(format!("horizon: {out}: already exists (not overwritten without force)"));
    }
    let train = subjects_of(inv, "dataset")?;
    let held_out = subjects_of(inv, "held_out")?;
    let spec = train_settings(inv, &train)?;
    let members = count(inv, "members", 1)? as usize;
    let cancelled = || inv.cancel.is_cancelled();

    let (kind, member_reports) = if members <= 1 {
        let mut on_progress = |p: &StepProgress| progress(describe(p, None));
        let mut hooks = Hooks { progress: Some(&mut on_progress), cancelled: Some(&cancelled) };
        let (saved, report) = fit::train(&train, &held_out, &spec, &mut hooks).map_err(cancelled_or)?;
        if cancelled() {
            return Err("cancelled".into());
        }
        saved.save_new(Path::new(&out), force).map_err(|e| format!("horizon: {e}"))?;
        let digest = saved.weights_digest().map_err(|e| format!("horizon: {e}"))?;
        ("single", vec![member_report(&report, spec.run_seed(), &digest)])
    } else {
        let kind = Kind::parse(&inv.get_str("ensemble").unwrap_or_else(|| "seeded".into()))
            .ok_or("horizon: ensemble must be seeded or bootstrap")?;
        let (ensemble, reports) = Ensemble::train(
            &train,
            &held_out,
            &spec,
            members,
            kind,
            &mut |i, p| progress(describe(p, Some((i, members)))),
            &cancelled,
        )
        .map_err(cancelled_or)?;
        if cancelled() {
            return Err("cancelled".into());
        }
        ensemble.save(Path::new(&out), force).map_err(|e| format!("horizon: {e}"))?;
        let records = &ensemble.manifest().members;
        (
            kind.name(),
            reports.iter().zip(records).map(|(r, m)| member_report(r, m.seed, &m.weights_sha256)).collect(),
        )
    };
    let report = json!({
        "kind": kind,
        "codes": spec.codes(),
        "train_subjects": train.len(),
        "held_out_subjects": held_out.len(),
        "members": member_reports,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| format!("horizon: {e}"))?;
    Ok(with_fields(Outcome::new(), &report).blob("report", Blob::new(Media::Text, text.into_bytes())))
}

/// `eval` on a loaded model.
pub fn eval(loaded: &Loaded, inv: &Invocation) -> ActionResult {
    let saved = match loaded {
        Loaded::Single(s) => s,
        Loaded::Ensemble(_) => {
            return Err("horizon: eval takes a single model, not an ensemble: evaluate a member (members/<n>)".into())
        }
    };
    let subjects = subjects_of(inv, "dataset")?;
    for s in &subjects {
        saved.vocab.check_units(s).map_err(|e| format!("horizon: {e}"))?;
    }
    let horizons = horizons(inv, true)?;
    let mut spec = EvaluationSpec::new(horizons);
    if let Some(n) = inv.get_i64("min_events") {
        spec = spec.min_events(usize::try_from(n).map_err(|_| "horizon: min_events is out of range".to_string())?);
    }
    let evaluation = evaluate(saved, &subjects, &spec).map_err(|e| format!("horizon: {e}"))?;
    let value = serde_json::to_value(&evaluation).map_err(|e| format!("horizon: {e}"))?;
    let text = serde_json::to_string_pretty(&value).map_err(|e| format!("horizon: {e}"))?;
    Ok(with_fields(Outcome::new(), &value).blob("evaluation", Blob::new(Media::Text, text.into_bytes())))
}

/// The calibration `inv` asks of `saved`, and the outcome that reports it (its
/// `calibration` blob is the file's content).
fn fit_calibration(saved: &Saved, inv: &Invocation) -> Result<(Calibration, Outcome), String> {
    let validation = subjects_of(inv, "validation")?;
    for s in &validation {
        saved.vocab.check_units(s).map_err(|e| format!("horizon: {e}"))?;
    }
    let horizons = horizons(inv, true)?;
    let kind = match inv.get_str("kind") {
        Some(name) => calibration::Kind::from_name(&name).map_err(|e| format!("horizon: {e}"))?,
        None => calibration::Kind::default(),
    };
    let min_events = match inv.get_i64("min_events") {
        Some(n) => usize::try_from(n).map_err(|_| "horizon: min_events is out of range".to_string())?,
        None => kind.min_events(),
    };
    let digest = saved.weights_digest().map_err(|e| format!("horizon: {e}"))?;
    let calibration = Calibration::fit(saved, digest, &validation, &horizons, kind, min_events)
        .map_err(|e| format!("horizon: {e}"))?;
    let text = calibration.to_json().map_err(|e| format!("horizon: {e}"))?;
    let outcome = Outcome::new()
        .set("kind", json!(kind.name()))
        .set("calibrated", json!(calibration.entries().len()))
        .set(
            "uncalibrated",
            serde_json::to_value(calibration.uncalibrated()).map_err(|e| format!("horizon: {e}"))?,
        )
        .blob("calibration", Blob::new(Media::Text, text.into_bytes()));
    Ok((calibration, outcome))
}

/// `calibrate` on a model a scheduler holds: the calibration is returned and
/// nothing is written, since the served directory is the host's.
pub fn calibrate_loaded(loaded: &Loaded, inv: &Invocation) -> ActionResult {
    match loaded {
        Loaded::Single(saved) => fit_calibration(saved, inv).map(|(_, outcome)| outcome),
        Loaded::Ensemble(_) => Err("horizon: calibrate takes a single model, not an ensemble".into()),
    }
}

/// `calibrate` from the command line: fit, then write `calibration.json` into
/// the model directory, or a calibrated copy of the model into `out`; neither
/// overwrites without `force`.
pub fn calibrate(inv: &Invocation) -> ActionResult {
    let dir = inv.get_str("weights").ok_or_else(|| missing("param 'weights'"))?;
    let dir = Path::new(&dir);
    let out = inv.get_str("out");
    let force = inv.get_bool("force").unwrap_or(false);
    // Refuse before the fit, not after.
    match &out {
        Some(out) if Path::new(out).exists() && !force => {
            return Err(format!("horizon: {out}: already exists (not overwritten without force)"))
        }
        None if dir.join(calibration::FILE).exists() && !force => {
            return Err(format!(
                "horizon: {}: already holds {} (not overwritten without force)",
                dir.display(),
                calibration::FILE
            ))
        }
        _ => {}
    }
    if Ensemble::is_dir(dir) {
        return Err("horizon: calibrate takes a single model, not an ensemble".into());
    }
    let mut saved = Saved::load(dir).map_err(|e| format!("horizon: {e}"))?;
    let (calibration, outcome) = fit_calibration(&saved, inv)?;
    match out {
        Some(out) => {
            saved.calibration = Some(std::sync::Arc::new(calibration));
            saved.save_new(Path::new(&out), force).map_err(|e| format!("horizon: {e}"))?;
        }
        None => {
            let text = calibration.to_json().map_err(|e| format!("horizon: {e}"))?;
            write_file_atomically(&dir.join(calibration::FILE), text.as_bytes()).map_err(|e| format!("horizon: {e}"))?;
        }
    }
    Ok(outcome)
}
