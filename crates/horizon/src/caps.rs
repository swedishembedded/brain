// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The timeline model behind the generalized [`capability`] interface: what
//! makes `brain horizon predict` and `brain do brain/horizon predict` work,
//! and (through the catalog's resident adapter) the same action on every
//! served surface.
//!
//! One action, `predict`. Its input is ONE of two blobs:
//!
//! - `subjects`: subjects in `timeline-v1` (one JSON object per line); for
//!   each, at every requested time, the probability of surviving every
//!   absorbing outcome and each outcome code's cumulative incidence come out
//!   as JSON lines;
//! - `history`: patient histories in the update format
//!   ([`crate::history`]: one object, an array or one per line); for each, a
//!   [`RiskForecast`](crate::forecast::RiskForecast) (identity, coverage,
//!   curves, risks at the requested times as horizons, support, warnings),
//!   one JSON line per history in the `predictions` blob and the same list
//!   as the `forecasts` output. A forecast does not diagnose or recommend
//!   treatment.
//!
//! A time past the model's last knot is refused, not extrapolated: the model
//! says nothing there. Both kinds of request share one forward pass.
//!
//! Every answer carries `support`: whether the subject is inside what the model
//! was trained on (`supported`: true, false, or null when the model records no
//! support), a continuous `ood_score` (1 at the edge of the support) and typed
//! `warnings`. A subject whose score is above the `max_ood_score` threshold
//! (default 1: any warning) gets `{"risk": "unavailable", "reason":
//! "insufficient support"}` INSTEAD of probabilities. Unknown variables,
//! categories and event codes score 10, so they are refused unless an
//! operator raises the threshold past it on purpose.
//!
//! A model saved with a calibration also answers `cif_calibrated` (the
//! calibrated risk) and `cif_interval` (its Venn-Abers interval) beside the
//! raw `cif`, per code and time; `null` at a time that was not calibrated.
//!
//! The model directory is the one `TimelineModel::save` (SDK) writes. On a
//! served surface it is host configuration (`BRAIN_HORIZON_DIR`), never a
//! caller's parameter.

use std::path::Path;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome,
    ParamSpec, ParamType, Progress, Provider,
};
use serde_json::{json, Value};

use crate::calibration::Calibrated;
use crate::forecast::{ForecastRequest, RiskForecast};
use crate::history::PatientHistory;
use crate::saved::{parse_jsonl, Saved, Scored};
use crate::support::AssessOptions;
use crate::timeline::Subject;

/// The model id on the CLI and every served surface.
pub const MODEL: &str = "brain/horizon";
/// The host's saved-model directory on a served surface.
pub const DIR_VAR: &str = "BRAIN_HORIZON_DIR";
/// The `max_ood_score` threshold when the caller names none.
pub use crate::support::DEFAULT_MAX_OOD_SCORE;
/// Times reported when the caller names none (in the dataset's unit).
pub const DEFAULT_TIMES: &str = "5,10";

/// The `predict` action's schema.
pub fn predict_spec() -> ActionSpec {
    ActionSpec::new(
        "predict",
        "outcome probabilities over time for each subject of a timeline-v1 file, or a structured risk forecast for each patient history",
    )
    .param(
        ParamSpec::new(
            "weights",
            ParamType::Str,
            "directory of a saved timeline model (model.safetensors + vocab.json, and calibration.json when calibrated)",
        )
        .required()
        .host_env(DIR_VAR),
    )
    .param(
        ParamSpec::new(
            "times",
            ParamType::Str,
            "comma-separated times after entry (after as_of, for a history: its forecast horizons, each above 0), in the data's unit",
        )
        .default(json!(DEFAULT_TIMES)),
    )
    .param(
        ParamSpec::new(
            "max_ood_score",
            ParamType::Float,
            "subjects whose out-of-distribution score is above this get `risk: unavailable` instead of probabilities (1 = the edge of the training support; unknown variables, categories and event codes score 10)",
        )
        .default(json!(DEFAULT_MAX_OOD_SCORE)),
    )
    .input(BlobSpec::new(
        "subjects",
        Media::Text,
        "timeline-v1: one subject per line (give this or history)",
    ))
    .input(BlobSpec::new(
        "history",
        Media::Text,
        "patient histories: JSON {as_of, birth?, static?, events: [{time, code, value?, unit?}]}, one object, an array or one per line (give this or subjects)",
    ))
    .output(BlobSpec::new(
        "predictions",
        Media::Text,
        "JSON lines. For subjects: {subject_id, times, survival, cif: {code: [...]}, support: {supported, ood_score, warnings}} (+ cif_calibrated, cif_interval of a calibrated model); an unsupported subject: {subject_id, risk: \"unavailable\", reason, support}. For a history: a risk forecast {subject_id, as_of, model, coverage, risk, curves, horizons, support, input_warnings, disclaimer}",
    ))
}

/// The full, static manifest: safe to build with no model loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "continuous-time timeline model: competing-outcome probabilities at any time from irregular records",
        vec![predict_spec()],
    )
}

/// The manifest a served surface advertises: the model directory is the
/// host's, so it is not a parameter there.
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// The requested times, each finite, non-negative and within the model.
fn times(inv: &Invocation, horizon: f64) -> Result<Vec<f64>, String> {
    let text = inv
        .get_str("times")
        .unwrap_or_else(|| DEFAULT_TIMES.to_string());
    let parsed: Vec<f64> = text
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f64>()
                .map_err(|e| format!("horizon: time '{t}': {e}"))
        })
        .collect::<Result<_, _>>()?;
    if parsed.is_empty() {
        return Err("horizon: no times requested".into());
    }
    if let Some(t) = parsed
        .iter()
        .find(|t| !(t.is_finite() && **t >= 0.0 && **t <= horizon))
    {
        return Err(format!(
            "horizon: time {t} is outside the model's range [0, {horizon}]"
        ));
    }
    Ok(parsed)
}

/// One request, parsed and validated: its subjects, the times asked for and
/// the support threshold above which a subject gets no probability. A request
/// made of patient histories keeps them beside the subjects they became.
struct Request {
    subjects: Vec<Subject>,
    histories: Option<Vec<PatientHistory>>,
    times: Vec<f64>,
    max_ood_score: f64,
}

/// The text of the input blob `name`, if it was given.
fn blob_text<'a>(inv: &'a Invocation, name: &str) -> Result<Option<&'a str>, String> {
    inv.get_blob(name)
        .map(|b| std::str::from_utf8(&b.bytes).map_err(|e| format!("horizon: {name} is not UTF-8: {e}")))
        .transpose()
}

fn parse_request(saved: &Saved, inv: &Invocation) -> Result<Request, String> {
    let (subjects, histories) = match (blob_text(inv, "subjects")?, blob_text(inv, "history")?) {
        (Some(_), Some(_)) => {
            return Err("horizon: give either the 'subjects' or the 'history' input, not both".into())
        }
        (Some(text), None) => {
            let subjects = parse_jsonl(text).map_err(|e| format!("horizon: subjects {e}"))?;
            // A unit mismatch fails THIS request, not the batch it is scored in.
            for s in &subjects {
                saved.vocab.check_units(s).map_err(|e| format!("horizon: {e}"))?;
            }
            (subjects, None)
        }
        (None, Some(text)) => {
            let histories = PatientHistory::parse_all(text).map_err(|e| format!("horizon: {e}"))?;
            let subjects = histories
                .iter()
                .map(|h| h.to_subject(&saved.vocab).map_err(|e| format!("horizon: {e}")))
                .collect::<Result<Vec<_>, _>>()?;
            (subjects, Some(histories))
        }
        (None, None) => {
            return Err("horizon: the 'subjects' or the 'history' input blob is required".into())
        }
    };
    let times = times(inv, saved.horizon())?;
    let max_ood_score = match inv.get_f64("max_ood_score") {
        None => DEFAULT_MAX_OOD_SCORE,
        Some(m) if m > 0.0 => m,
        Some(m) => return Err(format!("horizon: max_ood_score must be positive, got {m}")),
    };
    if histories.is_some() {
        ForecastRequest::new(times.iter().copied())
            .max_ood_score(max_ood_score)
            .validate(saved.horizon())
            .map_err(|e| format!("horizon: {e}"))?;
    }
    Ok(Request { subjects, histories, times, max_ood_score })
}

/// One request of patient histories: a forecast per history, as JSON lines
/// and as the `forecasts` output.
fn render_forecasts(saved: &Saved, histories: &[PatientHistory], req: &Request, scored: &[Scored]) -> ActionResult {
    let identity = saved.identity().map_err(|e| format!("horizon: {e}"))?;
    let forecast_request = ForecastRequest::new(req.times.iter().copied()).max_ood_score(req.max_ood_score);
    let forecasts: Vec<RiskForecast> = histories
        .iter()
        .zip(scored)
        .map(|(h, s)| RiskForecast::from_scored(saved, &identity, h, s, &forecast_request))
        .collect();
    let mut out = String::new();
    for f in &forecasts {
        out.push_str(&serde_json::to_string(f).map_err(|e| format!("horizon: {e}"))?);
        out.push('\n');
    }
    Ok(Outcome::new()
        .set("subjects", json!(forecasts.len()))
        .set("abstained", json!(forecasts.iter().filter(|f| !f.is_available()).count()))
        .set("forecasts", serde_json::to_value(&forecasts).map_err(|e| format!("horizon: {e}"))?)
        .blob("predictions", Blob::new(Media::Text, out.into_bytes())))
}

/// One request's answer: a JSON line per subject.
fn render(saved: &Saved, req: &Request, scored: &[Scored]) -> ActionResult {
    if let Some(histories) = &req.histories {
        return render_forecasts(saved, histories, req, scored);
    }
    Ok(render_subjects(saved, req, scored))
}

fn render_subjects(saved: &Saved, req: &Request, scored: &[Scored]) -> Outcome {
    let (mut out, mut abstained) = (String::new(), 0);
    for (s, Scored { curves: c, assessment }) in req.subjects.iter().zip(scored) {
        let mut support = json!({
            "supported": assessment.supported,
            "ood_score": assessment.ood_score,
            "warnings": assessment.warnings,
        });
        if !assessment.advisories.is_empty() {
            support["advisories"] = json!(assessment.advisories);
        }
        if assessment.abstains(req.max_ood_score) {
            abstained += 1;
            let line = json!({
                "subject_id": s.subject_id,
                "risk": "unavailable",
                "reason": "insufficient support",
                "support": support,
            });
            out.push_str(&line.to_string());
            out.push('\n');
            continue;
        }
        let cif: serde_json::Map<String, Value> = saved
            .vocab
            .codes
            .iter()
            .enumerate()
            .map(|(k, code)| {
                (
                    code.clone(),
                    json!(req.times.iter().map(|&t| c.cif(k, t)).collect::<Vec<_>>()),
                )
            })
            .collect();
        let mut line = json!({
            "subject_id": s.subject_id,
            "times": req.times,
            "survival": req.times.iter().map(|&t| c.survival(t)).collect::<Vec<_>>(),
            "cif": cif,
            "support": support,
        });
        if let Some(cal) = &saved.calibration {
            // `null` where the horizon is not calibrated: absent, never 0.
            let per_code = |pick: &dyn Fn(Calibrated) -> Value| -> serde_json::Map<String, Value> {
                saved
                    .vocab
                    .codes
                    .iter()
                    .enumerate()
                    .map(|(k, code)| {
                        let at = req.times.iter().map(|&t| {
                            cal.apply(code, t, c.cif(k, t)).map_or(Value::Null, pick)
                        });
                        (code.clone(), Value::Array(at.collect()))
                    })
                    .collect()
            };
            line["cif_calibrated"] = Value::Object(per_code(&|c| json!(c.risk)));
            line["cif_interval"] = Value::Object(per_code(&|c| json!([c.lower, c.upper])));
        }
        out.push_str(&line.to_string());
        out.push('\n');
    }
    Outcome::new()
        .set("subjects", json!(req.subjects.len()))
        .set("abstained", json!(abstained))
        .blob("predictions", Blob::new(Media::Text, out.into_bytes()))
}

/// `predict` on a loaded model: the work every surface shares.
pub fn predict(saved: &Saved, inv: &Invocation) -> ActionResult {
    predict_batch(saved, std::slice::from_ref(inv))
        .pop()
        .ok_or_else(|| "horizon: no result".to_string())?
}

/// Several `predict` requests through ONE forward pass over all their
/// subjects (in device batches of the model's batch size), the curves split
/// back per request, in order. A request that cannot be answered - a missing
/// or malformed file, a time outside the model - fails alone: the others get
/// their answers. A subject's curves never depend on the others in the pass
/// (padding and neighbours are masked), so each answer equals the request run
/// by itself.
pub fn predict_batch(saved: &Saved, invs: &[Invocation]) -> Vec<ActionResult> {
    let parsed: Vec<Result<Request, String>> =
        invs.iter().map(|inv| parse_request(saved, inv)).collect();
    let subjects: Vec<Subject> = parsed
        .iter()
        .flatten()
        .flat_map(|r| r.subjects.iter().cloned())
        .collect();
    let curves = match saved.score(&subjects, &AssessOptions::default()) {
        Ok(c) => c,
        // Every subject was validated at parse time, so this is a device
        // failure: it is every valid request's failure, reported as such.
        Err(e) => {
            return parsed
                .into_iter()
                .map(|r| r.and_then(|_| Err(format!("horizon: {e}"))))
                .collect()
        }
    };
    let mut rest = curves.as_slice();
    parsed
        .into_iter()
        .map(|req| {
            let req = req?;
            let (mine, tail) = rest.split_at(req.subjects.len());
            rest = tail;
            render(saved, &req, mine)
        })
        .collect()
}

/// The model loaded on first use, kept with the directory it came from.
type Hot = Arc<Mutex<Option<(String, Saved)>>>;

/// The executable timeline model behind the manifest. Construction is free:
/// the model loads on the first run and stays resident until another
/// directory is asked for.
#[derive(Default)]
pub struct HorizonProvider {
    hot: Hot,
}

impl HorizonProvider {
    pub fn new() -> HorizonProvider {
        HorizonProvider::default()
    }
}

impl Provider for HorizonProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        (name == "predict").then(|| {
            Arc::new(PredictAction {
                hot: self.hot.clone(),
            }) as Arc<dyn Action>
        })
    }
}

struct PredictAction {
    hot: Hot,
}

impl Action for PredictAction {
    fn spec(&self) -> ActionSpec {
        predict_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv
            .get_str("weights")
            .ok_or("horizon: missing required param 'weights'")?;
        let mut guard = self
            .hot
            .lock()
            .map_err(|_| "horizon: model lock poisoned")?;
        if !matches!(&*guard, Some((d, _)) if *d == dir) {
            *guard = None;
            *guard = Some((
                dir.clone(),
                Saved::load(Path::new(&dir)).map_err(|e| format!("horizon: {e}"))?,
            ));
        }
        progress(Progress::step(1, 1, "predict"));
        let (_, saved) = guard.as_ref().ok_or("horizon: no model loaded")?;
        predict(saved, inv)
    }
}
