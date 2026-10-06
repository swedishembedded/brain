// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The timeline model behind the generalized [`capability`] interface: what
//! makes `brain horizon predict` and `brain do brain/horizon predict` work,
//! and (through the catalog's resident adapter) the same action on every
//! served surface.
//!
//! One action, `predict`: subjects in `timeline-v1` (one JSON object per
//! line) go in as the `subjects` blob; for each, at every requested time, the
//! probability of surviving every absorbing outcome and each outcome code's
//! cumulative incidence come out as JSON lines. A time past the model's last
//! knot is refused, not extrapolated: the model says nothing there.
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

use crate::saved::{parse_jsonl, Saved};
use crate::survival::Curves;
use crate::timeline::Subject;

/// The model id on the CLI and every served surface.
pub const MODEL: &str = "brain/horizon";
/// The host's saved-model directory on a served surface.
pub const DIR_VAR: &str = "BRAIN_HORIZON_DIR";
/// Times reported when the caller names none (in the dataset's unit).
pub const DEFAULT_TIMES: &str = "5,10";

/// The `predict` action's schema.
pub fn predict_spec() -> ActionSpec {
    ActionSpec::new(
        "predict",
        "outcome probabilities over time for each subject of a timeline-v1 file",
    )
    .param(
        ParamSpec::new(
            "weights",
            ParamType::Str,
            "directory of a saved timeline model (model.safetensors + vocab.json)",
        )
        .required()
        .host_env(DIR_VAR),
    )
    .param(
        ParamSpec::new(
            "times",
            ParamType::Str,
            "comma-separated times after entry, in the data's unit",
        )
        .default(json!(DEFAULT_TIMES)),
    )
    .input(BlobSpec::new("subjects", Media::Text, "timeline-v1: one subject per line").required())
    .output(BlobSpec::new(
        "predictions",
        Media::Text,
        "JSON lines: {subject_id, times, survival, cif: {code: [...]}}",
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

/// One request, parsed and validated: its subjects and the times asked for.
struct Request {
    subjects: Vec<Subject>,
    times: Vec<f64>,
}

fn parse_request(saved: &Saved, inv: &Invocation) -> Result<Request, String> {
    let blob = inv
        .get_blob("subjects")
        .ok_or("horizon: the 'subjects' input blob is required")?;
    let text = std::str::from_utf8(&blob.bytes)
        .map_err(|e| format!("horizon: subjects are not UTF-8: {e}"))?;
    let subjects = parse_jsonl(text).map_err(|e| format!("horizon: subjects {e}"))?;
    let times = times(inv, saved.horizon())?;
    Ok(Request { subjects, times })
}

/// One request's answer: a JSON line per subject.
fn render(saved: &Saved, req: &Request, curves: &[Curves]) -> Outcome {
    let mut out = String::new();
    for (s, c) in req.subjects.iter().zip(curves) {
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
        let line = json!({
            "subject_id": s.subject_id,
            "times": req.times,
            "survival": req.times.iter().map(|&t| c.survival(t)).collect::<Vec<_>>(),
            "cif": cif,
        });
        out.push_str(&line.to_string());
        out.push('\n');
    }
    Outcome::new()
        .set("subjects", json!(req.subjects.len()))
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
    let curves = match saved.predict(&subjects) {
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
            Ok(render(saved, &req, mine))
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
