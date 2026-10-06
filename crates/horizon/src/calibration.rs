// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibration persisted beside the weights: for each outcome code and each
//! horizon asked for, a calibrator fitted on VALIDATION subjects (never the
//! test set) under inverse-probability-of-censoring weights, the censoring
//! distribution estimated on those same validation subjects.
//!
//! Two kinds of calibrator ([`Kind`]), chosen by measurement on a population
//! whose true risk is known:
//!
//! - [`Kind::Logistic`] (the default): the logistic recalibration
//!   `logit P = a + b logit F` of [`survival::recalibration`], with the slope
//!   `b` estimated only where the validation subjects show it is not one and
//!   the intercept alone otherwise. Two parameters at most, so its error falls
//!   steadily as events accumulate and a model that was already calibrated
//!   comes out essentially unchanged. It corrects a shift, over- and under-
//!   confidence; it has no interval.
//! - [`Kind::VennAbers`]: an isotonic fit per (code, horizon) with its
//!   Venn-Abers interval. A free-form step function chases noise: with a few
//!   hundred events its output is more spread than the truth and an already
//!   calibrated model comes out worse, so it asks for many more events.
//!
//! Swedish Embedded AB implements risk models whose probabilities hold up
//! against outcomes observed years later, for its clients. If your team needs
//! expertise in calibrating survival predictions under censoring and
//! competing risks, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A calibrator is only as good as the events behind it: a horizon whose
//! validation subjects hold fewer events of the code than the minimum of the
//! kind ([`Kind::min_events`]), or fewer than that many still event-free at
//! the horizon, is NOT calibrated and is listed in
//! [`Calibration::uncalibrated`]; asking for its calibrated risk gives
//! `None`, never a number. The artifact records the digest of the
//! weights it was fitted for and [`crate::saved::Saved::load`] refuses it
//! beside any other weights: a calibration of a different model is a wrong
//! probability that looks right.

use serde::{Deserialize, Serialize};
use survival::estimate::censoring;
use survival::recalibration::{Recalibration, Slope};
use survival::venn_abers::{merged, State, VennAbers};
use survival::Obs;

use crate::saved::Saved;
use crate::timeline::Subject;

/// The calibration file inside a saved model's directory.
pub const FILE: &str = "calibration.json";
/// The fewest events a metric is judged on by default (see
/// [`crate::evaluation::EvaluationSpec`]); a calibrator needs more
/// ([`Kind::min_events`]).
pub const MIN_EVENTS: usize = 30;
/// The slope of a [`Kind::Logistic`] calibrator is kept only if it differs
/// from one by more than this many standard errors.
pub const SLOPE_STANDARD_ERRORS: f64 = 2.0;
/// The format [`Calibration::to_json`] writes. Version 1 (Venn-Abers only,
/// no `kind`) is still read.
const VERSION: u32 = 2;

/// How a calibration maps a raw risk to a calibrated one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// Logistic recalibration, intercept always and slope on evidence; no
    /// interval ([`Calibrated::interval`] is `None`).
    #[default]
    Logistic,
    /// Isotonic regression with its Venn-Abers interval.
    VennAbers,
}

impl Kind {
    /// The fewest validation events of a code by a horizon (and as many
    /// still event-free) at which the kind is fitted unless the caller asks
    /// otherwise. Below it a fitted calibrator adds more noise than the
    /// correction it can find: the slope of a logistic fit is poorly
    /// determined by fewer events, and an isotonic fit flattens the slope of
    /// even a perfect model until it has several hundred.
    pub const fn min_events(self) -> usize {
        match self {
            Kind::Logistic => 100,
            Kind::VennAbers => 500,
        }
    }

    /// The name in `calibration.json` and in requests.
    pub const fn name(self) -> &'static str {
        match self {
            Kind::Logistic => "logistic",
            Kind::VennAbers => "venn_abers",
        }
    }

    /// The kind called `name`.
    pub fn from_name(name: &str) -> Result<Kind, String> {
        match name {
            "logistic" => Ok(Kind::Logistic),
            "venn_abers" => Ok(Kind::VennAbers),
            other => Err(format!("calibration: unknown kind {other:?} (logistic or venn_abers)")),
        }
    }
}

fn legacy_kind() -> Kind {
    Kind::VennAbers
}

/// Each subject's observed outcome among `codes`, for the metrics in
/// [`survival`]: the time from entry to its first event of any of the
/// codes inside that code's observation window (its cause is the code's
/// index in `codes`), else censored at the end of the window of `codes[0]`.
/// Codes listed together compete: evaluate a cause of death with the other
/// causes listed after it, a first diagnosis with the causes of death after
/// it.
pub fn observed(subjects: &[Subject], codes: &[&str]) -> Vec<Obs> {
    subjects
        .iter()
        .map(|s| {
            let mut first: Option<(f64, usize)> = None;
            for e in s.events.iter().filter(|e| e.t > s.entry) {
                let Some(k) = codes.iter().position(|c| *c == e.code) else {
                    continue;
                };
                let Some(w) = s.window(&e.code) else { continue };
                // Inside the window, with the same rounding slack the encoder allows.
                if e.t <= w.to + 1e-9 * w.to.abs().max(1.0) && first.is_none_or(|(t, _)| e.t < t) {
                    first = Some((e.t, k));
                }
            }
            let weight = s.weight;
            match first {
                Some((t, k)) => Obs {
                    time: t - s.entry,
                    cause: Some(k),
                    weight,
                },
                None => {
                    let end = codes
                        .first()
                        .and_then(|c| s.window(c))
                        .map_or(s.entry, |w| w.to);
                    Obs {
                        time: end - s.entry,
                        cause: None,
                        weight,
                    }
                }
            }
        })
        .collect()
}

/// One calibrated (code, horizon).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    /// The outcome code.
    pub code: String,
    /// The horizon, after entry, in the dataset's unit.
    pub horizon: f64,
    /// Validation subjects whose outcome by the horizon was known (not
    /// censored before it).
    pub labelled: usize,
    /// Of them, the subjects with an event of the code by the horizon.
    pub events: usize,
    // Venn-Abers data (empty for a logistic entry).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    scores: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    weight: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    event_weight: Vec<f64>,
    // Logistic recalibration (absent for a Venn-Abers entry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    intercept: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    slope: Option<f64>,
    #[serde(skip)]
    calibrator: Option<Fitted>,
}

/// A calibrator rebuilt from its stored data.
#[derive(Clone, Debug)]
enum Fitted {
    VennAbers(VennAbers),
    Logistic(Recalibration),
}

/// Why a horizon was not calibrated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// Fewer events (or event-free subjects) than the minimum.
    #[default]
    TooFewEvents,
    /// Enough events, but the data gave no usable map (no spread in the
    /// predictions, a separable fit, a slope that is not positive).
    NoFit,
}

/// A horizon the validation data could not support.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gap {
    /// The outcome code.
    pub code: String,
    /// The horizon.
    pub horizon: f64,
    /// Validation events of the code by the horizon.
    pub events: usize,
    /// Validation subjects still event-free at the horizon.
    pub event_free: usize,
    /// Why it was left out.
    #[serde(default)]
    pub reason: Reason,
}

/// A calibrated risk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibrated {
    /// The calibrated probability: the logistic recalibration of the raw
    /// risk, or the Venn-Abers merged probability `p1 / (1 - p0 + p1)`.
    pub risk: f64,
    /// The Venn-Abers interval `(p0, p1)` the risk is taken from (one end is
    /// calibrated whatever the model; the width is how little calibration
    /// data stands behind the score). `None` for a logistic calibration,
    /// which has no interval.
    pub interval: Option<(f64, f64)>,
}

/// The persisted calibration of one model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    version: u32,
    /// How the entries calibrate (a version 1 file is Venn-Abers).
    #[serde(default = "legacy_kind")]
    pub kind: Kind,
    /// SHA-256 (hex) of the weights file these calibrators were fitted for.
    pub weights_sha256: String,
    /// Validation subjects the calibrators were fitted on.
    pub validation_subjects: usize,
    /// The event minimum that applied.
    pub min_events: usize,
    entries: Vec<Entry>,
    uncalibrated: Vec<Gap>,
}

impl Calibration {
    /// Fit calibrators of `kind` on `validation` for every outcome code at
    /// every horizon in `horizons` (each in `(0, last knot]`). `weights_sha256`
    /// is the digest of the weights of `saved` (see [`Saved::weights_digest`]).
    /// A horizon with fewer than `min_events` events of the code (or as many
    /// still event-free) is left out ([`Kind::min_events`] is the default).
    pub fn fit(
        saved: &Saved,
        weights_sha256: String,
        validation: &[Subject],
        horizons: &[f64],
        kind: Kind,
        min_events: usize,
    ) -> Result<Calibration, String> {
        if validation.is_empty() {
            return Err("calibration: validation subjects are required".into());
        }
        if horizons.is_empty() {
            return Err("calibration: at least one horizon is required".into());
        }
        let last = saved.horizon();
        if let Some(h) = horizons
            .iter()
            .find(|h| !(h.is_finite() && **h > 0.0 && **h <= last))
        {
            return Err(format!(
                "calibration: horizon {h} is outside the model's range (0, {last}]"
            ));
        }
        let mut horizons = horizons.to_vec();
        horizons.sort_by(f64::total_cmp);
        horizons.dedup();
        let curves = saved.predict(validation)?;
        let codes = &saved.vocab.codes;
        let (mut entries, mut uncalibrated) = (Vec::new(), Vec::new());
        for (k, code) in codes.iter().enumerate() {
            // The code, then the codes it competes with (see `Curves::cif`).
            let mut list: Vec<&str> = vec![code.as_str()];
            list.extend(
                codes
                    .iter()
                    .enumerate()
                    .filter(|(j, c)| *j != k && saved.vocab.absorbing.contains(c))
                    .map(|(_, c)| c.as_str()),
            );
            let obs = observed(validation, &list);
            // The censoring distribution of THIS data, not of the training set.
            let g = censoring(&obs);
            for &t in &horizons {
                let events = obs
                    .iter()
                    .filter(|o| o.time <= t && o.cause == Some(0))
                    .count();
                let event_free = obs.iter().filter(|o| o.time > t).count();
                let gap = |reason| Gap { code: code.clone(), horizon: t, events, event_free, reason };
                if events < min_events || event_free < min_events {
                    uncalibrated.push(gap(Reason::TooFewEvents));
                    continue;
                }
                let scores: Vec<f64> = curves.iter().map(|c| c.cif(k, t)).collect();
                let entry = Entry {
                    code: code.clone(),
                    horizon: t,
                    labelled: 0,
                    events,
                    scores: Vec::new(),
                    weight: Vec::new(),
                    event_weight: Vec::new(),
                    intercept: None,
                    slope: None,
                    calibrator: None,
                };
                match kind {
                    Kind::VennAbers => {
                        let va = VennAbers::at_horizon(&scores, &obs, 0, t, &g);
                        let State { scores, weight, events: event_weight, count } = va.state();
                        entries.push(Entry {
                            labelled: count,
                            scores,
                            weight,
                            event_weight,
                            calibrator: Some(Fitted::VennAbers(va)),
                            ..entry
                        });
                    }
                    Kind::Logistic => {
                        let slope = Slope::Evidence(SLOPE_STANDARD_ERRORS);
                        let Some(r) = Recalibration::at_horizon(&scores, &obs, 0, t, &g, slope) else {
                            uncalibrated.push(gap(Reason::NoFit));
                            continue;
                        };
                        entries.push(Entry {
                            labelled: obs.iter().filter(|o| o.time > t || o.cause.is_some()).count(),
                            intercept: Some(r.intercept),
                            slope: Some(r.slope),
                            calibrator: Some(Fitted::Logistic(r)),
                            ..entry
                        });
                    }
                }
            }
        }
        Ok(Calibration {
            version: VERSION,
            kind,
            weights_sha256,
            validation_subjects: validation.len(),
            min_events,
            entries,
            uncalibrated,
        })
    }

    /// The calibrated (code, horizon) pairs.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The pairs the validation data could not support.
    pub fn uncalibrated(&self) -> &[Gap] {
        &self.uncalibrated
    }

    /// The entry for `code` at `horizon` (matched to rounding), if calibrated.
    pub fn entry(&self, code: &str, horizon: f64) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|e| e.code == code && (e.horizon - horizon).abs() <= 1e-9 * horizon.abs().max(1.0))
    }

    /// The calibrated risk for a raw model risk `raw` of `code` by `horizon`;
    /// `None` where that horizon is not calibrated.
    pub fn apply(&self, code: &str, horizon: f64, raw: f64) -> Option<Calibrated> {
        match self.entry(code, horizon)?.calibrator.as_ref()? {
            Fitted::VennAbers(va) => {
                // A new subject stands for an average calibration subject, not
                // for its own sampling weight (see `VennAbers::mean_weight`).
                let interval = va.interval(raw, va.mean_weight()?);
                Some(Calibrated { risk: merged(interval), interval: Some(interval) })
            }
            Fitted::Logistic(r) => Some(Calibrated { risk: r.apply(raw), interval: None }),
        }
    }

    /// Serialise for [`FILE`].
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("calibration: {e}"))
    }

    /// Parse [`FILE`]'s text, rebuilding and checking every calibrator.
    pub fn from_json(text: &str) -> Result<Calibration, String> {
        let mut c: Calibration =
            serde_json::from_str(text).map_err(|e| format!("calibration: {e}"))?;
        if !(1..=VERSION).contains(&c.version) {
            return Err(format!(
                "calibration: format version {} is not supported (this build reads 1 to {VERSION})",
                c.version
            ));
        }
        if c.version == 1 && c.kind != Kind::VennAbers {
            return Err("calibration: a version 1 file holds Venn-Abers calibrators only".into());
        }
        let kind = c.kind;
        for e in &mut c.entries {
            let at = |m: String| format!("calibration: {} at {}: {m}", e.code, e.horizon);
            e.calibrator = Some(match kind {
                Kind::VennAbers => {
                    let state = State {
                        scores: e.scores.clone(),
                        weight: e.weight.clone(),
                        events: e.event_weight.clone(),
                        count: e.labelled,
                    };
                    Fitted::VennAbers(VennAbers::from_state(state).map_err(at)?)
                }
                Kind::Logistic => {
                    let (Some(intercept), Some(slope)) = (e.intercept, e.slope) else {
                        return Err(at("a logistic entry needs an intercept and a slope".into()));
                    };
                    // Stored data is untrusted: a non-positive slope would
                    // reverse the order of the risks.
                    if !(intercept.is_finite() && slope.is_finite() && slope > 0.0) {
                        return Err(at("the intercept and slope must be finite and the slope positive".into()));
                    }
                    Fitted::Logistic(Recalibration { intercept, slope })
                }
            });
        }
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_outcomes_compete_and_censor_at_the_window() {
        let line = |events: &str| {
            Subject::from_json_line(&format!(
                r#"{{"subject_id":"a","weight":2,"source":"s","entry":50,"calendar_at_entry":2000,"events":[{events}],"at_risk":[{{"code":"*","from":50,"to":60}}]}}"#
            ))
            .unwrap()
        };
        let s = [
            line(r#"{"t":45,"code":"x"},{"t":53,"code":"y"},{"t":55,"code":"x"}"#),
            line(""),
            line(r#"{"t":61,"code":"x"}"#),
        ];
        let o = observed(&s, &["x", "y"]);
        assert_eq!(
            o[0],
            Obs {
                time: 3.0,
                cause: Some(1),
                weight: 2.0
            },
            "y came first; history before entry ignored"
        );
        assert_eq!(
            o[1],
            Obs {
                time: 10.0,
                cause: None,
                weight: 2.0
            }
        );
        assert_eq!(o[2].cause, None, "an event after the window is not observed");
    }

    /// A version 1 file, as written before kinds existed: Venn-Abers, no `kind`.
    const VERSION_1: &str = r#"{"version":1,"weights_sha256":"w","validation_subjects":20,"min_events":30,
        "entries":[{"code":"x","horizon":5.0,"labelled":20,"events":5,
            "scores":[0.1,0.3],"weight":[10.0,10.0],"event_weight":[1.0,4.0]}],
        "uncalibrated":[{"code":"y","horizon":5.0,"events":2,"event_free":18}]}"#;

    #[test]
    fn a_version_1_file_loads_as_venn_abers_exactly_as_before() {
        let c = Calibration::from_json(VERSION_1).unwrap();
        assert_eq!(c.kind, Kind::VennAbers);
        assert_eq!(c.uncalibrated()[0].reason, Reason::TooFewEvents);
        let va = VennAbers::from_state(State {
            scores: vec![0.1, 0.3],
            weight: vec![10.0, 10.0],
            events: vec![1.0, 4.0],
            count: 20,
        })
        .unwrap();
        for raw in [0.0, 0.1, 0.2, 0.5] {
            let interval = va.interval(raw, va.mean_weight().unwrap());
            let got = c.apply("x", 5.0, raw).unwrap();
            assert_eq!(got, Calibrated { risk: merged(interval), interval: Some(interval) });
        }
        assert!(c.apply("x", 10.0, 0.2).is_none() && c.apply("y", 5.0, 0.2).is_none());
        // Written again it is a version 2 file that reads back the same.
        let again = Calibration::from_json(&c.to_json().unwrap()).unwrap();
        assert_eq!(again.kind, Kind::VennAbers);
        assert_eq!(again.apply("x", 5.0, 0.2), c.apply("x", 5.0, 0.2));
    }

    fn logistic_file(entry: &str) -> String {
        format!(
            r#"{{"version":2,"kind":"logistic","weights_sha256":"w","validation_subjects":500,"min_events":100,
            "entries":[{{"code":"x","horizon":5.0,"labelled":400,"events":120{entry}}}],"uncalibrated":[]}}"#
        )
    }

    #[test]
    fn a_logistic_entry_maps_the_log_odds_and_has_no_interval() {
        let c = Calibration::from_json(&logistic_file(r#","intercept":-0.5,"slope":0.8"#)).unwrap();
        assert_eq!(c.kind, Kind::Logistic);
        // logit P = -0.5 + 0.8 logit(0.2); logit(0.2) = ln(0.25).
        let want = 1.0 / (1.0 + (-(-0.5 + 0.8 * 0.25f64.ln())).exp());
        let got = c.apply("x", 5.0, 0.2).unwrap();
        assert!((got.risk - want).abs() < 1e-12 && got.interval.is_none(), "{got:?}");
        let back = Calibration::from_json(&c.to_json().unwrap()).unwrap();
        assert_eq!(back.apply("x", 5.0, 0.2), c.apply("x", 5.0, 0.2));
        // The order of the risks is kept.
        assert!(c.apply("x", 5.0, 0.1).unwrap().risk < c.apply("x", 5.0, 0.3).unwrap().risk);
    }

    #[test]
    fn a_stored_calibration_that_would_misreport_is_refused() {
        for (entry, why) in [
            ("", "an intercept and a slope"),
            (r#","intercept":0.1"#, "an intercept and a slope"),
            (r#","intercept":0.1,"slope":-1.0"#, "slope positive"),
            (r#","intercept":0.1,"slope":0.0"#, "slope positive"),
        ] {
            let err = Calibration::from_json(&logistic_file(entry)).unwrap_err();
            assert!(err.contains(why), "{entry}: {err}");
        }
        let v3 = logistic_file(r#","intercept":0.1,"slope":1.0"#).replace(r#""version":2"#, r#""version":3"#);
        assert!(Calibration::from_json(&v3).unwrap_err().contains("not supported"));
        let v1_logistic = logistic_file(r#","intercept":0.1,"slope":1.0"#).replace(r#""version":2"#, r#""version":1"#);
        assert!(Calibration::from_json(&v1_logistic).unwrap_err().contains("version 1"));
        let unknown = logistic_file("").replace("logistic", "platt");
        assert!(Calibration::from_json(&unknown).is_err());
        assert!(Kind::from_name("platt").is_err());
        assert_eq!(Kind::from_name("logistic"), Ok(Kind::Logistic));
    }

    #[test]
    fn each_kind_asks_for_the_events_it_needs() {
        assert_eq!(Kind::default(), Kind::Logistic);
        assert!(Kind::Logistic.min_events() >= MIN_EVENTS && Kind::VennAbers.min_events() > Kind::Logistic.min_events());
    }
}
