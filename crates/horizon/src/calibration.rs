// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibration persisted beside the weights: for each outcome code and each
//! horizon asked for, a Venn-Abers calibrator fitted on VALIDATION subjects
//! (never the test set) under inverse-probability-of-censoring weights, the
//! censoring distribution estimated on those same validation subjects.
//!
//! Swedish Embedded AB implements risk models whose probabilities hold up
//! against outcomes observed years later, for its clients. If your team needs
//! expertise in calibrating survival predictions under censoring and
//! competing risks, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A calibrator is only as good as the events behind it: a horizon whose
//! validation subjects hold fewer than [`MIN_EVENTS`] events of the code (or
//! fewer than that many still event-free at the horizon) is NOT calibrated
//! and is listed in [`Calibration::uncalibrated`]; asking for its calibrated
//! risk gives `None`, never a number. The artifact records the digest of the
//! weights it was fitted for and [`crate::saved::Saved::load`] refuses it
//! beside any other weights: a calibration of a different model is a wrong
//! probability that looks right.

use serde::{Deserialize, Serialize};
use survival::estimate::censoring;
use survival::venn_abers::{merged, State, VennAbers};
use survival::Obs;

use crate::saved::Saved;
use crate::timeline::Subject;

/// The calibration file inside a saved model's directory.
pub const FILE: &str = "calibration.json";
/// The fewest validation events (and the fewest still event-free at the
/// horizon) a horizon needs to be calibrated. Below it the isotonic fit is a
/// handful of steps whose level is mostly noise.
pub const MIN_EVENTS: usize = 30;
const VERSION: u32 = 1;

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
    scores: Vec<f64>,
    weight: Vec<f64>,
    event_weight: Vec<f64>,
    #[serde(skip)]
    calibrator: Option<VennAbers>,
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
}

/// A calibrated risk: the single probability and the Venn-Abers interval it
/// is taken from (one of the two ends is calibrated whatever the model; the
/// width is how little calibration data stands behind the score).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Calibrated {
    /// The merged probability `p1 / (1 - p0 + p1)`.
    pub risk: f64,
    /// The interval's lower end.
    pub lower: f64,
    /// The interval's upper end.
    pub upper: f64,
}

/// The persisted calibration of one model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    version: u32,
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
    /// Fit on `validation` for every outcome code at every horizon in
    /// `horizons` (each in `(0, last knot]`). `weights_sha256` is the digest
    /// of the weights of `saved` (see [`Saved::weights_digest`]).
    pub fn fit(
        saved: &Saved,
        weights_sha256: String,
        validation: &[Subject],
        horizons: &[f64],
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
                if events < min_events || event_free < min_events {
                    uncalibrated.push(Gap {
                        code: code.clone(),
                        horizon: t,
                        events,
                        event_free,
                    });
                    continue;
                }
                let scores: Vec<f64> = curves.iter().map(|c| c.cif(k, t)).collect();
                let va = VennAbers::at_horizon(&scores, &obs, 0, t, &g);
                let State { scores, weight, events: event_weight, count } = va.state();
                entries.push(Entry {
                    code: code.clone(),
                    horizon: t,
                    labelled: count,
                    events,
                    scores,
                    weight,
                    event_weight,
                    calibrator: Some(va),
                });
            }
        }
        Ok(Calibration {
            version: VERSION,
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
        let va = self.entry(code, horizon)?.calibrator.as_ref()?;
        // A new subject stands for an average calibration subject, not for
        // its own sampling weight (see `VennAbers::mean_weight`).
        let interval = va.interval(raw, va.mean_weight()?);
        Some(Calibrated {
            risk: merged(interval),
            lower: interval.0,
            upper: interval.1,
        })
    }

    /// Serialise for [`FILE`].
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("calibration: {e}"))
    }

    /// Parse [`FILE`]'s text, rebuilding and checking every calibrator.
    pub fn from_json(text: &str) -> Result<Calibration, String> {
        let mut c: Calibration =
            serde_json::from_str(text).map_err(|e| format!("calibration: {e}"))?;
        if c.version != VERSION {
            return Err(format!(
                "calibration: format version {} is not supported (this build reads {VERSION})",
                c.version
            ));
        }
        for e in &mut c.entries {
            let state = State {
                scores: e.scores.clone(),
                weight: e.weight.clone(),
                events: e.event_weight.clone(),
                count: e.labelled,
            };
            e.calibrator = Some(
                VennAbers::from_state(state)
                    .map_err(|m| format!("calibration: {} at {}: {m}", e.code, e.horizon))?,
            );
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
}
