// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `timeline-v1`: one subject's irregular history, one JSON object per line.
//!
//! Every time is a real number in ONE unit the caller declares for the whole
//! dataset (years, for the first application), on the subject's own clock -
//! for a person, attained age. `entry` is the prediction time ("now" for this
//! record): what is known at or before it is input, what happens after it is
//! outcome. `calendar_at_entry` places the subject's clock on the calendar,
//! so a period effect (a population whose mortality falls over the decades)
//! is separable from ageing.
//!
//! The format is validated at entry ([`Subject::validate`]): an unknown field
//! is refused by `deny_unknown_fields`, and a record whose times contradict
//! each other is refused with the field that does.

use serde::{Deserialize, Serialize};

/// One measured value. A number, a category, or a number known only to lie
/// below (or above) a detection limit - which is information about the value,
/// not a missing one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// An exact numeric value, in the variable's canonical unit.
    Number(f64),
    /// The true value is at or below this limit.
    Below {
        /// The detection limit, in the variable's canonical unit.
        below: f64,
    },
    /// The true value is at or above this limit.
    Above {
        /// The detection limit, in the variable's canonical unit.
        above: f64,
    },
    /// A categorical level.
    Category(String),
}

/// A measurement of one variable at one time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// When it was measured (or, for a recalled quantity, when it held).
    pub t: f64,
    /// Variable name, as the dataset's vocabulary declares it.
    pub var: String,
    /// The value.
    pub value: Value,
}

/// An event of one code at one time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    /// When it happened.
    pub t: f64,
    /// Event code.
    pub code: String,
}

/// The window in which an event of `code` would have been OBSERVED had it
/// happened: entry into observation (left truncation) to exit (censoring, or
/// the event itself). `code = "*"` covers every outcome code the model
/// predicts that has no window of its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtRisk {
    /// Event code, or `*`.
    pub code: String,
    /// Start of observation, on the subject's clock.
    pub from: f64,
    /// End of observation, on the subject's clock.
    pub to: f64,
}

/// One subject.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subject {
    /// Unique within the dataset.
    pub subject_id: String,
    /// Records sharing a group are never split between a training and a test
    /// set (a household, a site, the same person seen twice).
    #[serde(default)]
    pub group_id: Option<String>,
    /// Sampling weight (a survey design's, or 1).
    #[serde(default = "one")]
    pub weight: f64,
    /// Where the record came from; its effect on predictions can be ablated.
    pub source: String,
    /// Prediction time on the subject's clock.
    pub entry: f64,
    /// The calendar time (e.g. decimal year) at `entry`.
    pub calendar_at_entry: f64,
    /// Measurements, at or before `entry` (input) or after it (forecast targets).
    #[serde(default)]
    pub observations: Vec<Observation>,
    /// Events, before `entry` (history) or after it (outcomes).
    #[serde(default)]
    pub events: Vec<Event>,
    /// Observation windows for outcome codes.
    #[serde(default)]
    pub at_risk: Vec<AtRisk>,
}

fn one() -> f64 {
    1.0
}

impl Subject {
    /// Parse one `timeline-v1` line and validate it.
    pub fn from_json_line(line: &str) -> Result<Subject, String> {
        let s: Subject = serde_json::from_str(line).map_err(|e| format!("timeline-v1: {e}"))?;
        s.validate()?;
        Ok(s)
    }

    /// The semantic checks `serde` cannot make: finite times, a positive
    /// weight, and observation windows that are intervals opening no earlier
    /// than entry. An event dated before its window opens is legal (history
    /// before entry is the input) and is simply not scored as an outcome.
    pub fn validate(&self) -> Result<(), String> {
        let who = &self.subject_id;
        let finite = |what: &str, v: f64| {
            if v.is_finite() {
                Ok(())
            } else {
                Err(format!("subject {who}: {what} is not finite ({v})"))
            }
        };
        finite("entry", self.entry)?;
        finite("calendar_at_entry", self.calendar_at_entry)?;
        if !(self.weight.is_finite() && self.weight > 0.0) {
            return Err(format!(
                "subject {who}: weight must be positive and finite, got {}",
                self.weight
            ));
        }
        for o in &self.observations {
            finite(&format!("observation {} time", o.var), o.t)?;
            let v = match &o.value {
                Value::Number(v) | Value::Below { below: v } | Value::Above { above: v } => *v,
                Value::Category(_) => 0.0,
            };
            finite(&format!("observation {} value", o.var), v)?;
        }
        for e in &self.events {
            finite(&format!("event {} time", e.code), e.t)?;
        }
        for w in &self.at_risk {
            finite(&format!("at_risk {} from", w.code), w.from)?;
            finite(&format!("at_risk {} to", w.code), w.to)?;
            if w.to < w.from {
                return Err(format!(
                    "subject {who}: at_risk {} ends ({}) before it starts ({})",
                    w.code, w.to, w.from
                ));
            }
            if w.from < self.entry {
                return Err(format!(
                    "subject {who}: at_risk {} opens at {} before entry {}: an outcome window cannot start before the prediction time",
                    w.code, w.from, self.entry
                ));
            }
        }
        Ok(())
    }

    /// The observation window for `code`: its own, else the `*` window.
    pub fn window(&self, code: &str) -> Option<&AtRisk> {
        self.at_risk
            .iter()
            .find(|w| w.code == code)
            .or_else(|| self.at_risk.iter().find(|w| w.code == "*"))
    }

    /// Observations known at prediction time: at or before `entry`. The only
    /// observations an encoder may read - anything later is the future.
    pub fn known_observations(&self) -> impl Iterator<Item = &Observation> {
        self.observations.iter().filter(move |o| o.t <= self.entry)
    }

    /// Events known at prediction time: strictly before `entry`.
    pub fn known_events(&self) -> impl Iterator<Item = &Event> {
        self.events.iter().filter(move |e| e.t < self.entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: &str = r#"{"subject_id":"a","weight":2.5,"source":"s","entry":50.0,"calendar_at_entry":2000.5,
        "observations":[{"t":50.0,"var":"sbp","value":131.0},{"t":50.0,"var":"crp","value":{"below":0.2}},
                        {"t":25.0,"var":"weight","value":70.0},{"t":51.0,"var":"sbp","value":140.0},
                        {"t":50.0,"var":"smoking","value":"never"}],
        "events":[{"t":44.0,"code":"dx:hypertension"},{"t":62.5,"code":"death:heart"}],
        "at_risk":[{"code":"*","from":50.0,"to":62.5}]}"#;

    #[test]
    fn a_valid_line_parses_and_splits_past_from_future() {
        let s = Subject::from_json_line(&LINE.replace('\n', " ")).unwrap();
        assert_eq!(s.weight, 2.5);
        assert_eq!(s.observations[1].value, Value::Below { below: 0.2 });
        assert_eq!(s.observations[4].value, Value::Category("never".into()));
        let known: Vec<_> = s.known_observations().map(|o| o.t).collect();
        assert_eq!(
            known,
            vec![50.0, 50.0, 25.0, 50.0],
            "the t=51 measurement is the future"
        );
        assert_eq!(
            s.known_events().count(),
            1,
            "death after entry is an outcome, not history"
        );
        assert_eq!(s.window("death:heart").unwrap().to, 62.5);
    }

    #[test]
    fn unknown_fields_and_contradictory_times_are_refused() {
        let unknown = LINE
            .replace('\n', " ")
            .replace("\"source\"", "\"sourse\":1,\"source\"");
        assert!(Subject::from_json_line(&unknown)
            .unwrap_err()
            .contains("unknown field"));
        let backwards = LINE
            .replace('\n', " ")
            .replace("\"to\":62.5", "\"to\":40.0");
        assert!(Subject::from_json_line(&backwards)
            .unwrap_err()
            .contains("ends"));
        let early = LINE
            .replace('\n', " ")
            .replace("\"from\":50.0", "\"from\":49.0");
        assert!(Subject::from_json_line(&early)
            .unwrap_err()
            .contains("before entry"));
        let weightless = LINE
            .replace('\n', " ")
            .replace("\"weight\":2.5", "\"weight\":0");
        assert!(Subject::from_json_line(&weightless)
            .unwrap_err()
            .contains("weight"));
    }
}
