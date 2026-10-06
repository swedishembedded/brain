// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The structured forecast: what a caller gets back for one patient history.
//!
//! Swedish Embedded AB implements risk forecasts that say what they were
//! computed from, how far to trust them and when to stay silent, for its
//! clients. If your team needs expertise in auditable, calibrated risk
//! reporting you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! A [`RiskForecast`] is a statistical statement about the TRAINING
//! POPULATION'S outcomes given a recorded history. It does not diagnose and it
//! recommends no treatment: predicted risks are associations in the data the
//! model was trained on, not the effects of any action, and carry that
//! disclaimer ([`DISCLAIMER`]) in every answer.
//!
//! What it holds:
//!
//! - the **model identity** (weights digest, configuration digest, brain
//!   version) and the history's `as_of`;
//! - the input **coverage**: which model variables and event codes the history
//!   had, how many observations and events, the oldest and newest observation,
//!   and which model variables were MISSING;
//! - **curves**: for every outcome code the cumulative incidence at the model's
//!   knots, and survival;
//! - **horizons**: for each horizon the request names, the raw risk and, only
//!   where the model was calibrated for exactly that horizon, the calibrated
//!   risk and, for a Venn-Abers calibration, its interval - ABSENT otherwise,
//!   never zero. A
//!   horizon past the last knot is refused, not extrapolated;
//! - the **uncertainty** that exists: that interval, and for an ensemble the
//!   range over its members;
//! - the data-quality **support** assessment and the **input warnings** of the
//!   history format;
//! - or, when the history is outside what the model was trained on,
//!   **abstention**: `risk: "unavailable"` with a reason and no numbers.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::history::{HistoryWarning, PatientHistory};
use crate::saved::{ModelIdentity, Saved, Scored};
use crate::support::{AssessOptions, Assessment, DEFAULT_MAX_OOD_SCORE};
use crate::survival::Curves;
use crate::vocab::Vocab;

/// Carried by every forecast.
pub const DISCLAIMER: &str = "This forecast is a statistical estimate of outcomes in the population the model was trained on, \
given the recorded history. It is not a diagnosis and not a treatment recommendation; predicted \
risks are associations in the training data, not effects of any action.";

/// What to forecast.
#[derive(Clone, Debug, PartialEq)]
pub struct ForecastRequest {
    /// Horizons after `as_of`, in the model's unit, each in `(0, last knot]`.
    pub horizons: Vec<f64>,
    /// A history whose out-of-distribution score is above this gets no numbers
    /// (1 is the edge of the training support).
    pub max_ood_score: f64,
    /// How strictly support is judged.
    pub assess: AssessOptions,
}

impl ForecastRequest {
    /// Forecast at `horizons`, abstaining at the default threshold.
    pub fn new(horizons: impl IntoIterator<Item = f64>) -> ForecastRequest {
        ForecastRequest {
            horizons: horizons.into_iter().collect(),
            max_ood_score: DEFAULT_MAX_OOD_SCORE,
            assess: AssessOptions::default(),
        }
    }

    /// Withhold the numbers above this out-of-distribution score.
    pub fn max_ood_score(mut self, max_ood_score: f64) -> Self {
        self.max_ood_score = max_ood_score;
        self
    }

    /// Check the request against a model whose last knot is `last_knot`.
    pub fn validate(&self, last_knot: f64) -> Result<(), String> {
        if !(self.max_ood_score.is_finite() && self.max_ood_score > 0.0) {
            return Err(format!(
                "forecast: max_ood_score must be positive, got {}",
                self.max_ood_score
            ));
        }
        match self.horizons.iter().find(|h| !(h.is_finite() && **h > 0.0 && **h <= last_knot)) {
            Some(h) => Err(format!(
                "forecast: horizon {h} is outside the model's range (0, {last_knot}]: the model says nothing later and does not extrapolate"
            )),
            None => Ok(()),
        }
    }
}

/// Whether numbers are given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskStatus {
    /// The forecast carries risks.
    Available,
    /// The history is outside what the model was trained on: no numbers.
    Unavailable,
}

/// The prediction time as given and as the model reads it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AsOf {
    /// As written in the history.
    pub input: String,
    /// On the model's clock (years of age, for a person).
    pub time: f64,
    /// The calendar time (decimal year) the model read.
    pub calendar: f64,
}

/// An observation's time on the model's clock and how long before `as_of`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Stamp {
    /// On the model's clock.
    pub time: f64,
    /// `as_of - time`.
    pub ago: f64,
}

/// What the history supplied, against what the model reads.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Coverage {
    /// Measured variables present, sorted.
    pub variables: Vec<String>,
    /// Event codes present, sorted.
    pub event_codes: Vec<String>,
    /// Measurements used.
    pub observations: usize,
    /// Events used.
    pub events: usize,
    /// The earliest measurement used.
    pub oldest_observation: Option<Stamp>,
    /// The latest measurement used.
    pub newest_observation: Option<Stamp>,
    /// Variables the model reads that the history did not measure, sorted.
    pub missing_variables: Vec<String>,
}

impl Coverage {
    /// The coverage of `history` against `vocab`'s variables.
    pub fn of(history: &PatientHistory, vocab: &Vocab) -> Coverage {
        let measured = || history.records.iter().filter(|r| r.value.is_some());
        let variables: BTreeSet<&str> = measured().map(|r| r.code.as_str()).collect();
        let events = history.records.iter().filter(|r| r.value.is_none());
        let event_codes: BTreeSet<&str> = events.clone().map(|r| r.code.as_str()).collect();
        let stamp = |time: f64| Stamp { time, ago: history.as_of - time };
        Coverage {
            variables: variables.iter().map(|v| v.to_string()).collect(),
            event_codes: event_codes.iter().map(|v| v.to_string()).collect(),
            observations: measured().count(),
            events: events.count(),
            // Records are in canonical (time) order.
            oldest_observation: measured().next().map(|r| stamp(r.time)),
            newest_observation: measured().next_back().map(|r| stamp(r.time)),
            missing_variables: vocab
                .variables()
                .into_iter()
                .filter(|v| !variables.contains(v.as_str()))
                .collect(),
        }
    }
}

/// The curves over the model's knots.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CurveSet {
    /// The model's knots (times after `as_of`).
    pub times: Vec<f64>,
    /// Probability of no absorbing outcome by each time.
    pub survival: Vec<f64>,
    /// Per outcome code, the raw cumulative incidence by each time.
    pub cif: BTreeMap<String, Vec<f64>>,
}

/// One outcome's risk at one horizon.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CodeRisk {
    /// The model's raw probability.
    pub raw: f64,
    /// The calibrated probability; absent where the model was not calibrated
    /// for this code at this horizon.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibrated: Option<f64>,
    /// The Venn-Abers interval behind `calibrated`; absent with it, and
    /// absent for a logistic calibration, which has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval: Option<[f64; 2]>,
    /// For an ensemble, the lowest and highest raw probability of its members.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub member_range: Option<[f64; 2]>,
}

/// The risks at one requested horizon.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HorizonRisks {
    /// After `as_of`, in the model's unit.
    pub horizon: f64,
    /// Probability of no absorbing outcome by then.
    pub survival: f64,
    /// Per outcome code.
    pub risks: BTreeMap<String, CodeRisk>,
}

/// An ensemble's members and the range of its curves.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnsembleSummary {
    /// Every member's identity.
    pub members: Vec<ModelIdentity>,
    /// Per outcome code, the lowest member cumulative incidence at each knot.
    pub cif_min: BTreeMap<String, Vec<f64>>,
    /// Per outcome code, the highest member cumulative incidence at each knot.
    pub cif_max: BTreeMap<String, Vec<f64>>,
}

/// The structured forecast for one patient history. See the module docs: it
/// does not diagnose or recommend treatment, and its risks are associations,
/// not effects.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RiskForecast {
    /// The history's id.
    pub subject_id: String,
    /// The prediction time.
    pub as_of: AsOf,
    /// The model (for an ensemble, its first member; see `ensemble`).
    pub model: ModelIdentity,
    /// What the history supplied.
    pub coverage: Coverage,
    /// Whether numbers are given.
    pub risk: RiskStatus,
    /// Why not, when `risk` is unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The curves over the knots; absent when unavailable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curves: Option<CurveSet>,
    /// The requested horizons; empty when unavailable.
    pub horizons: Vec<HorizonRisks>,
    /// Present for an ensemble.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ensemble: Option<EnsembleSummary>,
    /// The data-quality assessment against the training support.
    pub support: Assessment,
    /// What the history format dropped or ignored.
    pub input_warnings: Vec<HistoryWarning>,
    /// Not a diagnosis, not treatment advice, associations not effects.
    pub disclaimer: &'static str,
}

impl RiskForecast {
    /// Whether the forecast carries numbers.
    pub fn is_available(&self) -> bool {
        self.risk == RiskStatus::Available
    }

    /// The forecast of `history` from its `scored` curves and assessment under
    /// `saved`'s model. `req` must have been validated against the model.
    pub fn from_scored(
        saved: &Saved,
        identity: &ModelIdentity,
        history: &PatientHistory,
        scored: &Scored,
        req: &ForecastRequest,
    ) -> RiskForecast {
        let mut forecast = RiskForecast {
            subject_id: history.id.clone(),
            as_of: AsOf {
                input: history.as_of_input.clone(),
                time: history.as_of,
                calendar: history.calendar_at_as_of,
            },
            model: identity.clone(),
            coverage: Coverage::of(history, &saved.vocab),
            risk: RiskStatus::Available,
            reason: None,
            curves: None,
            horizons: Vec::new(),
            ensemble: None,
            support: scored.assessment.clone(),
            input_warnings: history.warnings.clone(),
            disclaimer: DISCLAIMER,
        };
        if scored.assessment.abstains(req.max_ood_score) {
            forecast.risk = RiskStatus::Unavailable;
            forecast.reason = Some("insufficient support".into());
            return forecast;
        }
        let codes = &saved.vocab.codes;
        let curves = &scored.curves;
        let knots: Vec<f64> = saved.model.cfg.knots.iter().map(|&k| f64::from(k)).collect();
        forecast.curves = Some(CurveSet {
            survival: knots.iter().map(|&t| curves.survival(t)).collect(),
            cif: codes
                .iter()
                .enumerate()
                .map(|(k, c)| (c.clone(), knots.iter().map(|&t| curves.cif(k, t)).collect()))
                .collect(),
            times: knots,
        });
        forecast.horizons = req
            .horizons
            .iter()
            .map(|&h| HorizonRisks {
                horizon: h,
                survival: curves.survival(h),
                risks: codes
                    .iter()
                    .enumerate()
                    .map(|(k, code)| (code.clone(), code_risk(saved, curves, code, k, h)))
                    .collect(),
            })
            .collect();
        forecast
    }

    /// The equal-weight ensemble of forecasts of the SAME history from models
    /// trained apart (say with different seeds): every probability is the mean
    /// of the members', the calibrated risk and interval exist only where every
    /// member has them, and the range over the members is reported. The
    /// ensemble abstains if any member does. An error for no parts, or parts
    /// that disagree on the history, the outcome codes, the knots or the
    /// horizons.
    pub fn ensemble(parts: &[RiskForecast]) -> Result<RiskForecast, String> {
        let first = parts.first().ok_or("ensemble: no forecasts")?;
        if let Some(p) = parts.iter().find(|p| !p.is_available()) {
            let mut abstained = p.clone();
            abstained.ensemble = None;
            return Ok(abstained);
        }
        let shape = |p: &RiskForecast| {
            let curves = p.curves.as_ref().map(|c| (c.times.clone(), c.cif.keys().cloned().collect::<Vec<_>>()));
            (p.subject_id.clone(), p.as_of.clone(), curves, p.horizons.iter().map(|h| h.horizon).collect::<Vec<_>>())
        };
        if parts.iter().any(|p| shape(p) != shape(first)) {
            return Err("ensemble: forecasts differ in history, as_of, outcome codes, knots or horizons".into());
        }
        let n = parts.len() as f64;
        let mut out = first.clone();
        out.coverage.missing_variables = parts
            .iter()
            .flat_map(|p| p.coverage.missing_variables.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        out.support = parts
            .iter()
            .map(|p| &p.support)
            .max_by(|a, b| a.ood_score.partial_cmp(&b.ood_score).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or_else(|| first.support.clone());
        let curves = out.curves.as_mut().expect("available forecasts have curves");
        let member_curves: Vec<&CurveSet> = parts.iter().filter_map(|p| p.curves.as_ref()).collect();
        let mean = |f: &dyn Fn(&CurveSet) -> &Vec<f64>, i: usize| member_curves.iter().map(|c| f(c)[i]).sum::<f64>() / n;
        for i in 0..curves.times.len() {
            curves.survival[i] = mean(&|c| &c.survival, i);
        }
        let (mut cif_min, mut cif_max) = (BTreeMap::new(), BTreeMap::new());
        for (code, values) in &mut curves.cif {
            let at = |i: usize| member_curves.iter().map(move |c| c.cif[code][i]);
            let (mut lo, mut hi) = (Vec::new(), Vec::new());
            for (i, v) in values.iter_mut().enumerate() {
                *v = at(i).sum::<f64>() / n;
                lo.push(at(i).fold(f64::INFINITY, f64::min));
                hi.push(at(i).fold(f64::NEG_INFINITY, f64::max));
            }
            cif_min.insert(code.clone(), lo);
            cif_max.insert(code.clone(), hi);
        }
        for (hi_idx, h) in out.horizons.iter_mut().enumerate() {
            let members: Vec<&HorizonRisks> = parts.iter().map(|p| &p.horizons[hi_idx]).collect();
            h.survival = members.iter().map(|m| m.survival).sum::<f64>() / n;
            for (code, risk) in &mut h.risks {
                let each: Vec<&CodeRisk> = members.iter().map(|m| &m.risks[code]).collect();
                risk.raw = each.iter().map(|r| r.raw).sum::<f64>() / n;
                risk.member_range = Some([
                    each.iter().map(|r| r.raw).fold(f64::INFINITY, f64::min),
                    each.iter().map(|r| r.raw).fold(f64::NEG_INFINITY, f64::max),
                ]);
                let calibrated: Option<Vec<f64>> = each.iter().map(|r| r.calibrated).collect();
                let interval: Option<Vec<[f64; 2]>> = each.iter().map(|r| r.interval).collect();
                risk.calibrated = calibrated.map(|c| c.iter().sum::<f64>() / n);
                // An interval exists only where every member has one (a
                // logistic calibration has none) and a calibrated risk does.
                risk.interval = interval.filter(|_| risk.calibrated.is_some()).map(|i| {
                    [i.iter().map(|x| x[0]).sum::<f64>() / n, i.iter().map(|x| x[1]).sum::<f64>() / n]
                });
            }
        }
        out.ensemble = Some(EnsembleSummary {
            members: parts.iter().map(|p| p.model.clone()).collect(),
            cif_min,
            cif_max,
        });
        Ok(out)
    }
}

fn code_risk(saved: &Saved, curves: &Curves, code: &str, k: usize, horizon: f64) -> CodeRisk {
    let raw = curves.cif(k, horizon);
    let calibrated = saved.calibration.as_ref().and_then(|c| c.apply(code, horizon, raw));
    CodeRisk {
        raw,
        calibrated: calibrated.map(|c| c.risk),
        interval: calibrated.and_then(|c| c.interval).map(|(lo, hi)| [lo, hi]),
        member_range: None,
    }
}

/// Forecast each history under `saved`'s model, in order, from one forward
/// pass. Errors (with the history's id) for a history the model cannot read,
/// such as a unit it was not trained on, and for a request outside the model.
pub fn forecast(
    saved: &Saved,
    histories: &[PatientHistory],
    req: &ForecastRequest,
) -> Result<Vec<RiskForecast>, String> {
    req.validate(saved.horizon())?;
    let subjects = histories
        .iter()
        .map(|h| h.to_subject(&saved.vocab))
        .collect::<Result<Vec<_>, _>>()?;
    let scored = saved.score(&subjects, &req.assess)?;
    let identity = saved.identity()?;
    Ok(histories
        .iter()
        .zip(&scored)
        .map(|(h, s)| RiskForecast::from_scored(saved, &identity, h, s, req))
        .collect())
}
