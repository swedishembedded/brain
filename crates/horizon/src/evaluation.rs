// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Held-out evaluation of a saved model: the metrics a risk model is judged
//! by, per outcome code and horizon, from [`survival`]'s arithmetic.
//!
//! Swedish Embedded AB implements validation of risk models whose outcomes
//! arrive years later, censored and competing, for its clients. If your team
//! needs expertise in evaluating time-to-event predictions honestly you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! For every outcome code, each subject's observed outcome is its first event
//! of the code or of an absorbing code inside the code's observation window,
//! else censoring ([`crate::calibration::observed`]: the code competes with
//! the absorbing codes exactly as its cumulative incidence does). At every
//! horizon with at least [`EvaluationSpec::min_events`] events of the code by
//! then (30 by default, as a calibration needs) it reports, all under the
//! subjects' sampling weights:
//!
//! - Uno's concordance truncated at the horizon, the time-dependent AUC and
//!   the IPCW Brier score at the horizon;
//! - the Brier score integrated over `(0, horizon]` on an even grid;
//! - the recalibration slope and intercept, the intercept with the slope fixed
//!   at one (calibration in the large), observed over expected and the
//!   expected calibration error, the observed side an Aalen-Johansen estimate;
//! - when subjects carry a `group_id`, percentile intervals of the
//!   concordance, AUC and Brier score from a bootstrap that resamples whole
//!   groups ([`survival::compare::cluster_bootstrap_by`]), so members of a
//!   household or a site are never treated as independent.
//!
//! A horizon with fewer events is ABSENT from `results` and listed in
//! `absent`: a metric from a handful of events is noise, not a number to
//! report. A metric that cannot be computed (no comparable pair, a vanishing
//! censoring distribution) is `None`, never zero.
//!
//! The censoring distribution behind the IPCW weights is estimated on the
//! evaluated subjects unless [`EvaluationSpec::censoring_from`] names a
//! reference set (the training subjects, as the concordance's definition
//! prefers). The held-out event NLL is one number per subject set, not per
//! horizon.

use std::collections::BTreeMap;

use serde::Serialize;
use survival::calibration::at_horizon;
use survival::compare::cluster_bootstrap_by;
use survival::estimate::{censoring, Step};
use survival::{auc, brier, concordance, Obs};

use crate::calibration::{observed, MIN_EVENTS};
use crate::saved::Saved;
use crate::timeline::Subject;
use crate::train::event_nll;

/// Risk groups in the calibration table behind the expected calibration error.
pub const CALIBRATION_GROUPS: usize = 10;
/// Grid points of the integrated Brier score.
pub const INTEGRATION_POINTS: usize = 20;

/// A cluster bootstrap of the metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bootstrap {
    /// Resamples.
    pub reps: usize,
    /// Coverage of the percentile interval (0.95).
    pub level: f64,
    /// Seed of the resampling: the same seed gives the same intervals.
    pub seed: u64,
}

impl Default for Bootstrap {
    fn default() -> Self {
        Bootstrap { reps: 200, level: 0.95, seed: 0 }
    }
}

/// What to evaluate.
#[derive(Clone, Debug)]
pub struct EvaluationSpec {
    /// Horizons after each subject's entry, each in `(0, last knot]`.
    pub horizons: Vec<f64>,
    /// The fewest events of a code by a horizon for its metrics to be reported.
    pub min_events: usize,
    /// Bootstrap intervals when the subjects have groups; `None` for none.
    pub bootstrap: Option<Bootstrap>,
    /// Subjects to estimate the censoring distribution on instead of the
    /// evaluated ones.
    pub censoring_reference: Option<Vec<Subject>>,
}

impl EvaluationSpec {
    /// Evaluate at `horizons`, with the default minimum and bootstrap.
    pub fn new(horizons: impl IntoIterator<Item = f64>) -> EvaluationSpec {
        EvaluationSpec {
            horizons: horizons.into_iter().collect(),
            min_events: MIN_EVENTS,
            bootstrap: Some(Bootstrap::default()),
            censoring_reference: None,
        }
    }
    /// Report a (code, horizon) only with at least `n` events by then.
    pub fn min_events(mut self, n: usize) -> Self {
        self.min_events = n;
        self
    }
    /// Bootstrap with these settings, or not at all (`None`).
    pub fn bootstrap(mut self, bootstrap: Option<Bootstrap>) -> Self {
        self.bootstrap = bootstrap;
        self
    }
    /// Estimate the censoring distribution on `reference` (typically the
    /// training subjects) instead of on the evaluated subjects.
    pub fn censoring_from(mut self, reference: Vec<Subject>) -> Self {
        self.censoring_reference = Some(reference);
        self
    }
}

/// A percentile bootstrap interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Interval {
    /// The statistic on the full sample.
    pub estimate: f64,
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
}

/// Intervals of the headline metrics from the group bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Intervals {
    /// Uno's concordance.
    pub uno_c: Option<Interval>,
    /// The time-dependent AUC.
    pub auc: Option<Interval>,
    /// The IPCW Brier score.
    pub brier: Option<Interval>,
}

/// Calibration of the risk by one horizon.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CalibrationMetrics {
    /// Recalibration slope (1 is ideal).
    pub slope: Option<f64>,
    /// Recalibration intercept (0 is ideal). It is read at a predicted risk of
    /// one half, so a slope away from one moves it a long way for risks that
    /// are nowhere near that: judge the average risk by
    /// [`Self::intercept_in_the_large`].
    pub intercept: Option<f64>,
    /// Calibration in the large: the intercept with the slope fixed at one
    /// (0 is ideal), the shift in log-odds that makes the average predicted
    /// risk the observed one.
    pub intercept_in_the_large: Option<f64>,
    /// Observed (Aalen-Johansen) over expected (mean predicted) risk.
    pub observed_over_expected: Option<f64>,
    /// Weighted mean absolute gap between observed and expected over
    /// [`CALIBRATION_GROUPS`] risk groups.
    pub ece: Option<f64>,
}

/// One outcome code at one horizon.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HorizonMetrics {
    /// The outcome code.
    pub code: String,
    /// The horizon.
    pub horizon: f64,
    /// Subjects with an event of the code by the horizon.
    pub events: usize,
    /// Uno's concordance truncated at the horizon.
    pub uno_c: Option<f64>,
    /// The time-dependent AUC at the horizon.
    pub auc: Option<f64>,
    /// The IPCW Brier score at the horizon.
    pub brier: Option<f64>,
    /// The Brier score integrated over `(0, horizon]`.
    pub integrated_brier: Option<f64>,
    /// Calibration at the horizon.
    pub calibration: CalibrationMetrics,
    /// Group-bootstrap intervals; present only when subjects have groups.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intervals: Option<Intervals>,
}

/// A (code, horizon) left out for want of events.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Absent {
    /// The outcome code.
    pub code: String,
    /// The horizon.
    pub horizon: f64,
    /// Events of the code by the horizon.
    pub events: usize,
    /// The minimum that applied.
    pub min_events: usize,
}

/// The result of [`evaluate`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Evaluation {
    /// Subjects evaluated.
    pub subjects: usize,
    /// The weighted mean event NLL of the outcome codes (lower is better).
    pub event_nll: f64,
    /// Metrics per (code, horizon) with enough events, codes in the model's
    /// order and horizons ascending.
    pub results: Vec<HorizonMetrics>,
    /// The pairs left out.
    pub absent: Vec<Absent>,
}

impl Evaluation {
    /// The metrics of `code` at `horizon`, if reported.
    pub fn at(&self, code: &str, horizon: f64) -> Option<&HorizonMetrics> {
        self.results.iter().find(|r| r.code == code && r.horizon == horizon)
    }
}

/// The three headline metrics of one sample, the ones a bootstrap resamples.
struct Headline {
    uno_c: Option<f64>,
    auc: Option<f64>,
    brier: Option<f64>,
}

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

/// Evaluate `saved` on `subjects` (held out: neither trained on nor used to
/// early-stop or calibrate). See the module docs.
pub fn evaluate(saved: &Saved, subjects: &[Subject], spec: &EvaluationSpec) -> Result<Evaluation, String> {
    if subjects.is_empty() {
        return Err("evaluate: subjects are required".into());
    }
    let last = saved.horizon();
    let mut horizons = spec.horizons.clone();
    if horizons.is_empty() {
        return Err("evaluate: at least one horizon is required".into());
    }
    if let Some(h) = horizons.iter().find(|h| !(h.is_finite() && **h > 0.0 && **h <= last)) {
        return Err(format!("evaluate: horizon {h} is outside the model's range (0, {last}]"));
    }
    horizons.sort_by(f64::total_cmp);
    horizons.dedup();
    let curves = saved.predict(subjects)?;
    let nll = f64::from(event_nll(&saved.model, &saved.encode(subjects)?));
    let clusters = clusters(subjects);
    let codes = &saved.vocab.codes;
    let (mut results, mut absent) = (Vec::new(), Vec::new());
    for (k, code) in codes.iter().enumerate() {
        // The code, then the absorbing codes it competes with.
        let mut list: Vec<&str> = vec![code.as_str()];
        list.extend(
            codes
                .iter()
                .enumerate()
                .filter(|(j, c)| *j != k && saved.vocab.absorbing.contains(c))
                .map(|(_, c)| c.as_str()),
        );
        let obs = observed(subjects, &list);
        let fixed_g = spec
            .censoring_reference
            .as_ref()
            .map(|r| censoring(&observed(r, &list)));
        let g = fixed_g.clone().unwrap_or_else(|| censoring(&obs));
        for &h in &horizons {
            let events = obs.iter().filter(|o| o.time <= h && o.cause == Some(0)).count();
            if events < spec.min_events {
                absent.push(Absent { code: code.clone(), horizon: h, events, min_events: spec.min_events });
                continue;
            }
            let cif_at = |t: f64| -> Vec<f64> { curves.iter().map(|c| c.cif(k, t)).collect() };
            let risk = cif_at(h);
            let grid: Vec<f64> = (1..=INTEGRATION_POINTS)
                .map(|i| h * i as f64 / INTEGRATION_POINTS as f64)
                .collect();
            let calibration = at_horizon(&risk, &obs, 0, h, &g, CALIBRATION_GROUPS);
            let metrics = |risk: &[f64], obs: &[Obs], g: &Step| Headline {
                uno_c: concordance::uno(risk, obs, 0, h, g),
                auc: auc::at(risk, obs, 0, h, g),
                brier: brier::brier(risk, obs, 0, h, g),
            };
            let full = metrics(&risk, &obs, &g);
            let intervals = match (spec.bootstrap, clusters.as_ref()) {
                (Some(b), Some(clusters)) => {
                    let stat = |pick: fn(&Headline) -> Option<f64>| {
                        let (risk, obs, fixed_g) = (&risk, &obs, &fixed_g);
                        move |idx: &[usize]| {
                            let sub_obs: Vec<Obs> = idx.iter().map(|&i| obs[i]).collect();
                            let sub_risk: Vec<f64> = idx.iter().map(|&i| risk[i]).collect();
                            let own;
                            let g = match fixed_g {
                                Some(g) => g,
                                None => {
                                    own = censoring(&sub_obs);
                                    &own
                                }
                            };
                            pick(&metrics(&sub_risk, &sub_obs, g)).and_then(finite)
                        }
                    };
                    let interval = |pick| {
                        cluster_bootstrap_by(clusters, b.reps, b.level, b.seed, stat(pick)).map(|i| Interval {
                            estimate: i.estimate,
                            lo: i.lo,
                            hi: i.hi,
                        })
                    };
                    Some(Intervals {
                        uno_c: interval(|m| m.uno_c),
                        auc: interval(|m| m.auc),
                        brier: interval(|m| m.brier),
                    })
                }
                _ => None,
            };
            results.push(HorizonMetrics {
                code: code.clone(),
                horizon: h,
                events,
                uno_c: full.uno_c.and_then(finite),
                auc: full.auc.and_then(finite),
                brier: full.brier.and_then(finite),
                integrated_brier: brier::integrated_brier(&grid, |i| cif_at(grid[i]), &obs, 0, &g)
                    .and_then(finite),
                calibration: CalibrationMetrics {
                    slope: finite(calibration.slope),
                    intercept: finite(calibration.intercept),
                    intercept_in_the_large: finite(calibration.intercept_in_the_large),
                    observed_over_expected: finite(calibration.oe_ratio),
                    ece: finite(calibration.ece()),
                },
                intervals,
            });
        }
    }
    Ok(Evaluation { subjects: subjects.len(), event_nll: nll, results, absent })
}

/// One cluster id per subject when any subject has a group (a subject without
/// one is its own cluster), else `None`.
fn clusters(subjects: &[Subject]) -> Option<Vec<u64>> {
    if subjects.iter().all(|s| s.group_id.is_none()) {
        return None;
    }
    let mut ids: BTreeMap<&str, u64> = BTreeMap::new();
    let mut next = 0u64;
    Some(
        subjects
            .iter()
            .map(|s| match &s.group_id {
                Some(g) => *ids.entry(g.as_str()).or_insert_with(|| {
                    next += 1;
                    next
                }),
                None => {
                    next += 1;
                    next
                }
            })
            .collect(),
    )
}
