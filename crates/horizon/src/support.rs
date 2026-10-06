// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the model was trained on, kept beside the weights, and the question
//! it answers: is THIS subject inside it?
//!
//! Swedish Embedded AB implements prediction systems that know when not to
//! answer, for its clients. If your team needs expertise in out-of-
//! distribution detection and abstention for risk models you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! [`Support`] is fitted on the TRAINING subjects when the model is trained
//! and saved as `support.json` (bound, like a calibration, to the SHA-256 of
//! the weights, because the state statistics belong to those weights). It
//! records, per numeric variable, the robust range of its values; per
//! categorical variable, the levels seen; the event codes seen in histories;
//! the ranges of entry clock, calendar time and history length; and the mean
//! and covariance of the learned state, for a Mahalanobis distance.
//!
//! [`Support::assess`] turns a subject into an [`Assessment`]: typed
//! [`Warning`]s and one continuous `ood_score` that is the largest of the
//! component scores, each scaled so that `1.0` is the edge of what is
//! supported:
//!
//! - a range component is `|x - mid| / (half * (1 + 2 margin))` over the
//!   robust range `[q0.5%, q99.5%]`, with the margin a share of its width
//!   ([`AssessOptions::margin`]);
//! - the state component is the Mahalanobis distance of the subject's state
//!   over the training set's 99th percentile of it, times `1 + state_margin`;
//! - an unknown variable, category or event code scores [`UNKNOWN_SCORE`].
//!
//! A subject is `supported` when its score is at most 1. A model saved before
//! this existed has no `support.json`: it assesses as support UNKNOWN
//! (`supported: None`) - neither supported nor unsupported, because nothing
//! is known about what it was trained on.
//!
//! Limits, so nobody reads more into it than it says: the ranges are
//! marginal (a pair of values that never co-occurred passes if each is
//! common), the state distance is the only joint check and is as good as the
//! state's Gaussian-like summary of the history, and in-distribution subjects
//! are flagged at a small but non-zero rate that the margins set.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::saved::Saved;
use crate::timeline::{Subject, Value};
use crate::train::predict_states;

/// The support file inside a saved model's directory.
pub const FILE: &str = "support.json";
/// The score of an unknown variable, category or event code: far outside
/// `[0, 1]`, so no threshold near the default tolerates it.
pub const UNKNOWN_SCORE: f64 = 10.0;
/// The default threshold on [`Assessment::ood_score`] above which a served
/// subject gets `risk: unavailable` instead of probabilities: the edge of the
/// support itself, so any warning withholds the answer. Conservative on
/// purpose; an operator who accepts more raises it.
pub const DEFAULT_MAX_OOD_SCORE: f64 = 1.0;
/// The default share of a robust range's width allowed beyond it.
pub const DEFAULT_MARGIN: f64 = 0.25;
/// The default relative slack on the training set's 99th-percentile state
/// distance.
pub const DEFAULT_STATE_MARGIN: f64 = 0.5;
/// A count range narrower than this is widened to it (history lengths of 5 to
/// 6 tokens must not make 7 an anomaly).
const COUNT_FLOOR_WIDTH: f64 = 4.0;
/// The state statistics need this many training subjects per state
/// dimension; fewer cannot pin a covariance down.
const STATE_SUBJECTS_PER_DIM: usize = 4;
const VERSION: u32 = 1;

/// How strict [`Support::assess`] is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssessOptions {
    /// Share of a range's width tolerated beyond it before a value is outside
    /// the support.
    pub margin: f64,
    /// Relative slack tolerated on the state distance.
    pub state_margin: f64,
}

impl Default for AssessOptions {
    fn default() -> Self {
        AssessOptions {
            margin: DEFAULT_MARGIN,
            state_margin: DEFAULT_STATE_MARGIN,
        }
    }
}

/// Why a subject is not (fully) supported.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Warning {
    /// A variable the model was never trained on (or trained on as the other
    /// kind: numeric against categorical).
    UnknownVariable {
        /// The variable.
        var: String,
    },
    /// A level of a categorical variable never seen in training.
    UnknownCategory {
        /// The variable.
        var: String,
        /// The level.
        level: String,
    },
    /// A history event code never seen in training.
    UnknownEventCode {
        /// The code.
        code: String,
    },
    /// A measurement outside the range the variable took in training, by more
    /// than the margin.
    ValueOutOfRange {
        /// The variable.
        var: String,
        /// The most extreme value of it in the subject's history.
        value: f64,
        /// The lowest supported value.
        low: f64,
        /// The highest supported value.
        high: f64,
    },
    /// An entry clock (e.g. age) outside the range seen in training.
    EntryOutOfRange {
        /// The subject's entry.
        value: f64,
        /// The lowest supported entry.
        low: f64,
        /// The highest supported entry.
        high: f64,
    },
    /// A calendar time at entry outside the range seen in training.
    CalendarOutOfRange {
        /// The subject's calendar time.
        value: f64,
        /// The earliest supported.
        low: f64,
        /// The latest supported.
        high: f64,
    },
    /// A history much shorter or longer than any in training.
    HistoryLength {
        /// `"tokens"` (observations and events) or `"visits"` (distinct
        /// observation times).
        measure: String,
        /// The subject's count.
        value: f64,
        /// The fewest supported.
        low: f64,
        /// The most supported.
        high: f64,
    },
    /// A learned state far from every training state.
    StateOutOfSupport {
        /// The subject's Mahalanobis distance.
        distance: f64,
        /// The largest supported distance.
        limit: f64,
    },
}

/// One subject's standing against the training support.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Assessment {
    /// `Some(true)` inside the support, `Some(false)` outside it, `None` when
    /// the model records no support (support unknown).
    pub supported: Option<bool>,
    /// The largest component score, `1.0` at the edge of the support; `None`
    /// with unknown support.
    pub ood_score: Option<f64>,
    /// Every component past the edge, most severe first.
    pub warnings: Vec<Warning>,
}

impl Assessment {
    /// Whether a subject with this assessment gets no probability under a
    /// threshold of `max_ood_score`. Unknown support never withholds: there
    /// is nothing to compare against, and the answer says so.
    pub fn abstains(&self, max_ood_score: f64) -> bool {
        self.ood_score.is_some_and(|s| s > max_ood_score)
    }

    /// The assessment of a model that records no support.
    pub fn unknown() -> Assessment {
        Assessment {
            supported: None,
            ood_score: None,
            warnings: Vec::new(),
        }
    }
}

/// The robust range of one quantity in the training set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Range {
    /// Values it was computed from.
    pub n: usize,
    /// Smallest value.
    pub min: f64,
    /// 0.5th percentile (the supported range's low end).
    pub q005: f64,
    /// 5th percentile.
    pub q05: f64,
    /// Median.
    pub q50: f64,
    /// 95th percentile.
    pub q95: f64,
    /// 99.5th percentile (the supported range's high end).
    pub q995: f64,
    /// Largest value.
    pub max: f64,
}

/// The `p` quantile of sorted `v`, linear between order statistics.
fn quantile(v: &[f64], p: f64) -> f64 {
    let pos = p * (v.len() - 1) as f64;
    let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + frac * (v[hi] - v[lo])
}

impl Range {
    fn of(mut values: Vec<f64>) -> Option<Range> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        Some(Range {
            n: values.len(),
            min: values[0],
            q005: quantile(&values, 0.005),
            q05: quantile(&values, 0.05),
            q50: quantile(&values, 0.5),
            q95: quantile(&values, 0.95),
            q995: quantile(&values, 0.995),
            max: values[values.len() - 1],
        })
    }

    /// The supported interval and the score of `x` against it: `1.0` at the
    /// interval's ends, `0.0` at its middle. `floor_width` widens a range
    /// narrower than it.
    fn judge(&self, x: f64, margin: f64, floor_width: f64) -> (f64, f64, f64) {
        let width = (self.q995 - self.q005).max(floor_width);
        let mid = 0.5 * (self.q995 + self.q005);
        let (low, high) = (mid - 0.5 * width - margin * width, mid + 0.5 * width + margin * width);
        let score = if width > 0.0 {
            (x - mid).abs() / (0.5 * width * (1.0 + 2.0 * margin))
        } else if x == mid {
            0.0
        } else {
            UNKNOWN_SCORE
        };
        (score, low, high)
    }
}

/// Mean and covariance of the learned state over the training subjects, and
/// the distribution of their Mahalanobis distances.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateSupport {
    /// State dimension.
    pub dim: usize,
    mean: Vec<f64>,
    /// Covariance, `dim * dim` row-major.
    cov: Vec<f64>,
    /// The training subjects' own distances.
    pub distance: Range,
    #[serde(skip)]
    cholesky: Vec<f64>,
}

/// The Cholesky factor `L` (row-major, lower) of `cov + ridge * I`, with the
/// ridge a fixed share of the mean variance so a state dimension the
/// training set never moved does not turn rounding noise into distance.
fn cholesky(cov: &[f64], dim: usize) -> Result<Vec<f64>, String> {
    let ridge = 1e-3 * (0..dim).map(|i| cov[i * dim + i]).sum::<f64>() / dim as f64 + 1e-12;
    let mut l = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = cov[i * dim + j] + if i == j { ridge } else { 0.0 };
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err("support: the state covariance is not positive definite".into());
                }
                l[i * dim + i] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }
    Ok(l)
}

impl StateSupport {
    fn new(mean: Vec<f64>, cov: Vec<f64>, states: &[Vec<f32>]) -> Result<StateSupport, String> {
        let dim = mean.len();
        let cholesky = cholesky(&cov, dim)?;
        let mut s = StateSupport {
            dim,
            mean,
            cov,
            distance: Range::of(vec![0.0]).expect("one value"),
            cholesky,
        };
        let distances: Vec<f64> = states.iter().map(|z| s.mahalanobis(z)).collect();
        s.distance = Range::of(distances).ok_or("support: no training states")?;
        Ok(s)
    }

    fn prepared(mut self) -> Result<StateSupport, String> {
        if self.mean.len() != self.dim || self.cov.len() != self.dim * self.dim || self.dim == 0 {
            return Err("support: the state statistics have inconsistent sizes".into());
        }
        if self.mean.iter().chain(&self.cov).any(|v| !v.is_finite()) {
            return Err("support: the state statistics are not finite".into());
        }
        self.cholesky = cholesky(&self.cov, self.dim)?;
        Ok(self)
    }

    /// The Mahalanobis distance of `z` from the training states.
    pub fn mahalanobis(&self, z: &[f32]) -> f64 {
        let d = self.dim;
        // Solve L y = z - mean by forward substitution; the distance is |y|.
        let mut y = vec![0.0; d];
        for i in 0..d {
            let solved: f64 = y[..i]
                .iter()
                .enumerate()
                .map(|(k, yk)| self.cholesky[i * d + k] * yk)
                .sum();
            y[i] = (f64::from(z[i]) - self.mean[i] - solved) / self.cholesky[i * d + i];
        }
        y.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

/// What the model was trained on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Support {
    version: u32,
    /// SHA-256 (hex) of the weights file the state statistics belong to.
    pub weights_sha256: String,
    /// Training subjects the statistics come from.
    pub subjects: usize,
    /// Numeric variable -> the range of its values (detection-limit values
    /// count at their limit). timeline-v1 carries no units, so none are
    /// recorded: a unit mix-up is caught by the range, not named.
    pub variables: BTreeMap<String, Range>,
    /// Categorical variable -> the levels seen.
    pub categories: BTreeMap<String, BTreeSet<String>>,
    /// History event codes seen, and the model's outcome codes.
    pub event_codes: BTreeSet<String>,
    /// Entry clock values (age, for a person).
    pub entry: Range,
    /// Calendar time at entry.
    pub calendar: Range,
    /// Known observations plus known events per subject.
    pub tokens: Range,
    /// Distinct observation times per subject (visits).
    pub visits: Range,
    /// The learned state's statistics; `None` when the training set was too
    /// small to pin them down.
    pub state: Option<StateSupport>,
}

fn history_len(s: &Subject) -> (f64, f64) {
    let times: BTreeSet<u64> = s.known_observations().map(|o| o.t.to_bits()).collect();
    (
        (s.known_observations().count() + s.known_events().count()) as f64,
        times.len() as f64,
    )
}

impl Support {
    /// Fit on the training subjects of `saved` (its model gives their
    /// states). `weights_sha256` is `saved`'s weights digest.
    pub fn fit(saved: &Saved, weights_sha256: String, train: &[Subject]) -> Result<Support, String> {
        if train.is_empty() {
            return Err("support: training subjects are required".into());
        }
        let mut numeric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut categories: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut event_codes: BTreeSet<String> = saved.vocab.codes.iter().cloned().collect();
        let (mut entry, mut calendar, mut tokens, mut visits) = (vec![], vec![], vec![], vec![]);
        for s in train {
            for o in s.known_observations() {
                match &o.value {
                    Value::Number(v) | Value::Below { below: v } | Value::Above { above: v } => {
                        numeric.entry(o.var.clone()).or_default().push(*v)
                    }
                    Value::Category(c) => {
                        categories.entry(o.var.clone()).or_default().insert(c.clone());
                    }
                }
            }
            event_codes.extend(s.known_events().map(|e| e.code.clone()));
            entry.push(s.entry);
            calendar.push(s.calendar_at_entry);
            let (t, v) = history_len(s);
            tokens.push(t);
            visits.push(v);
        }
        let dim = saved.model.cfg.d_model as usize;
        let state = if train.len() >= STATE_SUBJECTS_PER_DIM * dim {
            let states = predict_states(&saved.model, &saved.encode(train)?);
            let n = states.len() as f64;
            let mean: Vec<f64> = (0..dim)
                .map(|j| states.iter().map(|z| f64::from(z[j])).sum::<f64>() / n)
                .collect();
            let mut cov = vec![0.0; dim * dim];
            for z in &states {
                for i in 0..dim {
                    let di = f64::from(z[i]) - mean[i];
                    for j in 0..=i {
                        cov[i * dim + j] += di * (f64::from(z[j]) - mean[j]) / (n - 1.0);
                    }
                }
            }
            for i in 0..dim {
                for j in 0..i {
                    cov[j * dim + i] = cov[i * dim + j];
                }
            }
            Some(StateSupport::new(mean, cov, &states)?)
        } else {
            None
        };
        let range = |v: Vec<f64>| Range::of(v).expect("train is not empty");
        Ok(Support {
            version: VERSION,
            weights_sha256,
            subjects: train.len(),
            variables: numeric
                .into_iter()
                .filter_map(|(k, v)| Range::of(v).map(|r| (k, r)))
                .collect(),
            categories,
            event_codes,
            entry: range(entry),
            calendar: range(calendar),
            tokens: range(tokens),
            visits: range(visits),
            state,
        })
    }

    /// `subject`, whose learned state is `state` (`None` skips the state
    /// component), against this support.
    pub fn assess(&self, subject: &Subject, state: Option<&[f32]>, opts: &AssessOptions) -> Assessment {
        // Worst score per (warning kind, name): a variable measured often is
        // one warning, at its most extreme value.
        let mut worst: BTreeMap<(u8, String), (f64, Warning)> = BTreeMap::new();
        let mut note = |rank: u8, name: &str, score: f64, w: Warning| {
            let slot = worst.entry((rank, name.to_string())).or_insert((f64::MIN, w.clone()));
            if score > slot.0 {
                *slot = (score, w);
            }
        };
        for o in subject.known_observations() {
            match &o.value {
                Value::Category(level) => match self.categories.get(&o.var) {
                    None => note(0, &o.var, UNKNOWN_SCORE, Warning::UnknownVariable { var: o.var.clone() }),
                    Some(seen) if !seen.contains(level) => note(
                        1,
                        &format!("{}={level}", o.var),
                        UNKNOWN_SCORE,
                        Warning::UnknownCategory { var: o.var.clone(), level: level.clone() },
                    ),
                    Some(_) => {}
                },
                Value::Number(v) | Value::Below { below: v } | Value::Above { above: v } => {
                    match self.variables.get(&o.var) {
                        None => note(0, &o.var, UNKNOWN_SCORE, Warning::UnknownVariable { var: o.var.clone() }),
                        Some(r) => {
                            let (score, low, high) = r.judge(*v, opts.margin, 0.0);
                            note(2, &o.var, score, Warning::ValueOutOfRange { var: o.var.clone(), value: *v, low, high });
                        }
                    }
                }
            }
        }
        for e in subject.known_events() {
            if !self.event_codes.contains(&e.code) {
                note(3, &e.code, UNKNOWN_SCORE, Warning::UnknownEventCode { code: e.code.clone() });
            }
        }
        let (score, low, high) = self.entry.judge(subject.entry, opts.margin, 0.0);
        note(4, "entry", score, Warning::EntryOutOfRange { value: subject.entry, low, high });
        let (score, low, high) = self.calendar.judge(subject.calendar_at_entry, opts.margin, 0.0);
        note(5, "calendar", score, Warning::CalendarOutOfRange { value: subject.calendar_at_entry, low, high });
        let (tokens, visits) = history_len(subject);
        for (measure, value, range) in [("tokens", tokens, &self.tokens), ("visits", visits, &self.visits)] {
            let (score, low, high) = range.judge(value, opts.margin, COUNT_FLOOR_WIDTH);
            note(6, measure, score, Warning::HistoryLength { measure: measure.into(), value, low, high });
        }
        if let (Some(ss), Some(z)) = (&self.state, state) {
            let limit = ss.distance.q995.max(f64::MIN_POSITIVE) * (1.0 + opts.state_margin);
            let distance = ss.mahalanobis(z);
            note(7, "state", distance / limit, Warning::StateOutOfSupport { distance, limit });
        }
        let mut scored: Vec<(f64, Warning)> = worst.into_values().collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let ood_score = scored.first().map_or(0.0, |s| s.0).max(0.0);
        Assessment {
            supported: Some(ood_score <= 1.0),
            ood_score: Some(ood_score),
            warnings: scored.into_iter().filter(|s| s.0 > 1.0).map(|s| s.1).collect(),
        }
    }

    /// Serialise for [`FILE`].
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("support: {e}"))
    }

    /// Parse [`FILE`]'s text.
    pub fn from_json(text: &str) -> Result<Support, String> {
        let mut s: Support = serde_json::from_str(text).map_err(|e| format!("support: {e}"))?;
        if s.version != VERSION {
            return Err(format!(
                "support: format version {} is not supported (this build reads {VERSION})",
                s.version
            ));
        }
        s.state = s.state.map(StateSupport::prepared).transpose()?;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(lo: f64, hi: f64) -> Range {
        Range::of((0..=1000).map(|i| lo + (hi - lo) * i as f64 / 1000.0).collect()).unwrap()
    }

    #[test]
    fn a_range_scores_one_at_its_margin_and_zero_at_its_middle() {
        let r = Range { q005: 0.0, q995: 10.0, ..range(0.0, 10.0) };
        let (mid, _, _) = r.judge(5.0, 0.25, 0.0);
        assert_eq!(mid, 0.0);
        // Width 10, margin 0.25: supported up to 12.5 and down to -2.5.
        let (edge, low, high) = r.judge(12.5, 0.25, 0.0);
        assert!((edge - 1.0).abs() < 1e-12 && low == -2.5 && high == 12.5);
        assert!(r.judge(12.6, 0.25, 0.0).0 > 1.0 && r.judge(-2.6, 0.25, 0.0).0 > 1.0);
        assert!(r.judge(7.0, 0.25, 0.0).0 < 1.0);
        // A constant quantity supports only its value; a floor widens it.
        let constant = Range { q005: 3.0, q995: 3.0, ..range(3.0, 3.0) };
        assert_eq!(constant.judge(3.0, 0.25, 0.0).0, 0.0);
        assert_eq!(constant.judge(3.1, 0.25, 0.0).0, UNKNOWN_SCORE);
        assert!(constant.judge(4.0, 0.25, 4.0).0 < 1.0, "within the floor's margin");
    }

    #[test]
    fn the_mahalanobis_distance_scales_by_the_covariance() {
        // Variances 4 and 1, no correlation: one unit of distance is 2 along
        // the first axis and 1 along the second.
        let states = vec![vec![0.0f32, 0.0]; 3];
        let s = StateSupport::new(vec![0.0, 0.0], vec![4.0, 0.0, 0.0, 1.0], &states).unwrap();
        assert!((s.mahalanobis(&[2.0, 0.0]) - 1.0).abs() < 1e-2);
        assert!((s.mahalanobis(&[0.0, 3.0]) - 3.0).abs() < 1e-2);
        assert!((s.mahalanobis(&[2.0, 1.0]) - 2f64.sqrt()).abs() < 1e-2);
        // Correlated: along the correlated direction the distance is small.
        let c = StateSupport::new(vec![0.0, 0.0], vec![1.0, 0.9, 0.9, 1.0], &states).unwrap();
        assert!(c.mahalanobis(&[1.0, 1.0]) < c.mahalanobis(&[1.0, -1.0]));
        assert!(StateSupport::new(vec![0.0], vec![f64::NAN], &states).is_err());
    }
}
