// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The token vocabulary and the per-variable value transform, fitted on the
//! TRAINING subjects only and saved beside the weights.
//!
//! A token names what was observed, never its value:
//! - `v:<var>` a numeric variable (its value enters through soft bins);
//! - `c:<var>=<level>` a categorical level seen at least `min_count` times,
//!   `c:<var>=?` every rarer or unseen level of that variable;
//! - `e:<code>` an event in the subject's history (before entry);
//! - `?` anything the fitted vocabulary has never seen.
//!
//! A numeric value is mapped to its empirical CDF `u` in `[0, 1]` against
//! quantile knots of the training values, so units and skew stop mattering
//! to the encoder, and to a normal score `Phi^-1(u)` as the measurement head's
//! target. Ties (a binary variable, a value reported to one decimal) map to
//! the midpoint of their run of knots: the mid-rank, not the first rank.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::timeline::{Subject, Value};

/// Reserved token ids.
pub const CLS: u32 = 0;
/// Padding (never a key in attention, never in a loss).
pub const PAD: u32 = 1;
/// A token the vocabulary does not know.
pub const UNKNOWN: u32 = 2;
const RESERVED: [&str; 3] = ["<cls>", "<pad>", "?"];

/// Fitting options.
#[derive(Clone, Debug)]
pub struct FitOptions {
    /// Quantile knots per numeric variable.
    pub knots: usize,
    /// A categorical level needs this many training subjects to get a token.
    pub min_count: usize,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            knots: 65,
            min_count: 20,
        }
    }
}

/// The fitted vocabulary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vocab {
    /// Token id -> name.
    pub tokens: Vec<String>,
    /// Numeric variable -> its sorted quantile knots.
    pub knots: BTreeMap<String, Vec<f64>>,
    /// Outcome codes the hazard head predicts, index = code id.
    pub codes: Vec<String>,
    /// Outcome codes that end observation of every code (death).
    pub absorbing: Vec<String>,
    /// Event codes whose FIRST occurrence after the prediction time, among
    /// them, is modelled by extra hazard columns after `codes`: the
    /// self-supervised next-event objective. Empty for a pure outcome model.
    #[serde(default)]
    pub next_events: Vec<String>,
    /// Numeric variable -> the unit its training values were stated in. A
    /// variable no training record gave a unit for is absent: unitless. A
    /// vocabulary saved before units existed has none, so every variable of
    /// it is unitless.
    #[serde(default)]
    pub units: BTreeMap<String, String>,
    #[serde(skip)]
    index: HashMap<String, u32>,
}

impl Vocab {
    /// Fit on training subjects. `codes` are the outcome codes to predict and
    /// `absorbing` the subset that ends follow-up for all of them.
    pub fn fit(
        subjects: &[Subject],
        codes: &[String],
        absorbing: &[String],
        opts: &FitOptions,
    ) -> Result<Vocab, String> {
        if codes.is_empty() {
            return Err("vocab: at least one outcome code is required".into());
        }
        if let Some(a) = absorbing.iter().find(|a| !codes.contains(a)) {
            return Err(format!("vocab: absorbing code {a} is not an outcome code"));
        }
        let mut numeric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let mut levels: BTreeMap<String, usize> = BTreeMap::new();
        let mut cat_vars: BTreeMap<String, ()> = BTreeMap::new();
        let mut history: BTreeMap<String, ()> = BTreeMap::new();
        let mut units: BTreeMap<String, String> = BTreeMap::new();
        for s in subjects {
            // Every observation states its unit, including the ones after
            // entry that only the forecast head reads.
            for o in &s.observations {
                let (Some(unit), false) = (&o.unit, matches!(o.value, Value::Category(_))) else {
                    continue;
                };
                match units.get(&o.var) {
                    Some(first) if first != unit => {
                        return Err(format!(
                            "vocab: variable {} is stated in unit {first:?} and in unit {unit:?} (subject {}): \
                             convert the training data to one unit",
                            o.var, s.subject_id
                        ))
                    }
                    Some(_) => {}
                    None => {
                        units.insert(o.var.clone(), unit.clone());
                    }
                }
            }
            let mut seen_level: BTreeMap<String, ()> = BTreeMap::new();
            for o in s.known_observations() {
                match &o.value {
                    Value::Number(v) | Value::Below { below: v } | Value::Above { above: v } => {
                        numeric.entry(o.var.clone()).or_default().push(*v)
                    }
                    Value::Category(c) => {
                        cat_vars.insert(o.var.clone(), ());
                        seen_level.insert(format!("c:{}={}", o.var, c), ());
                    }
                }
            }
            for l in seen_level.into_keys() {
                *levels.entry(l).or_default() += 1;
            }
            for e in s.known_events() {
                history.insert(format!("e:{}", e.code), ());
            }
        }
        let numeric_vars: std::collections::BTreeSet<String> = numeric.keys().cloned().collect();
        let mut tokens: Vec<String> = RESERVED.iter().map(|s| s.to_string()).collect();
        let mut knots = BTreeMap::new();
        for (var, mut vals) in numeric {
            vals.sort_by(|a, b| a.partial_cmp(b).expect("validated finite"));
            knots.insert(var.clone(), quantile_knots(&vals, opts.knots));
            tokens.push(format!("v:{var}"));
        }
        for var in cat_vars.keys() {
            tokens.push(format!("c:{var}=?"));
        }
        tokens.extend(
            levels
                .into_iter()
                .filter(|(_, n)| *n >= opts.min_count)
                .map(|(l, _)| l),
        );
        tokens.extend(history.into_keys());
        let mut v = Vocab {
            tokens,
            knots,
            codes: codes.to_vec(),
            absorbing: absorbing.to_vec(),
            next_events: Vec::new(),
            units: units.into_iter().filter(|(var, _)| numeric_vars.contains(var)).collect(),
            index: HashMap::new(),
        };
        v.reindex();
        Ok(v)
    }

    /// The same vocabulary with a next-event group: `events` are event codes
    /// (not necessarily outcome codes) that compete with one another for being
    /// the next to happen, each owning a hazard column after the outcome
    /// columns. A subject's death, if it is an outcome, ends follow-up for the
    /// group as it does for every code.
    pub fn with_next_events(mut self, events: &[String]) -> Result<Vocab, String> {
        if events.is_empty() {
            return Err("vocab: a next-event group needs at least one event code".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        if let Some(d) = events.iter().find(|e| !seen.insert(e.as_str())) {
            return Err(format!("vocab: next-event code {d} is listed twice"));
        }
        self.next_events = events.to_vec();
        Ok(self)
    }

    /// Hazard columns: the outcome codes, then the next-event group.
    pub fn head_codes(&self) -> usize {
        self.codes.len() + self.next_events.len()
    }

    /// Rebuild the name -> id index (after deserialising).
    pub fn reindex(&mut self) {
        self.index = self
            .tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
    }

    /// Load from JSON.
    pub fn from_json(text: &str) -> Result<Vocab, String> {
        let mut v: Vocab = serde_json::from_str(text).map_err(|e| format!("vocab: {e}"))?;
        v.reindex();
        Ok(v)
    }

    /// Number of token ids.
    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Always false: the reserved tokens are always present.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The token id of a numeric variable.
    pub fn numeric_token(&self, var: &str) -> u32 {
        self.id(&format!("v:{var}"))
    }

    /// The token id of a categorical level (its variable's `?` level when the
    /// level is rare or unseen; [`UNKNOWN`] when the variable is unseen).
    pub fn category_token(&self, var: &str, level: &str) -> u32 {
        self.index
            .get(&format!("c:{var}={level}"))
            .copied()
            .unwrap_or_else(|| self.id(&format!("c:{var}=?")))
    }

    /// The token id of a history event.
    pub fn event_token(&self, code: &str) -> u32 {
        self.id(&format!("e:{code}"))
    }

    fn id(&self, name: &str) -> u32 {
        self.index.get(name).copied().unwrap_or(UNKNOWN)
    }

    /// Outcome code id.
    pub fn code_id(&self, code: &str) -> Option<usize> {
        self.codes.iter().position(|c| c == code)
    }

    /// Refuse a subject with a measurement stated in a unit other than the one
    /// the model was trained on. The error names the variable, the subject and
    /// both units; nothing is converted (horizon has no conversion table) and
    /// nothing is guessed. A measurement with no unit, or of a variable the
    /// model records as unitless, passes: see [`crate::support::unit_advisories`]
    /// for the second.
    pub fn check_units(&self, subject: &Subject) -> Result<(), String> {
        for o in &subject.observations {
            if let (Some(given), Some(canonical)) = (&o.unit, self.units.get(&o.var)) {
                if given != canonical {
                    return Err(format!(
                        "subject {}: observation of {} at {} is stated in unit {given:?} but the model was trained on {canonical:?}: \
                         convert it before predicting (horizon never converts units)",
                        subject.subject_id, o.var, o.t
                    ));
                }
            }
        }
        Ok(())
    }

    /// The empirical CDF of `value` for numeric `var`, in `[0, 1]`; `None`
    /// for a variable the vocabulary has no knots for.
    pub fn cdf(&self, var: &str, value: f64) -> Option<f64> {
        self.knots.get(var).map(|k| empirical_cdf(k, value))
    }
}

impl Vocab {
    /// The `q` quantile, in `var`'s own unit, of a forecast `(mu, sigma)` on
    /// its normal-score scale: the normal quantile mapped back through the
    /// variable's empirical distribution.
    pub fn forecast_quantile(&self, var: &str, mu: f64, sigma: f64, q: f64) -> Option<f64> {
        let y = mu + sigma * model::hostmath::ndtri(q);
        self.value_at(var, (model::hostmath::log_ndtr(y as f32) as f64).exp())
    }

    /// The value of numeric `var` at empirical CDF `u` (the inverse of
    /// [`Vocab::cdf`]: linear between knots, clamped to the observed range).
    pub fn value_at(&self, var: &str, u: f64) -> Option<f64> {
        let k = self.knots.get(var)?;
        let pos = u.clamp(0.0, 1.0) * (k.len() - 1) as f64;
        let lo = (pos.floor() as usize).min(k.len() - 1);
        let hi = (lo + 1).min(k.len() - 1);
        Some(k[lo] + (pos - lo as f64) * (k[hi] - k[lo]))
    }
}

/// `n` knots at evenly spaced probabilities over sorted `vals` (linear
/// interpolation between order statistics).
fn quantile_knots(sorted: &[f64], n: usize) -> Vec<f64> {
    let n = n.max(2);
    let last = (sorted.len() - 1) as f64;
    (0..n)
        .map(|i| {
            let pos = last * i as f64 / (n - 1) as f64;
            let (lo, frac) = (pos.floor() as usize, pos - pos.floor());
            let hi = (lo + 1).min(sorted.len() - 1);
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        })
        .collect()
}

/// Position of `v` among `knots`, in `[0, 1]`, mid-rank over ties.
fn empirical_cdf(knots: &[f64], v: f64) -> f64 {
    let last = (knots.len() - 1) as f64;
    if v < knots[0] {
        return 0.0;
    }
    if v > knots[knots.len() - 1] {
        return 1.0;
    }
    let lo = knots.partition_point(|&k| k < v); // first knot >= v
    let hi = knots.partition_point(|&k| k <= v); // one past the last knot <= v
    if hi > lo {
        // v equals knots[lo..hi]: the mid-rank of the run.
        return (lo + hi - 1) as f64 / 2.0 / last;
    }
    // knots[lo - 1] < v < knots[lo]
    let (a, b) = (knots[lo - 1], knots[lo]);
    ((lo - 1) as f64 + (v - a) / (b - a)) / last
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(id: &str, sbp: f64, smoke: &str) -> Subject {
        Subject::from_json_line(&format!(
            r#"{{"subject_id":"{id}","source":"s","entry":50,"calendar_at_entry":2000,
            "observations":[{{"t":50,"var":"sbp","value":{sbp}}},{{"t":50,"var":"smoking","value":"{smoke}"}}],
            "events":[{{"t":40,"code":"dx:diabetes"}}]}}"#
        ))
        .unwrap()
    }

    #[test]
    fn fitting_assigns_tokens_and_routes_rare_levels_to_the_variable_bucket() {
        let mut subjects: Vec<Subject> = (0..30)
            .map(|i| subject(&i.to_string(), 100.0 + i as f64, "never"))
            .collect();
        subjects.push(subject("rare", 120.0, "pipe"));
        let codes = vec!["death".to_string()];
        let v = Vocab::fit(&subjects, &codes, &codes, &FitOptions::default()).unwrap();
        assert_eq!(v.tokens[..3], ["<cls>", "<pad>", "?"]);
        assert_ne!(v.numeric_token("sbp"), UNKNOWN);
        assert_ne!(
            v.category_token("smoking", "never"),
            v.category_token("smoking", "pipe")
        );
        assert_eq!(
            v.category_token("smoking", "pipe"),
            v.category_token("smoking", "cigar"),
            "rare and unseen share the bucket"
        );
        assert_eq!(v.category_token("hair", "red"), UNKNOWN);
        assert_ne!(v.event_token("dx:diabetes"), UNKNOWN);
        let back = Vocab::from_json(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(back.numeric_token("sbp"), v.numeric_token("sbp"));
    }

    #[test]
    fn the_cdf_is_monotone_clamped_and_mid_ranks_ties() {
        let k = quantile_knots(&[0.0, 0.0, 0.0, 1.0, 1.0], 5);
        assert_eq!(k, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            empirical_cdf(&k, 0.0),
            0.25,
            "the mid-rank of three tied knots out of five"
        );
        assert_eq!(empirical_cdf(&k, 1.0), 0.875);
        assert_eq!(empirical_cdf(&k, -1.0), 0.0);
        assert_eq!(empirical_cdf(&k, 2.0), 1.0);
        let lin = quantile_knots(&[0.0, 1.0, 2.0, 3.0, 4.0], 5);
        assert!((empirical_cdf(&lin, 2.5) - 0.625).abs() < 1e-12);
        let mut prev = -1.0;
        for i in 0..=40 {
            let u = empirical_cdf(&lin, -0.5 + i as f64 * 0.125);
            assert!(u >= prev, "monotone");
            prev = u;
        }
    }

    #[test]
    fn value_at_inverts_the_cdf_between_distinct_knots() {
        let subjects: Vec<Subject> = (0..50)
            .map(|i| subject(&i.to_string(), 100.0 + i as f64, "never"))
            .collect();
        let codes = vec!["death".to_string()];
        let v = Vocab::fit(&subjects, &codes, &codes, &FitOptions::default()).unwrap();
        for x in [100.0, 112.5, 131.0, 149.0] {
            let back = v.value_at("sbp", v.cdf("sbp", x).unwrap()).unwrap();
            assert!((back - x).abs() < 1e-9, "{x} -> {back}");
        }
        assert_eq!(v.value_at("sbp", 2.0), Some(149.0), "clamped to the range");
        assert_eq!(v.value_at("nope", 0.5), None);
    }

    /// `sbp` stated in `unit` (None: no unit), plus an unitless `age`.
    fn with_unit(id: &str, unit: Option<&str>) -> Subject {
        let unit = unit.map_or(String::new(), |u| format!(r#","unit":"{u}""#));
        Subject::from_json_line(&format!(
            r#"{{"subject_id":"{id}","source":"s","entry":50,"calendar_at_entry":2000,
            "observations":[{{"t":50,"var":"sbp","value":120{unit}}},{{"t":50,"var":"age","value":50}}]}}"#
        ))
        .unwrap()
    }

    fn fit(subjects: &[Subject]) -> Result<Vocab, String> {
        let codes = vec!["death".to_string()];
        Vocab::fit(subjects, &codes, &codes, &FitOptions::default())
    }

    #[test]
    fn the_canonical_unit_is_recorded_per_variable_and_a_disagreement_is_a_fit_error() {
        let v = fit(&[with_unit("a", Some("mmHg")), with_unit("b", None), with_unit("c", Some("mmHg"))]).unwrap();
        assert_eq!(v.units.get("sbp").map(String::as_str), Some("mmHg"));
        assert!(!v.units.contains_key("age"), "a variable that states no unit stays unitless");
        let err = fit(&[with_unit("a", Some("mmHg")), with_unit("b", Some("kPa"))]).unwrap_err();
        assert!(err.contains("sbp") && err.contains("mmHg") && err.contains("kPa"), "{err}");
    }

    #[test]
    fn a_vocabulary_without_units_loads_as_unitless() {
        let v = fit(&[with_unit("a", Some("mmHg"))]).unwrap();
        let mut json: serde_json::Value = serde_json::to_value(&v).unwrap();
        json.as_object_mut().unwrap().remove("units");
        let old = Vocab::from_json(&json.to_string()).unwrap();
        assert!(old.units.is_empty());
        assert!(old.check_units(&with_unit("x", Some("kPa"))).is_ok(), "nothing to compare against");
    }

    #[test]
    fn a_unit_other_than_the_canonical_one_is_rejected_by_name_never_converted() {
        let v = fit(&[with_unit("a", Some("mmHg"))]).unwrap();
        assert!(v.check_units(&with_unit("ok", Some("mmHg"))).is_ok());
        assert!(v.check_units(&with_unit("ok", None)).is_ok(), "no stated unit is not a mismatch");
        let err = v.check_units(&with_unit("p7", Some("kPa"))).unwrap_err();
        assert!(err.contains("sbp") && err.contains("p7") && err.contains("kPa") && err.contains("mmHg"), "{err}");
    }

    #[test]
    fn an_absorbing_code_must_be_an_outcome() {
        let codes = vec!["death".to_string()];
        assert!(Vocab::fit(&[], &codes, &["other".to_string()], &FitOptions::default()).is_err());
    }
}
