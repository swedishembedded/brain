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
        for s in subjects {
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
            index: HashMap::new(),
        };
        v.reindex();
        Ok(v)
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

    /// The empirical CDF of `value` for numeric `var`, in `[0, 1]`; `None`
    /// for a variable the vocabulary has no knots for.
    pub fn cdf(&self, var: &str, value: f64) -> Option<f64> {
        self.knots.get(var).map(|k| empirical_cdf(k, value))
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
    fn an_absorbing_code_must_be_an_outcome() {
        let codes = vec!["death".to_string()];
        assert!(Vocab::fit(&[], &codes, &["other".to_string()], &FitOptions::default()).is_err());
    }
}
