// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Latency statistics: exact percentiles over a recorded sample, plus the
//! best-of-N + spread bookkeeping the suite reports instead of a bare mean.
//!
//! Samples are kept in full rather than histogrammed. A benchmark run holds tens
//! of thousands of latencies at most (one per request, or one per emitted
//! artifact), so exact percentiles cost nothing and avoid the bucket-boundary
//! lies an approximate histogram tells at P99.9 — precisely the tail the suite
//! exists to report.

use serde_json::{json, Value};

/// A latency distribution in milliseconds.
#[derive(Clone, Debug, Default)]
pub struct Dist {
    samples: Vec<f64>,
    sorted: bool,
}

impl Dist {
    pub fn new() -> Dist {
        Dist { samples: Vec::new(), sorted: true }
    }

    pub fn from_millis(v: Vec<f64>) -> Dist {
        Dist { samples: v, sorted: false }
    }

    pub fn push(&mut self, ms: f64) {
        self.samples.push(ms);
        self.sorted = false;
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn sort(&mut self) {
        if !self.sorted {
            self.samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
    }

    /// Nearest-rank percentile (`q` in 0..=1). `None` for an empty sample.
    pub fn percentile(&mut self, q: f64) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        self.sort();
        let n = self.samples.len();
        // Nearest-rank: rank = ceil(q*n), clamped to 1..=n.
        let rank = (q * n as f64).ceil().max(1.0).min(n as f64) as usize;
        Some(self.samples[rank - 1])
    }

    pub fn mean(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.samples.iter().sum::<f64>() / self.samples.len() as f64)
    }

    pub fn min(&mut self) -> Option<f64> {
        self.sort();
        self.samples.first().copied()
    }

    pub fn max(&mut self) -> Option<f64> {
        self.sort();
        self.samples.last().copied()
    }

    /// The share of samples at or below `ms` — used for SLO attainment.
    pub fn fraction_under(&self, ms: f64) -> f64 {
        if self.samples.is_empty() {
            return 1.0;
        }
        let n = self.samples.iter().filter(|&&s| s <= ms).count();
        n as f64 / self.samples.len() as f64
    }

    /// The standard `{p50,p95,p99,p999,mean,min,max,n}` block. `null` throughout
    /// for an empty sample — "not measured" must stay distinguishable from zero.
    pub fn to_json(&mut self) -> Value {
        if self.samples.is_empty() {
            return json!({
                "p50": Value::Null, "p95": Value::Null, "p99": Value::Null,
                "p999": Value::Null, "mean": Value::Null,
                "min": Value::Null, "max": Value::Null, "n": 0
            });
        }
        json!({
            "p50": r3(self.percentile(0.50).unwrap()),
            "p95": r3(self.percentile(0.95).unwrap()),
            "p99": r3(self.percentile(0.99).unwrap()),
            "p999": r3(self.percentile(0.999).unwrap()),
            "mean": r3(self.mean().unwrap()),
            "min": r3(self.min().unwrap()),
            "max": r3(self.max().unwrap()),
            "n": self.samples.len(),
        })
    }
}

/// Round to 3 decimals so artifacts diff cleanly.
pub fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Best-of-N over repeated runs of the same measurement, with the observed
/// spread. The suite reports both: the best value is the stable signal on a
/// shared or thermally-throttled box, and the spread says how much to trust it.
#[derive(Clone, Debug, Default)]
pub struct BestOf {
    values: Vec<f64>,
    higher_is_better: bool,
}

impl BestOf {
    pub fn higher_better() -> BestOf {
        BestOf { values: Vec::new(), higher_is_better: true }
    }
    pub fn lower_better() -> BestOf {
        BestOf { values: Vec::new(), higher_is_better: false }
    }
    pub fn push(&mut self, v: f64) {
        self.values.push(v);
    }
    pub fn n(&self) -> usize {
        self.values.len()
    }
    pub fn best(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        let mut it = self.values.iter().copied();
        let first = it.next().unwrap();
        Some(it.fold(first, |acc, v| if self.higher_is_better { acc.max(v) } else { acc.min(v) }))
    }
    /// `(max - min) / max` as a percentage — how noisy the box was.
    pub fn spread_pct(&self) -> Option<f64> {
        if self.values.len() < 2 {
            return None;
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &self.values {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if hi <= 0.0 {
            return Some(0.0);
        }
        Some(r3((hi - lo) / hi * 100.0))
    }
}

/// Jain's fairness index over per-class (or per-model) throughputs: 1.0 is
/// perfectly fair, 1/n is maximally unfair.
pub fn jain_fairness(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let n = xs.len() as f64;
    let sum: f64 = xs.iter().sum();
    let sq: f64 = xs.iter().map(|x| x * x).sum();
    if sq <= 0.0 {
        return Some(1.0);
    }
    Some(r3(sum * sum / (n * sq)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_are_nearest_rank() {
        let mut d = Dist::from_millis((1..=100).map(|i| i as f64).collect());
        // Nearest-rank: p50 -> rank 50, p99 -> rank 99.
        assert_eq!(d.percentile(0.50), Some(50.0));
        assert_eq!(d.percentile(0.95), Some(95.0));
        assert_eq!(d.percentile(0.99), Some(99.0));
        assert_eq!(d.min(), Some(1.0));
        assert_eq!(d.max(), Some(100.0));
    }

    #[test]
    fn p999_reports_the_real_tail() {
        // 999 fast samples and one 10s outlier: the mean hides it, p999 must not.
        let mut v: Vec<f64> = vec![1.0; 999];
        v.push(10_000.0);
        let mut d = Dist::from_millis(v);
        assert_eq!(d.percentile(0.999), Some(1.0));
        assert_eq!(d.max(), Some(10_000.0));
        assert!(d.mean().unwrap() < 12.0, "mean hides the outlier, as expected");
    }

    #[test]
    fn empty_dist_is_null_not_zero() {
        let mut d = Dist::new();
        let j = d.to_json();
        assert!(j["p50"].is_null());
        assert_eq!(j["n"], 0);
    }

    #[test]
    fn fraction_under_drives_slo_attainment() {
        let d = Dist::from_millis(vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(d.fraction_under(25.0), 0.5);
        assert_eq!(d.fraction_under(100.0), 1.0);
    }

    #[test]
    fn best_of_picks_the_right_end() {
        let mut hi = BestOf::higher_better();
        hi.push(100.0);
        hi.push(120.0);
        hi.push(90.0);
        assert_eq!(hi.best(), Some(120.0));
        assert_eq!(hi.spread_pct(), Some(25.0));

        let mut lo = BestOf::lower_better();
        lo.push(100.0);
        lo.push(120.0);
        assert_eq!(lo.best(), Some(100.0));
    }

    #[test]
    fn jain_is_one_when_equal() {
        assert_eq!(jain_fairness(&[5.0, 5.0, 5.0]), Some(1.0));
        // One class getting everything and two starving is maximal unfairness:
        // 1/n. Values are rounded to 3dp for stable artifact diffs.
        let unfair = jain_fairness(&[10.0, 0.0, 0.0]).unwrap();
        assert!((unfair - 1.0 / 3.0).abs() < 1e-3, "got {unfair}");
    }
}
