// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `residency` — many models, one engine, not enough memory. **The headline.**
//!
//! Serving a catalogue far larger than device memory from a *single* engine is
//! brain's clearest architectural differentiator: `crates/residency` tiers
//! weights across GPU/RAM/disk by LRU inside a budget and schedules jobs across
//! per-device lanes. The usual alternative is one process per model plus an
//! external router, which cannot batch across models, cannot share a budget, and
//! turns every model switch into a cold process start.
//!
//! So the workload is deliberately hostile:
//!
//! ```text
//! device budget:  B
//! catalogue:      N models, total 3-5x B     (cannot all be resident)
//! popularity:     Zipf(alpha)                 (a hot head, a long cold tail)
//! traffic shift:  popularity re-rolled mid-run (the cache must re-converge)
//! ```
//!
//! The number that matters is not the hit rate — a hit rate is easy to make look
//! good by serving only the hot head. It is **aggregate goodput across all
//! models together with per-model fairness**: a scheduler that pins the top
//! three models and starves the tail posts a fine hit rate and fails the tail
//! completely.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::stats::{jain_fairness, r3, Dist};

/// A synthetic model catalogue entry.
#[derive(Clone, Debug)]
pub struct CatalogEntry {
    pub name: String,
    /// Bytes resident when hot.
    pub bytes: u64,
    /// Milliseconds to activate (weight load + upload), proportional to size.
    pub load_ms: f64,
    /// Relative popularity weight.
    pub weight: f64,
}

/// Build a catalogue of `n` models whose total size is `over` times `budget`,
/// with Zipf-distributed popularity.
pub fn catalogue(n: usize, budget: u64, over: f64, alpha: f64) -> Vec<CatalogEntry> {
    let n = n.max(1);
    let total = (budget as f64 * over).max(1.0);
    let each = total / n as f64;
    (0..n)
        .map(|i| {
            // Sizes vary around the mean so eviction has real choices to make;
            // a catalogue of identical models makes any policy look the same.
            let scale = 0.5 + 1.0 * ((i % 4) as f64 / 3.0);
            let bytes = (each * scale) as u64;
            CatalogEntry {
                name: format!("model{i:02}"),
                bytes,
                // ~1 GB/s activation, floored so tiny models still cost something.
                load_ms: (bytes as f64 / 1e6).max(5.0),
                weight: 1.0 / ((i + 1) as f64).powf(alpha),
            }
        })
        .collect()
}

/// Draw `n` model indices from the catalogue's Zipf popularity.
pub fn draw(catalog: &[CatalogEntry], n: usize, rng: &mut data::rng::Rng) -> Vec<usize> {
    let total: f64 = catalog.iter().map(|c| c.weight).sum();
    (0..n)
        .map(|_| {
            let mut pick = rng.next_f64() * total;
            for (i, c) in catalog.iter().enumerate() {
                pick -= c.weight;
                if pick <= 0.0 {
                    return i;
                }
            }
            catalog.len() - 1
        })
        .collect()
}

/// Re-roll popularity: rotate the weights so a different set of models is hot.
/// This is what makes the benchmark about *adaptation* rather than about a
/// static working set.
pub fn shift_popularity(catalog: &mut [CatalogEntry], by: usize) {
    let weights: Vec<f64> = catalog.iter().map(|c| c.weight).collect();
    let n = weights.len();
    for (i, c) in catalog.iter_mut().enumerate() {
        c.weight = weights[(i + by) % n];
    }
}

/// One served request.
#[derive(Clone, Debug)]
pub struct Served {
    pub model: usize,
    /// True when the model was already resident.
    pub warm: bool,
    /// Time to first artifact, including any activation.
    pub ttfa_ms: f64,
    /// Time spent activating the model, if it was cold.
    pub load_ms: f64,
    /// Time this request spent blocked behind *another* model's activation.
    pub blocked_ms: f64,
}

/// One eviction: what it cost and whether it turned out to be wanted again.
#[derive(Clone, Debug)]
pub struct Eviction {
    /// Bytes evicted — reload cost is proportional to this, which is why the
    /// bytes-weighted regret is the number that matters: evicting a 200 MB
    /// model that comes back is a nuisance, evicting a 4 GB one is an outage.
    pub bytes: u64,
    /// How long until that model was next requested (ms).
    /// `None` = never requested again — a correct eviction.
    pub until_rerequest_ms: Option<f64>,
}

/// Aggregate outcome.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub served: Vec<Served>,
    pub evictions: Vec<Eviction>,
    pub wall_s: f64,
    pub models: usize,
    pub budget_bytes: u64,
    pub catalogue_bytes: u64,
}

impl Report {
    pub fn warm_ttfa(&self) -> Dist {
        Dist::from_millis(self.served.iter().filter(|s| s.warm).map(|s| s.ttfa_ms).collect())
    }
    pub fn cold_ttfa(&self) -> Dist {
        Dist::from_millis(self.served.iter().filter(|s| !s.warm).map(|s| s.ttfa_ms).collect())
    }
    pub fn weight_cache_hit_rate(&self) -> Option<f64> {
        (!self.served.is_empty())
            .then(|| self.served.iter().filter(|s| s.warm).count() as f64 / self.served.len() as f64)
    }
    /// Evictions of a model that was wanted again almost immediately —
    /// event-counted (every eviction weighs the same).
    pub fn eviction_regret(&self) -> Option<f64> {
        if self.evictions.is_empty() {
            return None;
        }
        let bad = self.evictions.iter().filter(|e| Self::regretted(e)).count();
        Some(bad as f64 / self.evictions.len() as f64)
    }

    /// Regret weighted by the bytes each eviction throws away. This is the
    /// number a cost-aware policy is supposed to move: at high overcommit
    /// *some* evictions are inevitable (event counts saturate), but a good
    /// policy makes the regretted ones the CHEAP ones.
    pub fn eviction_regret_bytes(&self) -> Option<f64> {
        let total: u64 = self.evictions.iter().map(|e| e.bytes).sum();
        if total == 0 {
            return None;
        }
        let bad: u64 = self.evictions.iter().filter(|e| Self::regretted(e)).map(|e| e.bytes).sum();
        Some(bad as f64 / total as f64)
    }

    fn regretted(e: &Eviction) -> bool {
        matches!(e.until_rerequest_ms, Some(ms) if ms <= 30_000.0)
    }
    /// Requests per second, per model — the input to fairness.
    pub fn per_model_rate(&self) -> Vec<f64> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for s in &self.served {
            *counts.entry(s.model).or_default() += 1;
        }
        (0..self.models).map(|i| counts.get(&i).copied().unwrap_or(0) as f64 / self.wall_s.max(1e-9)).collect()
    }
    /// Fairness across the whole catalogue, including models never served.
    ///
    /// Computed over *every* model, not just the ones that got traffic: a
    /// scheduler that pins the hot head and starves the tail must not be able to
    /// hide the tail by omitting it.
    pub fn fairness(&self) -> Option<f64> {
        jain_fairness(&self.per_model_rate())
    }
    /// Total time requests spent waiting behind another model being loaded.
    pub fn blocked_ms(&self) -> f64 {
        self.served.iter().map(|s| s.blocked_ms).sum()
    }
    pub fn goodput_per_s(&self) -> f64 {
        self.served.len() as f64 / self.wall_s.max(1e-9)
    }

    pub fn to_json(&self) -> Value {
        let mut warm = self.warm_ttfa();
        let mut cold = self.cold_ttfa();
        json!({
            "models": self.models,
            "budget_bytes": self.budget_bytes,
            "catalogue_bytes": self.catalogue_bytes,
            "overcommit": r3(self.catalogue_bytes as f64 / self.budget_bytes.max(1) as f64),
            "requests": self.served.len(),
            "aggregate_goodput_per_s": r3(self.goodput_per_s()),
            "weight_cache_hit_rate": self.weight_cache_hit_rate().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "warm_ttfa_ms": warm.to_json(),
            "cold_ttfa_ms": cold.to_json(),
            "evictions": self.evictions.len(),
            "eviction_regret": self.eviction_regret().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "eviction_regret_bytes": self.eviction_regret_bytes().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "blocked_behind_load_ms": r3(self.blocked_ms()),
            "per_model_fairness": self.fairness().map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
            "models_never_served": self.per_model_rate().iter().filter(|r| **r == 0.0).count(),
        })
    }
}

pub fn render(r: &Report) -> String {
    let mut warm = r.warm_ttfa();
    let mut cold = r.cold_ttfa();
    let f = |d: &mut Dist| d.percentile(0.50).map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into());
    let mut s = format!(
        "\n  {} models, {:.1}x over budget, {} requests\n",
        r.models,
        r.catalogue_bytes as f64 / r.budget_bytes.max(1) as f64,
        r.served.len()
    );
    s.push_str(&format!("  warm TTFA p50 {} ms   cold TTFA p50 {} ms\n", f(&mut warm), f(&mut cold)));
    s.push_str(&format!(
        "  hit-rate {}   eviction regret {} (bytes-weighted {})   fairness {}\n",
        r.weight_cache_hit_rate().map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "—".into()),
        r.eviction_regret().map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "—".into()),
        r.eviction_regret_bytes().map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "—".into()),
        r.fairness().map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into()),
    ));
    let starved = r.per_model_rate().iter().filter(|x| **x == 0.0).count();
    if starved > 0 {
        s.push_str(&format!(
            "  ! {starved} of {} models were never served — a good hit-rate can hide a starved tail\n",
            r.models
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served(model: usize, warm: bool, ttfa: f64) -> Served {
        Served { model, warm, ttfa_ms: ttfa, load_ms: if warm { 0.0 } else { ttfa }, blocked_ms: 0.0 }
    }

    #[test]
    fn catalogue_overcommits_the_budget() {
        let budget = 24 * 1024 * 1024 * 1024u64;
        let c = catalogue(20, budget, 4.0, 1.0);
        assert_eq!(c.len(), 20);
        let total: u64 = c.iter().map(|e| e.bytes).sum();
        assert!(total > budget * 2, "catalogue must not fit: {total} vs {budget}");
    }

    #[test]
    fn zipf_popularity_favours_the_head() {
        let c = catalogue(10, 1_000_000, 4.0, 1.0);
        assert!(c[0].weight > c[9].weight * 5.0, "alpha=1 should make model0 far hotter");
        let mut rng = data::rng::Rng::new(1);
        let draws = draw(&c, 2000, &mut rng);
        let head = draws.iter().filter(|&&i| i == 0).count();
        let tail = draws.iter().filter(|&&i| i == 9).count();
        assert!(head > tail * 3, "head {head} vs tail {tail}");
    }

    #[test]
    fn popularity_shift_moves_the_hot_set() {
        let mut c = catalogue(5, 1_000_000, 4.0, 1.0);
        let before = c[0].weight;
        shift_popularity(&mut c, 2);
        assert_ne!(c[0].weight, before, "a shift must actually change who is hot");
        // Weights are permuted, so the multiset is unchanged.
        let mut w: Vec<f64> = c.iter().map(|e| e.weight).collect();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((w.iter().sum::<f64>() - catalogue(5, 1_000_000, 4.0, 1.0).iter().map(|e| e.weight).sum::<f64>()).abs() < 1e-9);
    }

    #[test]
    fn cold_requests_are_slower_than_warm_ones() {
        let r = Report {
            served: vec![served(0, true, 10.0), served(1, false, 500.0)],
            wall_s: 1.0,
            models: 2,
            ..Default::default()
        };
        assert_eq!(r.weight_cache_hit_rate(), Some(0.5));
        assert!(r.cold_ttfa().percentile(0.5).unwrap() > r.warm_ttfa().percentile(0.5).unwrap());
    }

    #[test]
    fn fairness_counts_models_that_were_never_served() {
        // All traffic on model 0; models 1..4 starved.
        let r = Report {
            served: (0..10).map(|i| served(0, true, 5.0 + i as f64)).collect(),
            wall_s: 1.0,
            models: 5,
            ..Default::default()
        };
        let f = r.fairness().unwrap();
        assert!(f < 0.3, "a starved tail must show as unfair, got {f}");
        assert!(render(&r).contains("never served"));
    }

    fn ev(bytes: u64, back_ms: Option<f64>) -> Eviction {
        Eviction { bytes, until_rerequest_ms: back_ms }
    }

    #[test]
    fn eviction_regret_distinguishes_cold_from_wrong() {
        let cold = Report { evictions: vec![ev(100, None), ev(100, None)], ..Default::default() };
        assert_eq!(cold.eviction_regret(), Some(0.0));
        let wrong =
            Report { evictions: vec![ev(100, Some(100.0)), ev(100, Some(200.0))], ..Default::default() };
        assert_eq!(wrong.eviction_regret(), Some(1.0));
    }

    /// The metric a cost-aware policy is supposed to move: when the regretted
    /// evictions are the cheap ones, bytes-weighted regret is low even though
    /// event-counted regret says half of them were wrong.
    #[test]
    fn bytes_weighting_separates_cheap_regret_from_expensive() {
        let good_policy = Report {
            // Regrets only the 200 MB model; the 4 GB one stays correct.
            evictions: vec![ev(200, Some(100.0)), ev(4000, None)],
            ..Default::default()
        };
        let bad_policy = Report {
            evictions: vec![ev(200, None), ev(4000, Some(100.0))],
            ..Default::default()
        };
        // Event-counted: identical. Bytes-weighted: an order of magnitude apart.
        assert_eq!(good_policy.eviction_regret(), bad_policy.eviction_regret());
        let g = good_policy.eviction_regret_bytes().unwrap();
        let b = bad_policy.eviction_regret_bytes().unwrap();
        assert!(g < 0.05 && b > 0.9, "bytes must expose the difference: {g} vs {b}");
    }

    #[test]
    fn json_reports_the_overcommit_it_actually_ran() {
        let r = Report {
            served: vec![served(0, true, 1.0)],
            wall_s: 1.0,
            models: 4,
            budget_bytes: 1000,
            catalogue_bytes: 4000,
            ..Default::default()
        };
        assert_eq!(r.to_json()["overcommit"], 4.0);
    }
}
