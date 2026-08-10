// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `weights` — the within-instance weight window (`crates/weightset`), driven
//! against the REAL policy code (`CyclicScan`/`Lru`/`AllResident`), not a
//! re-simulation: this scenario's cross-instance sibling is `residency`
//! (many *models* over one budget, LRU/cost-aware, scored from the past). A
//! denoise/decode loop's block order is known exactly in advance, which is
//! the whole reason `weightset::CyclicScan` exists instead of scoring
//! recency — this scenario is the leaderboard row that proves CyclicScan
//! beats Lru on identical seeds, rather than a doc merely asserting it.

use serde_json::{json, Value};
use weightset::{AllResident, CyclicScan, Lru, Schedule, WeightSet};

use crate::stats::r3;

/// One policy's outcome over one `(n_groups, budget, passes)` run.
#[derive(Clone, Debug)]
pub struct Run {
    pub policy: String,
    pub n_groups: u32,
    pub budget: u32,
    pub passes: u32,
    pub reloads: u64,
    /// The fixed yardstick every policy's `reloads` is measured against:
    /// `CyclicScan`'s own achievable pin at this budget, not each policy's
    /// own (possibly much smaller) pin — see `weightset::churn_overhead`'s
    /// doc for why that is the fair comparison.
    pub required_per_pass: u64,
}

impl Run {
    pub fn churn_overhead(&self) -> f64 {
        weightset::churn_overhead(self.reloads, self.required_per_pass.max(1), self.passes)
    }
    pub fn to_json(&self) -> Value {
        json!({
            "policy": self.policy,
            "n_groups": self.n_groups,
            "budget": self.budget,
            "passes": self.passes,
            "reloads": self.reloads,
            "required_per_pass": self.required_per_pass,
            "churn_overhead": r3(self.churn_overhead()),
        })
    }
}

/// Drive `plan` over `n_groups` groups, `budget` slots, `passes` — the real
/// `weightset::WeightSet` code, not a re-implementation of it.
fn drive(policy_name: &str, plan: Box<dyn weightset::ResidencyPlan + Send + Sync>, n_groups: u32, budget: u32, passes: u32, required_per_pass: u64) -> Run {
    let sched = Schedule::cyclic(n_groups, passes);
    let total = sched.order.len();
    let mut ws = WeightSet::build(n_groups, budget, sched, plan).expect("budget already validated by the caller");
    for cursor in 0..total {
        ws.advance(cursor);
    }
    Run { policy: policy_name.to_string(), n_groups, budget, passes, reloads: ws.reloads(), required_per_pass }
}

/// Z-Image-Turbo's real block count: 30 main layers + 2 noise + 2 context
/// refiners = 34 groups (see `zimage::ZImageConfig::turbo`). `budget`/
/// `passes` are the caller's scenario knobs — `budget` is the window size
/// (device slots), `passes` the number of denoise steps simulated.
pub fn run(budget: u32, passes: u32) -> Vec<Run> {
    let n_groups: u32 = 34;
    let lookahead: u32 = 1;
    let cyclic_pin = if budget >= n_groups { n_groups } else { budget.saturating_sub(lookahead.max(1)) };
    let required_per_pass = (n_groups - cyclic_pin) as u64;
    vec![
        drive("cyclic_scan", Box::new(CyclicScan { lookahead }), n_groups, budget, passes, required_per_pass),
        drive("lru", Box::new(Lru), n_groups, budget, passes, required_per_pass),
        // AllResident is the bit-identical control arm, valid only with a
        // full budget -- run it there rather than erroring at a small one.
        drive("all_resident", Box::new(AllResident), n_groups, n_groups.max(budget), passes, required_per_pass),
    ]
}

pub fn to_json(runs: &[Run]) -> Value {
    json!({ "runs": runs.iter().map(Run::to_json).collect::<Vec<_>>() })
}

pub fn render(runs: &[Run]) -> String {
    let mut s = String::from("\n  policy         budget  reloads  required/pass  churn_overhead\n");
    for r in runs {
        s.push_str(&format!("  {:<13}  {:>6}  {:>7}  {:>13}  {:.3}\n", r.policy, r.budget, r.reloads, r.required_per_pass, r.churn_overhead()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The leaderboard claim itself, run against the real `weightset` code:
    /// CyclicScan is optimal (`1.0`, exact), Lru is strictly worse, and
    /// AllResident never reloads at all — on identical seeds (there is no
    /// randomness here at all; the schedule is deterministic), not asserted
    /// in a doc.
    #[test]
    fn cyclic_scan_is_optimal_lru_is_worse_all_resident_never_reloads() {
        let runs = run(10, 4);
        let cyclic = runs.iter().find(|r| r.policy == "cyclic_scan").unwrap();
        let lru = runs.iter().find(|r| r.policy == "lru").unwrap();
        let all_resident = runs.iter().find(|r| r.policy == "all_resident").unwrap();
        assert_eq!(cyclic.churn_overhead(), 1.0);
        assert!(lru.churn_overhead() > cyclic.churn_overhead(), "Lru ({}) must lose to CyclicScan ({})", lru.churn_overhead(), cyclic.churn_overhead());
        assert_eq!(all_resident.reloads, 0);
    }

    #[test]
    fn a_window_at_least_as_wide_as_the_model_never_reloads_for_cyclic_scan_either() {
        let runs = run(34, 8);
        let cyclic = runs.iter().find(|r| r.policy == "cyclic_scan").unwrap();
        assert_eq!(cyclic.reloads, 0);
    }
}
