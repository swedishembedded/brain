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
    // Unreachable via `run()`, which checks its knobs at the entry point (the
    // old message here claimed "budget already validated by the caller" when
    // NO caller validated — run(0, _) panicked right here).
    let mut ws = WeightSet::build(n_groups, budget, sched, plan).unwrap_or_else(|e| panic!("weights scenario: WeightSet::build({n_groups}, {budget}): {e}"));
    for cursor in 0..total {
        ws.advance(cursor);
    }
    Run { policy: policy_name.to_string(), n_groups, budget, passes, reloads: ws.reloads(), required_per_pass }
}

/// Z-Image-Turbo's real block count: 30 main layers + 2 noise + 2 context
/// refiners = 34 groups (see `s3dit::ZImageConfig::turbo`). `budget`/
/// `passes` are the caller's scenario knobs — `budget` is the window size
/// (device slots), `passes` the number of denoise steps simulated.
pub fn run(budget: u32, passes: u32) -> Result<Vec<Run>, String> {
    // The honest precondition, AT the entry point, as an Err rather than a
    // panic: `WeightSet::build` refuses a zero budget, and passes = 0 would
    // divide churn_overhead by nothing -- a caller's bad knob is a reportable
    // input error, not a crash.
    if budget < 1 || passes < 1 {
        return Err(format!("weights scenario: budget and passes must both be >= 1 (got budget={budget}, passes={passes})"));
    }
    let n_groups: u32 = 34;
    let lookahead: u32 = 1;
    let cyclic_pin = if budget >= n_groups { n_groups } else { budget.saturating_sub(lookahead.max(1)) };
    let required_per_pass = (n_groups - cyclic_pin) as u64;
    Ok(vec![
        drive("cyclic_scan", Box::new(CyclicScan { lookahead }), n_groups, budget, passes, required_per_pass),
        drive("lru", Box::new(Lru), n_groups, budget, passes, required_per_pass),
        // AllResident is the bit-identical control arm, valid only with a
        // full budget -- run it there rather than erroring at a small one.
        drive("all_resident", Box::new(AllResident), n_groups, n_groups.max(budget), passes, required_per_pass),
    ])
}

pub fn to_json(runs: &[Run]) -> Value {
    json!({ "runs": runs.iter().map(Run::to_json).collect::<Vec<_>>() })
}

pub fn render(runs: &[Run]) -> String {
    let mut s = String::from("\n  policy         budget  reloads  required/pass  churn_overhead\n");
    let min_budget = runs.iter().map(|r| r.budget).min().unwrap_or(0);
    let mut budgets_differ = false;
    for r in runs {
        budgets_differ |= r.budget != min_budget;
        s.push_str(&format!("  {:<13}  {:>6}  {:>7}  {:>13}  {:.3}\n", r.policy, r.budget, r.reloads, r.required_per_pass, r.churn_overhead()));
    }
    if budgets_differ {
        // The control arm (AllResident) runs at a FULL budget by construction
        // — say so, or the rows read as peers measured at the same window.
        s.push_str("  (budgets differ: all_resident is the full-budget control arm, not a peer at the window size)\n");
    }
    s
}

// ---------------------------------------------------------------------------
// qwen35's real per-layer int8 byte-cost profile - the SAME CyclicScan/Lru/
// AllResident policies above, driven against a SECOND real model's real
// group count (64 decoder layers, not Z-Image's 34 blocks) AND, unlike the
// Z-Image arm above (which only ever counted reloads, implicitly treating
// every block as equal-cost), a real per-group BYTE cost: `full_attention_
// interval=4` means 3 of every 4 layers are Gated DeltaNet (`LayerType::
// Linear`) and 1 of every 4 is GQA (`LayerType::Full`), and the two differ in
// real int8 footprint (`qwen35::config::Qwen35Config::layer_i8_bytes`,
// ~372-383 MB depending on type - verified against the real checkpoint's own
// dims by that function's own pinned test). [`ByteRun`] is additive, not a
// replacement for [`Run`]: Z-Image's block sizes were never claimed uniform
// either, but nothing upstream of this milestone ever measured whether that
// mattered, so [`Run`] stays exactly as it was rather than being widened
// for a claim only this crate's second model actually backs with numbers.
// ---------------------------------------------------------------------------

/// `qwen35::config::Qwen35Config::qwen38_27b()`'s real per-layer int8 byte
/// cost, one entry per of its 64 real decoder layers, in schedule order -
/// this scenario's other real-shape input, alongside Z-Image's block count
/// above.
fn qwen35_layer_bytes() -> Vec<u64> {
    let cfg = qwen35::config::Qwen35Config::qwen38_27b();
    cfg.layer_types().iter().map(|&ty| cfg.layer_i8_bytes(ty)).collect()
}

/// One policy's outcome, weighted by qwen35's real per-layer byte cost -
/// [`Run`]'s pure reload-COUNT sibling, extended with the real bytes moved.
/// `churn_overhead` (count-based) and `bytes_churn_overhead` (byte-weighted)
/// are reported side by side so a reader can see directly whether qwen35's
/// real ~372-383 MB per-layer spread (a real, but small - ~3% - heterogeneity)
/// changes which policy wins, rather than a doc merely asserting it doesn't.
#[derive(Clone, Debug)]
pub struct ByteRun {
    pub policy: String,
    pub n_groups: u32,
    pub budget: u32,
    pub passes: u32,
    pub reloads: u64,
    pub reload_bytes: u64,
    pub required_per_pass: u64,
    pub required_bytes_per_pass: u64,
}

impl ByteRun {
    pub fn churn_overhead(&self) -> f64 {
        weightset::churn_overhead(self.reloads, self.required_per_pass.max(1), self.passes)
    }
    pub fn bytes_churn_overhead(&self) -> f64 {
        self.reload_bytes as f64 / (self.required_bytes_per_pass.max(1) as f64 * self.passes as f64)
    }
    pub fn to_json(&self) -> Value {
        json!({
            "policy": self.policy,
            "n_groups": self.n_groups,
            "budget": self.budget,
            "passes": self.passes,
            "reloads": self.reloads,
            "reload_bytes": self.reload_bytes,
            "reload_mb": r3(self.reload_bytes as f64 / 1e6),
            "required_per_pass": self.required_per_pass,
            "required_bytes_per_pass": self.required_bytes_per_pass,
            "churn_overhead": r3(self.churn_overhead()),
            "bytes_churn_overhead": r3(self.bytes_churn_overhead()),
        })
    }
}

/// [`drive`]'s byte-weighted sibling: drives the real `WeightSet` exactly the
/// same way, additionally summing `group_bytes[gid]` for every schedule
/// position that was a miss.
fn drive_bytes(
    policy_name: &str,
    plan: Box<dyn weightset::ResidencyPlan + Send + Sync>,
    group_bytes: &[u64],
    budget: u32,
    passes: u32,
    required_per_pass: u64,
    required_bytes_per_pass: u64,
) -> ByteRun {
    let n_groups = group_bytes.len() as u32;
    let sched = Schedule::cyclic(n_groups, passes);
    let total = sched.order.len();
    let mut ws = WeightSet::build(n_groups, budget, sched.clone(), plan)
        .unwrap_or_else(|e| panic!("weights scenario (qwen35): WeightSet::build({n_groups}, {budget}): {e}"));
    let mut reload_bytes = 0u64;
    for cursor in 0..total {
        let (_, miss) = ws.advance(cursor);
        if miss {
            reload_bytes += group_bytes[sched.order[cursor].0 as usize];
        }
    }
    ByteRun {
        policy: policy_name.to_string(),
        n_groups,
        budget,
        passes,
        reloads: ws.reloads(),
        reload_bytes,
        required_per_pass,
        required_bytes_per_pass,
    }
}

/// Drive `CyclicScan`/`Lru`/`AllResident` over qwen35's real 64-layer int8
/// byte-cost profile ([`qwen35_layer_bytes`]) at `budget` slots, `passes`
/// passes over the schedule - the byte-weighted counterpart of [`run`] above,
/// same knob validation, same `AllResident`-runs-at-full-budget convention.
pub fn run_qwen35(budget: u32, passes: u32) -> Result<Vec<ByteRun>, String> {
    if budget < 1 || passes < 1 {
        return Err(format!("weights scenario (qwen35): budget and passes must both be >= 1 (got budget={budget}, passes={passes})"));
    }
    let group_bytes = qwen35_layer_bytes();
    let n_groups = group_bytes.len() as u32; // 64
    let lookahead: u32 = 1;
    let cyclic_pin = if budget >= n_groups { n_groups } else { budget.saturating_sub(lookahead.max(1)) };
    let required_per_pass = (n_groups - cyclic_pin) as u64;
    // The real byte sum of CyclicScan's own unpinned tail - the byte-weighted
    // counterpart of `required_per_pass`, and the fixed yardstick every
    // policy's `reload_bytes` is measured against (mirrors `run`'s own
    // `required_per_pass` role above).
    let required_bytes_per_pass: u64 = group_bytes[cyclic_pin as usize..].iter().sum();
    Ok(vec![
        drive_bytes("cyclic_scan", Box::new(CyclicScan { lookahead }), &group_bytes, budget, passes, required_per_pass, required_bytes_per_pass),
        drive_bytes("lru", Box::new(Lru), &group_bytes, budget, passes, required_per_pass, required_bytes_per_pass),
        drive_bytes(
            "all_resident",
            Box::new(AllResident),
            &qwen35_layer_bytes(),
            n_groups.max(budget),
            passes,
            required_per_pass,
            required_bytes_per_pass,
        ),
    ])
}

pub fn to_json_qwen35(runs: &[ByteRun]) -> Value {
    json!({ "runs": runs.iter().map(ByteRun::to_json).collect::<Vec<_>>() })
}

pub fn render_qwen35(runs: &[ByteRun]) -> String {
    let mut s = String::from("\n  policy         budget  reloads  reload_MB  required/pass  churn_overhead  bytes_churn_overhead\n");
    for r in runs {
        s.push_str(&format!(
            "  {:<13}  {:>6}  {:>7}  {:>9.1}  {:>13}  {:>14.3}  {:>21.3}\n",
            r.policy,
            r.budget,
            r.reloads,
            r.reload_bytes as f64 / 1e6,
            r.required_per_pass,
            r.churn_overhead(),
            r.bytes_churn_overhead(),
        ));
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
        let runs = run(10, 4).expect("valid knobs");
        let cyclic = runs.iter().find(|r| r.policy == "cyclic_scan").unwrap();
        let lru = runs.iter().find(|r| r.policy == "lru").unwrap();
        let all_resident = runs.iter().find(|r| r.policy == "all_resident").unwrap();
        assert_eq!(cyclic.churn_overhead(), 1.0);
        assert!(lru.churn_overhead() > cyclic.churn_overhead(), "Lru ({}) must lose to CyclicScan ({})", lru.churn_overhead(), cyclic.churn_overhead());
        assert_eq!(all_resident.reloads, 0);
    }

    #[test]
    fn a_window_at_least_as_wide_as_the_model_never_reloads_for_cyclic_scan_either() {
        let runs = run(34, 8).expect("valid knobs");
        let cyclic = runs.iter().find(|r| r.policy == "cyclic_scan").unwrap();
        assert_eq!(cyclic.reloads, 0);
    }

    /// qwen35's real per-layer byte profile has real heterogeneity (GDN vs
    /// GQA, ~372-383 MB) - this is the check that it does NOT flip the
    /// leaderboard result `run`'s own test above already established on
    /// Z-Image's uniform blocks: CyclicScan stays optimal on BOTH metrics,
    /// exactly (`1.0`), because its pin/tail split is fixed by the schedule
    /// and identical every pass regardless of what each group costs.
    #[test]
    fn qwen35_cyclic_scan_is_optimal_on_both_count_and_byte_metrics() {
        for budget in [2u32, 4, 8, 16, 32] {
            let runs = run_qwen35(budget, 8).expect("valid knobs");
            let cyclic = runs.iter().find(|r| r.policy == "cyclic_scan").unwrap();
            let lru = runs.iter().find(|r| r.policy == "lru").unwrap();
            assert_eq!(cyclic.churn_overhead(), 1.0, "budget={budget}");
            assert_eq!(cyclic.bytes_churn_overhead(), 1.0, "budget={budget}: CyclicScan's tail is the SAME set of groups every pass, so its own byte sum matches its own baseline exactly regardless of per-group heterogeneity");
            assert!(lru.churn_overhead() > 1.0, "budget={budget}");
            assert!(lru.bytes_churn_overhead() > 1.0, "budget={budget}");
        }
    }

    /// The real, measured answer to "does qwen35's real ~3% per-layer byte
    /// spread change the picture a uniform-cost model would give": no - at
    /// every budget tested, the byte-weighted and count-weighted overhead
    /// ratios agree to within 1%, because Lru's real reload set (every group,
    /// every pass, at any budget < n_groups - `weightset`'s own doc) touches
    /// a near-average mix of GDN/GQA layers regardless of budget.
    #[test]
    fn qwen35_byte_weighted_and_count_weighted_overhead_agree_closely() {
        for budget in [2u32, 4, 8, 16, 32] {
            let runs = run_qwen35(budget, 8).expect("valid knobs");
            let lru = runs.iter().find(|r| r.policy == "lru").unwrap();
            let count = lru.churn_overhead();
            let bytes = lru.bytes_churn_overhead();
            let rel_diff = (count - bytes).abs() / count;
            assert!(rel_diff < 0.01, "budget={budget}: count={count:.4} bytes={bytes:.4} rel_diff={rel_diff:.4} (expected <1%)");
        }
    }

    /// AllResident's control arm still holds for the byte-weighted metric:
    /// a full window never reloads, in bytes or in count.
    #[test]
    fn qwen35_all_resident_never_reloads_bytes_either() {
        let runs = run_qwen35(64, 8).expect("valid knobs");
        let all_resident = runs.iter().find(|r| r.policy == "all_resident").unwrap();
        assert_eq!(all_resident.reloads, 0);
        assert_eq!(all_resident.reload_bytes, 0);
    }

    #[test]
    fn qwen35_rejects_the_same_bad_knobs_as_the_zimage_arm() {
        assert!(run_qwen35(0, 4).is_err());
        assert!(run_qwen35(4, 0).is_err());
    }
}
