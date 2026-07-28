// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `cancel` — what a cancelled request costs.
//!
//! Production requests are abandoned constantly: a user closes the tab, an
//! upstream times out, an agent gives up on a branch, a speculative call loses
//! its race. If the engine keeps decoding after the client is gone, that compute
//! is spent on output nobody will read — while the requests that *are* still
//! wanted wait behind it.
//!
//! The headline is therefore:
//!
//! ```text
//! cancelled_compute_waste = artifacts produced after cancellation
//!                         / artifacts produced in total
//! ```
//!
//! Cancellation is also where resources leak. A request killed mid-decode must
//! return its KV blocks; if it does not, a server under normal churn slowly
//! loses its cache to sequences nobody is reading. So the scenario checks block
//! accounting across a cancellation storm, not just latency.

use serde_json::{json, Value};

use crate::stats::r3;

/// Where in a request's life the cancellation lands. Each exercises a different
/// teardown path, and the interesting failures are stage-specific: cancelling
/// during prefill must not leave half-written KV, during decode must stop the
/// batch cleanly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Queued,
    Prefill,
    Decode,
    Streaming,
}

impl Stage {
    pub fn name(&self) -> &'static str {
        match self {
            Stage::Queued => "queued",
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
            Stage::Streaming => "streaming",
        }
    }
    pub fn all() -> &'static [Stage] {
        &[Stage::Queued, Stage::Prefill, Stage::Decode, Stage::Streaming]
    }
}

/// What one cancellation cost.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    pub stage: Option<&'static str>,
    /// Artifacts produced before the cancel was requested (legitimately useful).
    pub before: usize,
    /// Artifacts produced *after* — pure waste.
    pub after: usize,
    /// Time from requesting cancellation to the request actually stopping.
    pub abort_ms: f64,
    /// Time until the request's KV blocks were back in the pool.
    pub reclaim_ms: f64,
    /// Blocks still missing from the pool afterwards. Non-zero is a leak.
    pub leaked_blocks: i64,
}

/// Aggregate across a cancellation run.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub observations: Vec<Observation>,
    /// Requests that were never cancelled, for the interference check.
    pub unaffected_completed: usize,
    pub unaffected_expected: usize,
}

impl Report {
    pub fn total_before(&self) -> usize {
        self.observations.iter().map(|o| o.before).sum()
    }
    pub fn total_after(&self) -> usize {
        self.observations.iter().map(|o| o.after).sum()
    }

    /// The headline: the share of produced work that was already pointless.
    pub fn waste(&self) -> f64 {
        let total = self.total_before() + self.total_after();
        if total == 0 {
            return 0.0;
        }
        self.total_after() as f64 / total as f64
    }

    /// Any block not returned to the pool is a leak, and leaks compound.
    pub fn leaked_blocks(&self) -> i64 {
        self.observations.iter().map(|o| o.leaked_blocks).sum()
    }

    /// True when every request that was *not* cancelled still completed —
    /// cancellation must not take down its neighbours.
    pub fn neighbours_unharmed(&self) -> bool {
        self.unaffected_completed >= self.unaffected_expected
    }

    pub fn to_json(&self) -> Value {
        let mut abort = crate::stats::Dist::new();
        let mut reclaim = crate::stats::Dist::new();
        for o in &self.observations {
            abort.push(o.abort_ms);
            reclaim.push(o.reclaim_ms);
        }
        let per_stage: Vec<Value> = Stage::all()
            .iter()
            .filter_map(|st| {
                let mine: Vec<&Observation> =
                    self.observations.iter().filter(|o| o.stage == Some(st.name())).collect();
                if mine.is_empty() {
                    return None;
                }
                let after: usize = mine.iter().map(|o| o.after).sum();
                let before: usize = mine.iter().map(|o| o.before).sum();
                let total = (after + before).max(1);
                Some(json!({
                    "stage": st.name(),
                    "cancellations": mine.len(),
                    "artifacts_after_cancel": after,
                    "waste": r3(after as f64 / total as f64),
                    "leaked_blocks": mine.iter().map(|o| o.leaked_blocks).sum::<i64>(),
                }))
            })
            .collect();
        json!({
            "cancellations": self.observations.len(),
            "cancelled_compute_waste": r3(self.waste()),
            "artifacts_after_cancel": self.total_after(),
            "abort_latency_ms": abort.to_json(),
            "block_reclaim_ms": reclaim.to_json(),
            "leaked_blocks": self.leaked_blocks(),
            "neighbours_unharmed": self.neighbours_unharmed(),
            "per_stage": per_stage,
        })
    }
}

pub fn render(r: &Report) -> String {
    let mut s = format!(
        "\n  cancellations {}  waste {:.1}%  leaked blocks {}\n",
        r.observations.len(),
        r.waste() * 100.0,
        r.leaked_blocks()
    );
    if r.leaked_blocks() != 0 {
        s.push_str("  ! KV blocks were not returned — a cancellation leak compounds under churn\n");
    }
    if !r.neighbours_unharmed() {
        s.push_str("  ! requests that were NOT cancelled failed to complete\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(stage: Stage, before: usize, after: usize, leaked: i64) -> Observation {
        Observation {
            stage: Some(stage.name()),
            before,
            after,
            abort_ms: 1.0,
            reclaim_ms: 2.0,
            leaked_blocks: leaked,
        }
    }

    #[test]
    fn a_clean_cancel_wastes_nothing() {
        let r = Report { observations: vec![obs(Stage::Decode, 10, 0, 0)], ..Default::default() };
        assert_eq!(r.waste(), 0.0);
        assert_eq!(r.leaked_blocks(), 0);
    }

    #[test]
    fn work_produced_after_cancellation_is_waste() {
        let r = Report { observations: vec![obs(Stage::Decode, 10, 30, 0)], ..Default::default() };
        assert_eq!(r.waste(), 0.75);
    }

    #[test]
    fn cancelling_before_any_output_is_not_a_divide_by_zero() {
        let r = Report { observations: vec![obs(Stage::Queued, 0, 0, 0)], ..Default::default() };
        assert_eq!(r.waste(), 0.0);
    }

    #[test]
    fn unreturned_blocks_are_reported_as_a_leak() {
        let r = Report { observations: vec![obs(Stage::Prefill, 0, 0, 3)], ..Default::default() };
        assert_eq!(r.leaked_blocks(), 3);
        assert!(render(&r).contains("leak"));
    }

    #[test]
    fn neighbours_must_survive_a_cancellation_storm() {
        let ok = Report { unaffected_completed: 8, unaffected_expected: 8, ..Default::default() };
        assert!(ok.neighbours_unharmed());
        let bad = Report { unaffected_completed: 5, unaffected_expected: 8, ..Default::default() };
        assert!(!bad.neighbours_unharmed());
        assert!(render(&bad).contains("NOT cancelled"));
    }

    #[test]
    fn per_stage_breakdown_only_lists_stages_that_ran() {
        let r = Report {
            observations: vec![obs(Stage::Decode, 4, 4, 0), obs(Stage::Queued, 0, 0, 0)],
            ..Default::default()
        };
        let j = r.to_json();
        let stages: Vec<&str> =
            j["per_stage"].as_array().unwrap().iter().map(|s| s["stage"].as_str().unwrap()).collect();
        assert!(stages.contains(&"decode") && stages.contains(&"queued"));
        assert!(!stages.contains(&"streaming"));
    }

    #[test]
    fn every_stage_has_a_name() {
        assert_eq!(Stage::all().len(), 4);
        for s in Stage::all() {
            assert!(!s.name().is_empty());
        }
    }
}
