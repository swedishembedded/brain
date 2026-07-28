// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `startup` — cold and warm time-to-first-artifact.
//!
//! Isolated startup time is not the same as steady-state throughput, and it is
//! what decides whether autoscaling is economically useful: a replica that takes
//! 40 s to become useful cannot absorb a traffic spike, however fast it runs
//! afterwards. It is also where `precompile` and pipeline caching pay off.
//!
//! Four phases are timed separately, because they have different fixes:
//!
//! | phase | what it is | what shortens it |
//! |---|---|---|
//! | `device_init` | building the backend + compiling every WGSL pipeline | pipeline cache, fewer kernels |
//! | `weights_load` | reading and uploading parameters | format, mmap, quantisation |
//! | `first_prefill` | the first prompt through a cold engine | warm-up |
//! | `first_artifact` | to the first token out | all of the above |
//!
//! **Cold** builds everything from scratch; **warm** reuses a process that has
//! already run, so it isolates per-request cost from per-process cost.

use std::time::Instant;

use serde_json::json;

use crate::stats::{r3, Dist};

/// One measured start-up.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub device_init_ms: f64,
    pub weights_load_ms: f64,
    pub first_prefill_ms: f64,
    pub first_artifact_ms: f64,
}

impl Timings {
    pub fn total_ms(&self) -> f64 {
        self.device_init_ms + self.weights_load_ms + self.first_prefill_ms
    }
}

/// A stopwatch the target build path reports phases into.
pub struct Watch {
    start: Instant,
    last: Instant,
    pub timings: Timings,
}

impl Default for Watch {
    fn default() -> Watch {
        Watch::new()
    }
}

impl Watch {
    pub fn new() -> Watch {
        let now = Instant::now();
        Watch { start: now, last: now, timings: Timings::default() }
    }
    fn split(&mut self) -> f64 {
        let now = Instant::now();
        let ms = now.saturating_duration_since(self.last).as_secs_f64() * 1000.0;
        self.last = now;
        ms
    }
    pub fn device_ready(&mut self) {
        self.timings.device_init_ms = self.split();
    }
    pub fn weights_ready(&mut self) {
        self.timings.weights_load_ms = self.split();
    }
    pub fn first_prefill_done(&mut self) {
        self.timings.first_prefill_ms = self.split();
    }
    pub fn first_artifact(&mut self) {
        self.timings.first_artifact_ms =
            Instant::now().saturating_duration_since(self.start).as_secs_f64() * 1000.0;
    }
}

/// Aggregate several cold runs and (optionally) several warm ones.
pub fn to_json(cold: &[Timings], warm: &[Timings]) -> serde_json::Value {
    let block = |runs: &[Timings]| {
        if runs.is_empty() {
            return serde_json::Value::Null;
        }
        let mut dev = Dist::new();
        let mut wts = Dist::new();
        let mut pre = Dist::new();
        let mut fst = Dist::new();
        let mut tot = Dist::new();
        for t in runs {
            dev.push(t.device_init_ms);
            wts.push(t.weights_load_ms);
            pre.push(t.first_prefill_ms);
            fst.push(t.first_artifact_ms);
            tot.push(t.total_ms());
        }
        json!({
            "runs": runs.len(),
            "device_init_ms": dev.to_json(),
            "weights_load_ms": wts.to_json(),
            "first_prefill_ms": pre.to_json(),
            "first_artifact_ms": fst.to_json(),
            "total_ms": tot.to_json(),
        })
    };
    json!({ "cold": block(cold), "warm": block(warm) })
}

/// A one-line summary for the terminal.
pub fn render(cold: &[Timings], warm: &[Timings]) -> String {
    let mut s = String::new();
    let line = |label: &str, runs: &[Timings]| -> String {
        if runs.is_empty() {
            return format!("  {label:<8} —\n");
        }
        let n = runs.len() as f64;
        let mean = |f: fn(&Timings) -> f64| runs.iter().map(f).sum::<f64>() / n;
        format!(
            "  {label:<8} device {:>8.1}  weights {:>8.1}  prefill {:>8.1}  first-artifact {:>8.1} ms\n",
            r3(mean(|t| t.device_init_ms)),
            r3(mean(|t| t.weights_load_ms)),
            r3(mean(|t| t.first_prefill_ms)),
            r3(mean(|t| t.first_artifact_ms)),
        )
    };
    s.push_str(&line("cold", cold));
    s.push_str(&line("warm", warm));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(a: f64, b: f64, c: f64, d: f64) -> Timings {
        Timings { device_init_ms: a, weights_load_ms: b, first_prefill_ms: c, first_artifact_ms: d }
    }

    #[test]
    fn watch_attributes_time_to_the_phase_that_was_running() {
        let mut w = Watch::new();
        std::thread::sleep(std::time::Duration::from_millis(12));
        w.device_ready();
        std::thread::sleep(std::time::Duration::from_millis(12));
        w.weights_ready();
        w.first_prefill_done();
        w.first_artifact();
        assert!(w.timings.device_init_ms >= 10.0, "{:?}", w.timings);
        assert!(w.timings.weights_load_ms >= 10.0, "{:?}", w.timings);
        // first_artifact is measured from the start, so it covers both phases.
        assert!(w.timings.first_artifact_ms >= w.timings.device_init_ms);
    }

    #[test]
    fn total_excludes_first_artifact_which_is_cumulative() {
        let x = t(10.0, 20.0, 30.0, 60.0);
        assert_eq!(x.total_ms(), 60.0);
    }

    #[test]
    fn json_separates_cold_from_warm() {
        let j = to_json(&[t(100.0, 200.0, 50.0, 350.0)], &[t(0.0, 0.0, 20.0, 20.0)]);
        assert_eq!(j["cold"]["runs"], 1);
        assert_eq!(j["warm"]["runs"], 1);
        assert_eq!(j["cold"]["device_init_ms"]["p50"], 100.0);
        assert_eq!(j["warm"]["device_init_ms"]["p50"], 0.0);
    }

    #[test]
    fn absent_warm_runs_are_null_not_zero() {
        let j = to_json(&[t(1.0, 2.0, 3.0, 6.0)], &[]);
        assert!(j["warm"].is_null(), "no warm runs must not read as instant start-up");
    }

    #[test]
    fn render_mentions_both_rows() {
        let s = render(&[t(1.0, 2.0, 3.0, 6.0)], &[]);
        assert!(s.contains("cold"));
        assert!(s.contains("warm"));
    }
}
