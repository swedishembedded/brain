// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `perf gate` — a candidate run against a committed baseline as HARD floors
//! (J2). The pattern is `scripts/wm-perf-gate.sh`: best-of-N against committed
//! baselines with generous floors, because tight deltas flap on shared boxes
//! and a flapping gate gets deleted.
//!
//! Rules, each of which exists because breaking it produces a gate that lies:
//!
//! * Two runs gate only if they are comparable: same scenario, same artifact
//!   unit, same environment axes (device, backend, adapter, build, cores). A
//!   baseline from different hardware is not a floor, it is a coincidence.
//! * `valid: false` (failed correctness gate) refuses to gate in either
//!   direction — a fast-but-broken run must not become a floor, nor pass one.
//! * A smoke-sized candidate refuses: its numbers are not measurements.
//! * A metric missing on either side is SKIPPED and said out loud — unmeasured
//!   never gates and never passes silently.
//! * Throughput floors are `candidate >= baseline * floor_frac`; latency
//!   ceilings are `candidate <= baseline / floor_frac`.

use crate::report::Row;

/// One gated metric's verdict.
#[derive(Debug)]
pub struct Check {
    pub metric: &'static str,
    pub candidate: f64,
    pub baseline: f64,
    pub bound: f64,
    pub pass: bool,
}

/// The gate's outcome: every checked metric, plus metrics skipped as
/// unmeasured. `refused` is set when the pair must not be gated at all.
#[derive(Debug, Default)]
pub struct Outcome {
    pub checks: Vec<Check>,
    pub skipped: Vec<&'static str>,
    pub refused: Option<String>,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.refused.is_none() && self.checks.iter().all(|c| c.pass)
    }
}

/// Gate `candidate` against `baseline` with hard floors at `floor_frac`
/// (e.g. 0.85 = tolerate a 15% regression before failing).
pub fn gate(candidate: &Row, baseline: &Row, floor_frac: f64) -> Outcome {
    let mut out = Outcome::default();
    if !(0.0..=1.0).contains(&floor_frac) {
        out.refused = Some(format!("floor fraction {floor_frac} outside (0, 1]"));
        return out;
    }
    if candidate.scenario != baseline.scenario {
        out.refused = Some(format!(
            "scenario mismatch: {} vs {}",
            candidate.scenario, baseline.scenario
        ));
        return out;
    }
    if candidate.unit != baseline.unit {
        out.refused =
            Some(format!("artifact unit mismatch: {} vs {}", candidate.unit, baseline.unit));
        return out;
    }
    for (r, which) in [(candidate, "candidate"), (baseline, "baseline")] {
        if !r.valid {
            out.refused = Some(format!(
                "{which} failed its correctness gate ({}) — a broken run neither sets nor passes a floor",
                r.invalid_reason.as_deref().unwrap_or("no reason recorded")
            ));
            return out;
        }
    }
    if candidate.smoke {
        out.refused = Some("candidate is a smoke run — not a measurement".into());
        return out;
    }
    for (axis, cv) in &candidate.axes {
        if let Some((_, bv)) = baseline.axes.iter().find(|(a, _)| a == axis) {
            if cv != bv {
                out.refused = Some(format!(
                    "environment axis {axis:?} differs ({cv} vs {bv}) — a baseline from \
                     different hardware/build is not a floor"
                ));
                return out;
            }
        }
    }

    let mut floor = |name, c: Option<f64>, b: Option<f64>| match (c, b) {
        (Some(c), Some(b)) => {
            let bound = b * floor_frac;
            out.checks.push(Check { metric: name, candidate: c, baseline: b, bound, pass: c >= bound });
        }
        _ => out.skipped.push(name),
    };
    floor("output_per_s", candidate.output_per_s, baseline.output_per_s);
    floor("goodput_per_s", candidate.goodput_per_s, baseline.goodput_per_s);

    let mut ceiling = |name, c: Option<f64>, b: Option<f64>| match (c, b) {
        (Some(c), Some(b)) => {
            let bound = b / floor_frac;
            out.checks.push(Check { metric: name, candidate: c, baseline: b, bound, pass: c <= bound });
        }
        _ => out.skipped.push(name),
    };
    ceiling("ttfa_p99_ms", candidate.ttfa_p99, baseline.ttfa_p99);
    ceiling("ial_p99_ms", candidate.ial_p99, baseline.ial_p99);

    // A gate that checked NOTHING must not read as a pass — an artifact whose
    // shape carries none of the gated metrics (e.g. a ladder-only sweep)
    // needs a scenario with flat metrics, not a vacuous green.
    if out.checks.is_empty() {
        out.refused = Some(
            "no comparable metrics between these artifacts — nothing was actually gated".into(),
        );
    }
    out
}

/// Human-readable verdict, one line per metric.
pub fn render(out: &Outcome, floor_frac: f64) -> String {
    if let Some(r) = &out.refused {
        return format!("gate REFUSED: {r}\n");
    }
    let mut s = format!("gate at {:.0}% of baseline (hard floors):\n", floor_frac * 100.0);
    for c in &out.checks {
        s.push_str(&format!(
            "  {:<14} {:>10.2} vs baseline {:>10.2}  (bound {:>10.2})  {}\n",
            c.metric,
            c.candidate,
            c.baseline,
            c.bound,
            if c.pass { "ok" } else { "FAIL" }
        ));
    }
    for m in &out.skipped {
        s.push_str(&format!("  {m:<14} unmeasured on one side — skipped, not passed\n"));
    }
    s.push_str(if out.passed() { "PASS\n" } else { "FAIL\n" });
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(out: f64, ttfa: f64) -> Row {
        Row {
            path: "x".into(),
            scenario: "sweep".into(),
            model: "m".into(),
            unit: "token".into(),
            label: "gpu".into(),
            valid: true,
            invalid_reason: None,
            smoke: false,
            software_gpu: false,
            output_per_s: Some(out),
            goodput_per_s: None,
            ttfa_p99: Some(ttfa),
            ial_p99: None,
            axes: vec![("adapter".into(), "P40".into())],
        }
    }

    #[test]
    fn floors_and_ceilings_bound_in_opposite_directions() {
        let base = row(100.0, 200.0);
        // 14% slower throughput, 10% worse latency: inside the 85% floor.
        let ok = row(86.0, 220.0);
        let o = gate(&ok, &base, 0.85);
        assert!(o.passed(), "{o:?}");
        // 20% slower throughput: through the floor.
        let slow = row(80.0, 200.0);
        assert!(!gate(&slow, &base, 0.85).passed());
        // Latency ceiling: 30% worse TTFA fails even with fine throughput.
        let laggy = row(100.0, 260.0);
        assert!(!gate(&laggy, &base, 0.85).passed());
    }

    #[test]
    fn unmeasured_metrics_skip_and_say_so() {
        let mut base = row(100.0, 200.0);
        base.ttfa_p99 = None;
        let o = gate(&row(100.0, 999.0), &base, 0.85);
        assert!(o.passed(), "an unmeasured baseline metric must not gate");
        assert!(o.skipped.contains(&"ttfa_p99_ms"));
        assert!(render(&o, 0.85).contains("skipped, not passed"));
    }

    /// Zero checks = zero evidence: a pair with NO comparable metrics refuses
    /// rather than passing vacuously.
    #[test]
    fn a_gate_that_checked_nothing_refuses() {
        let mut a = row(0.0, 0.0);
        let mut b = row(0.0, 0.0);
        for r in [&mut a, &mut b] {
            r.output_per_s = None;
            r.goodput_per_s = None;
            r.ttfa_p99 = None;
            r.ial_p99 = None;
        }
        let o = gate(&a, &b, 0.85);
        assert!(!o.passed());
        assert!(o.refused.as_deref().unwrap_or("").contains("nothing was actually gated"));
    }

    #[test]
    fn incomparable_or_broken_runs_refuse() {
        let base = row(100.0, 200.0);
        let mut other_box = row(100.0, 200.0);
        other_box.axes = vec![("adapter".into(), "llvmpipe".into())];
        assert!(gate(&other_box, &base, 0.85).refused.is_some());
        let mut invalid = row(500.0, 10.0);
        invalid.valid = false;
        assert!(gate(&invalid, &base, 0.85).refused.is_some(), "fast-but-broken must not pass");
        let mut smoke = row(100.0, 200.0);
        smoke.smoke = true;
        assert!(gate(&smoke, &base, 0.85).refused.is_some());
        let mut unit = row(100.0, 200.0);
        unit.unit = "frame".into();
        assert!(gate(&unit, &base, 0.85).refused.is_some(), "cross-unit ranking is meaningless");
    }
}
