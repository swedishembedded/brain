// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The scenarios. Each owns its workload shape and what it reports; the runner
//! owns driving them — the same split `crates/bench` uses for learnability.
//!
//! Tier 1 (here): `latency`, `throughput`, `serve`, `sweep`.
//! Tier 2 (`residency`, `kvcache`, `placement`, `mixed`, `overload`, `cancel`,
//! `soak`, `frontend`, `faults`, `energy`) is designed in
//! `docs/performance/benchmarking.md` and not yet implemented.
//!
//! The three core scenarios differ **only** in their arrival process, and that
//! is the entire point: requests that all exist at t=0 exercise a different
//! engine than requests arriving over time, so each is measured separately and
//! never averaged together.

use crate::driver;
use crate::env::Env;
use crate::metrics::Summary;
use crate::report;
use crate::schema::{memory_with, Artifact};
use crate::stats::BestOf;
use crate::target::PerfTarget;
use crate::workload::{Arrival, Workload};

/// How to run a scenario.
#[derive(Clone, Debug)]
pub struct Options {
    pub device: String,
    pub seed: u64,
    pub best_of: usize,
    pub smoke: bool,
    /// Concurrency ladder for `sweep`.
    pub concurrency: Vec<usize>,
    /// Requests per measurement.
    pub num_requests: usize,
    pub warmup_requests: usize,
    /// Override the workload's input length (prompt artifacts).
    pub input_override: Option<usize>,
    /// Override the workload's output length.
    ///
    /// Run time is dominated by output length × per-artifact cost, not by
    /// request count: 256 output tokens at 90 ms/token is 23 s *per request*
    /// however few requests you run. Shortening the output is what makes an
    /// iteration loop fast, and it leaves the *rates* (out/s, IAL) comparable
    /// while making TTFA-relative-to-E2E incomparable — so the override is
    /// recorded in the artifact.
    pub output_override: Option<usize>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            device: "cpu".into(),
            seed: 1234,
            best_of: 1,
            smoke: false,
            concurrency: vec![1, 2, 4, 8, 16, 32],
            num_requests: 64,
            warmup_requests: 4,
            input_override: None,
            output_override: None,
        }
    }
}

impl Options {
    /// Shrink everything so the suite is CI-runnable in seconds. Smoke artifacts
    /// are marked and never compared against full runs.
    pub fn smoke(mut self) -> Options {
        self.smoke = true;
        self.num_requests = 4;
        self.warmup_requests = 1;
        self.best_of = 1;
        self.concurrency = vec![1, 2];
        // Cap the output length too: request count alone does not bound run
        // time, output length does.
        self.output_override = Some(self.output_override.unwrap_or(8).min(8));
        self
    }
}

/// The registered scenario names, in the order `brain perf list` prints them.
pub const SCENARIOS: &[(&str, &str)] = &[
    ("latency", "fixed batch, in-process, no transport — kernel/engine regression signal"),
    ("throughput", "offline saturated: every request available at t=0 — the efficiency ceiling"),
    ("serve", "arrival-process load at a fixed concurrency — realistic behaviour"),
    ("sweep", "a concurrency ladder — the throughput-vs-latency curve and max goodput under SLO"),
];

pub fn is_scenario(name: &str) -> bool {
    SCENARIOS.iter().any(|(n, _)| *n == name)
}

/// Build the workload a scenario uses for a given standard workload name.
fn workload_for(scenario: &str, name: &str, opt: &Options, concurrency: usize) -> Option<Workload> {
    let arrival = match scenario {
        // `latency` is a fixed batch: exactly `concurrency` in flight, nothing queued
        // behind it, so the number is the engine's own latency and not queue time.
        "latency" => Arrival::ClosedLoop { concurrency },
        "throughput" => Arrival::Saturated,
        "serve" | "sweep" => Arrival::ClosedLoop { concurrency },
        _ => return None,
    };
    let mut w = crate::workload::standard(name, arrival, opt.num_requests, opt.seed)?;
    w.warmup_requests = opt.warmup_requests;
    if let Some(n) = opt.input_override {
        for c in &mut w.classes {
            c.input = crate::workload::Lengths::Fixed(n.max(1));
        }
        w.name = format!("{}/in{n}", w.name);
    }
    if let Some(n) = opt.output_override {
        for c in &mut w.classes {
            c.output = crate::workload::Lengths::Fixed(n.max(1));
        }
        w.name = format!("{}/out{n}", w.name);
    }
    Some(w)
}

/// Run one measurement and return `(summary, workload)`.
fn measure(target: &mut dyn PerfTarget, w: &Workload) -> Summary {
    let run = driver::drive(target, w);
    let slo = w.slo_for(0);
    Summary::build(&run.records, run.wall_s, slo)
}

/// Run a Tier-1 scenario, producing the result artifact.
///
/// `concurrency` selects the fixed level for `latency`/`serve`; `sweep` ignores
/// it and walks `opt.concurrency`.
pub fn run(
    scenario: &str,
    target: &mut dyn PerfTarget,
    workload_name: &str,
    concurrency: usize,
    opt: &Options,
) -> Result<Artifact, String> {
    if !is_scenario(scenario) {
        return Err(format!("unknown scenario '{scenario}'"));
    }
    let info = target.describe();
    let mut art = Artifact::new(scenario, Env::capture(&opt.device), info);
    art.smoke = opt.smoke;

    if scenario == "sweep" {
        return run_sweep(target, workload_name, opt, art);
    }

    let w = workload_for(scenario, workload_name, opt, concurrency)
        .ok_or_else(|| format!("unknown workload '{workload_name}'"))?;

    // Best-of-N: the stable signal on a shared or throttled box. The spread is
    // reported alongside so the reader knows how noisy the machine was.
    let mut best = BestOf::higher_better();
    let mut chosen: Option<Summary> = None;
    for i in 0..opt.best_of.max(1) {
        target.reset(i > 0);
        let s = measure(target, &w);
        best.push(s.output_per_s);
        if chosen.as_ref().is_none_or(|c| s.output_per_s > c.output_per_s) {
            chosen = Some(s);
        }
    }
    let mut s = chosen.expect("at least one measurement");

    art.workload = w.to_json();
    art.performance = s.to_json();
    art.scheduling = s.scheduling_json(None, &[s.output_per_s]);
    art.memory = memory_with(&target.counters());
    art.best_of_n = opt.best_of.max(1);
    art.spread_pct = best.spread_pct();
    Ok(art)
}

/// The concurrency ladder. The deliverable is not the peak rate — it is the
/// **highest goodput that still satisfies the SLO**, plus the curve showing
/// where latency falls off.
fn run_sweep(
    target: &mut dyn PerfTarget,
    workload_name: &str,
    opt: &Options,
    mut art: Artifact,
) -> Result<Artifact, String> {
    let mut curve = Vec::new();
    let mut best_point: Option<(usize, f64)> = None;
    let mut last_workload = None;

    for &c in &opt.concurrency {
        let w = workload_for("sweep", workload_name, opt, c)
            .ok_or_else(|| format!("unknown workload '{workload_name}'"))?;
        target.reset(true);
        let mut s = measure(target, &w);
        let attained = s.slo_attainment;
        let goodput = s.goodput_per_s;
        curve.push(serde_json::json!({
            "concurrency": c,
            "requests_per_s": crate::stats::r3(s.requests_per_s),
            "output_artifacts_per_s": crate::stats::r3(s.output_per_s),
            "goodput_per_s": crate::stats::r3(goodput),
            "slo_attainment": crate::stats::r3(attained),
            "ttfa_ms": s.ttfa.to_json(),
            "ial_ms": s.ial.to_json(),
            "e2e_ms": s.e2e.to_json(),
        }));
        // "Sustainable" = the SLO actually held, not merely "it produced output".
        if attained >= 0.99 && best_point.map(|(_, g)| goodput > g).unwrap_or(true) {
            best_point = Some((c, goodput));
        }
        last_workload = Some(w);
    }

    if let Some(w) = last_workload {
        art.workload = w.to_json();
    }
    let (best_c, best_g) = best_point.unzip();
    art.performance = serde_json::json!({
        "max_sustainable_concurrency": best_c.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
        "max_goodput_per_s": best_g.map(|g| serde_json::Value::from(crate::stats::r3(g))).unwrap_or(serde_json::Value::Null),
        "peak_output_artifacts_per_s": curve
            .iter()
            .filter_map(|p| p["output_artifacts_per_s"].as_f64())
            .fold(None::<f64>, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    });
    art.curve = Some(curve);
    art.memory = memory_with(&target.counters());
    Ok(art)
}

/// Render a finished artifact for the terminal.
pub fn render(art: &Artifact) -> String {
    if art.scenario == "sweep" {
        return render_sweep(art);
    }
    report::render(art)
}

fn render_sweep(art: &Artifact) -> String {
    let mut s = format!(
        "\nsweep — {} on {} (workload {})\n",
        art.target.model,
        art.env.label(),
        art.workload["name"].as_str().unwrap_or("-")
    );
    if art.env.is_software_gpu() {
        s.push_str("  ! software rasteriser: this is NOT a hardware GPU result\n");
    }
    s.push_str(&format!(
        "\n{:>5} {:>12} {:>12} {:>12} {:>11} {:>11}\n",
        "conc", "req/s", "out/s", "goodput/s", "ttfa p99", "ial p99"
    ));
    s.push_str(&format!("{:-<70}\n", ""));
    for p in art.curve.iter().flatten() {
        let g = |k: &str| p[k].as_f64().map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into());
        s.push_str(&format!(
            "{:>5} {:>12} {:>12} {:>12} {:>11} {:>11}\n",
            // `Value`'s Display ignores width/fill, so extract before formatting.
            p["concurrency"].as_u64().unwrap_or(0),
            g("requests_per_s"),
            g("output_artifacts_per_s"),
            g("goodput_per_s"),
            p["ttfa_ms"]["p99"].as_f64().map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into()),
            p["ial_ms"]["p99"].as_f64().map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into()),
        ));
    }
    match art.performance["max_sustainable_concurrency"].as_u64() {
        Some(c) => s.push_str(&format!(
            "\nmax sustainable concurrency (SLO held): {c} at {} {}s/s goodput\n",
            art.performance["max_goodput_per_s"],
            art.target.artifact_unit
        )),
        None => s.push_str("\nno concurrency level met the SLO — the workload exceeds this configuration\n"),
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::testing::FakeTarget;

    fn opt() -> Options {
        Options { num_requests: 8, warmup_requests: 2, concurrency: vec![1, 2, 4], ..Default::default() }
    }

    #[test]
    fn every_registered_scenario_runs_end_to_end() {
        for (name, _) in SCENARIOS {
            let mut t = FakeTarget::new(8, 1);
            let art = run(name, &mut t, "interactive", 4, &opt())
                .unwrap_or_else(|e| panic!("scenario {name} failed: {e}"));
            assert_eq!(art.scenario, *name);
            assert!(art.to_json()["workload"]["name"].is_string(), "{name} must record its workload");
        }
    }

    #[test]
    fn unknown_scenario_and_workload_are_errors() {
        let mut t = FakeTarget::new(4, 1);
        assert!(run("nope", &mut t, "chat", 1, &opt()).is_err());
        assert!(run("serve", &mut t, "not-a-workload", 1, &opt()).is_err());
    }

    #[test]
    fn throughput_and_serve_use_different_arrival_processes() {
        let mut t = FakeTarget::new(8, 1);
        let a = run("throughput", &mut t, "interactive", 4, &opt()).unwrap();
        let mut t2 = FakeTarget::new(8, 1);
        let b = run("serve", &mut t2, "interactive", 4, &opt()).unwrap();
        assert_eq!(a.workload["arrival"], "saturated");
        assert_eq!(b.workload["arrival"], "closed_loop");
    }

    #[test]
    fn sweep_produces_one_curve_point_per_level() {
        let mut t = FakeTarget::new(64, 1);
        let art = run("sweep", &mut t, "interactive", 0, &opt()).unwrap();
        let curve = art.curve.as_ref().expect("sweep must emit a curve");
        assert_eq!(curve.len(), 3);
        assert_eq!(curve[0]["concurrency"], 1);
        assert_eq!(curve[2]["concurrency"], 4);
    }

    #[test]
    fn sweep_reports_max_sustainable_not_just_peak() {
        let mut t = FakeTarget::new(64, 1);
        let art = run("sweep", &mut t, "interactive", 0, &opt()).unwrap();
        let p = &art.performance;
        assert!(p.get("max_sustainable_concurrency").is_some());
        assert!(p.get("peak_output_artifacts_per_s").is_some());
        // The two are reported separately on purpose: peak is not the deliverable.
        assert!(render(&art).contains("max sustainable") || render(&art).contains("no concurrency level"));
    }

    #[test]
    fn best_of_n_records_repeat_count_and_spread() {
        let mut t = FakeTarget::new(8, 1);
        let mut o = opt();
        o.best_of = 3;
        let art = run("latency", &mut t, "interactive", 2, &o).unwrap();
        assert_eq!(art.best_of_n, 3);
        assert!(art.spread_pct.is_some(), "repeats must report the observed spread");
    }

    #[test]
    fn smoke_marks_the_artifact() {
        let mut t = FakeTarget::new(8, 1);
        let o = Options { num_requests: 4, warmup_requests: 0, ..Default::default() }.smoke();
        let art = run("serve", &mut t, "interactive", 2, &o).unwrap();
        assert!(art.smoke, "a smoke run must be labelled so compare can refuse to mix it");
    }

    #[test]
    fn warmup_is_excluded_from_the_reported_request_count() {
        let mut t = FakeTarget::new(8, 1);
        let o = Options { num_requests: 6, warmup_requests: 3, ..Default::default() };
        let art = run("throughput", &mut t, "interactive", 4, &o).unwrap();
        assert_eq!(art.performance["requests"], 6, "warm-up must not be counted");
    }

    #[test]
    fn results_are_unverified_until_a_gate_runs() {
        let mut t = FakeTarget::new(8, 1);
        let art = run("serve", &mut t, "interactive", 2, &opt()).unwrap();
        assert!(art.to_json()["correctness"]["passed"].is_null());
        assert!(render(&art).contains("unverified"));
    }
}
