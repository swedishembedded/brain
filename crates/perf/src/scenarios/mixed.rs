// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `mixed` — traffic-class isolation.
//!
//! A real server does not receive one workload. It receives interactive chat,
//! RAG, a coding agent, batch generation and summarisation *at once*, each with
//! a different shape and a different latency contract. Aggregate tokens/s says
//! nothing about whether the interactive class still gets served while a 128k
//! summarisation prefills, and that is usually the question that matters.
//!
//! So this scenario runs several classes concurrently through one engine and
//! reports **per class**: SLO goodput, P99 TTFA/IAL, and — the honest contention
//! number — **normalised slowdown**, each class's latency under load divided by
//! the same class measured alone. Plus Jain fairness across classes and the
//! worst starvation observed.
//!
//! This is what exposes head-of-line blocking: a big prefill that stalls every
//! running decode shows up as a slowdown on the interactive class while the
//! aggregate rate looks healthy.

use serde_json::{json, Value};

use crate::driver;
use crate::metrics::{ReqRecord, Summary};
use crate::stats::{jain_fairness, r3};
use crate::target::PerfTarget;
use crate::workload::{Arrival, Class, Lengths, Slo, Workload};

/// The default class mix — the shapes a production server actually blends.
pub fn default_classes() -> Vec<Class> {
    vec![
        Class {
            name: "interactive".into(),
            input: Lengths::Fixed(128),
            output: Lengths::Fixed(128),
            weight: 4.0,
            slo: Slo::interactive(500.0, 50.0),
        },
        Class {
            name: "rag".into(),
            input: Lengths::Fixed(8192),
            output: Lengths::Fixed(256),
            weight: 2.0,
            slo: Slo::interactive(4000.0, 80.0),
        },
        Class {
            name: "agent".into(),
            input: Lengths::Fixed(16384),
            output: Lengths::Fixed(2048),
            weight: 1.0,
            // A coding agent cares about a steady stream, not first byte.
            slo: Slo { ttfa_ms: Some(10_000.0), ial_ms: Some(80.0), e2e_ms: None },
        },
        Class {
            name: "batch".into(),
            input: Lengths::Fixed(1024),
            output: Lengths::Fixed(8192),
            weight: 1.0,
            // Background work: no latency contract at all.
            slo: Slo::NONE,
        },
    ]
}

/// Scale every class down by `div` so the mix runs on a small device.
pub fn scaled_classes(div: usize) -> Vec<Class> {
    let d = div.max(1);
    let shrink = |l: Lengths| match l {
        Lengths::Fixed(n) => Lengths::Fixed((n / d).max(1)),
        other => other,
    };
    default_classes()
        .into_iter()
        .map(|mut c| {
            c.input = shrink(c.input);
            c.output = shrink(c.output);
            c
        })
        .collect()
}

/// Build the mixed workload.
pub fn workload(classes: Vec<Class>, concurrency: usize, num_requests: usize, warmup: usize, seed: u64) -> Workload {
    Workload {
        name: "mixed".into(),
        classes,
        arrival: Arrival::ClosedLoop { concurrency },
        num_requests,
        warmup_requests: warmup,
        ignore_stop: true,
        seed,
    }
}

/// A single class run on its own, for the slowdown baseline.
pub fn isolated_workload(class: &Class, num_requests: usize, seed: u64) -> Workload {
    Workload {
        name: format!("mixed/alone/{}", class.name),
        classes: vec![class.clone()],
        arrival: Arrival::ClosedLoop { concurrency: 1 },
        num_requests,
        warmup_requests: 0,
        ignore_stop: true,
        seed,
    }
}

/// Per-class results for the artifact.
pub struct ClassResult {
    pub name: String,
    pub summary: Summary,
    /// P50 end-to-end for this class measured alone, if a baseline was run.
    pub alone_e2e_ms: Option<f64>,
}

/// Split a run's records by class and summarise each against its own SLO.
pub fn split(records: &[ReqRecord], wall_s: f64, classes: &[Class], alone: &[Option<f64>]) -> Vec<ClassResult> {
    classes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mine: Vec<ReqRecord> = records.iter().filter(|r| r.class == i).cloned().collect();
            ClassResult {
                name: c.name.clone(),
                summary: Summary::build(&mine, wall_s, c.slo),
                alone_e2e_ms: alone.get(i).copied().flatten(),
            }
        })
        .collect()
}

/// The `per_class` block plus the cross-class fairness numbers.
pub fn to_json(results: &mut [ClassResult]) -> (Vec<Value>, Value) {
    let mut blocks = Vec::new();
    let mut rates = Vec::new();
    for r in results.iter_mut() {
        let alone = r.alone_e2e_ms;
        let loaded_p50 = r.summary.e2e.percentile(0.50);
        // The number that says what contention actually cost this class.
        let slowdown = match (alone, loaded_p50) {
            (Some(a), Some(l)) if a > 0.0 => Value::from(r3(l / a)),
            _ => Value::Null,
        };
        rates.push(r.summary.goodput_per_s);
        let mut perf = r.summary.to_json();
        perf["normalised_slowdown"] = slowdown;
        perf["alone_e2e_ms_p50"] = alone.map(|v| Value::from(r3(v))).unwrap_or(Value::Null);
        blocks.push(json!({ "class": r.name, "performance": perf }));
    }
    let starvation = results
        .iter_mut()
        .filter_map(|r| r.summary.queue.max())
        .fold(None::<f64>, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))));
    let fairness = json!({
        // Fairness over per-class GOODPUT, not raw throughput: a class being fed
        // tokens that all miss its deadline is not being served.
        "jain_fairness": jain_fairness(&rates).map(Value::from).unwrap_or(Value::Null),
        "starvation_ms_max": starvation.map(|v| Value::from(r3(v))).unwrap_or(Value::Null),
        "classes": results.len(),
    });
    (blocks, fairness)
}

/// Run the isolated baselines that make `normalised_slowdown` meaningful.
pub fn baselines(target: &mut dyn PerfTarget, classes: &[Class], num_requests: usize, seed: u64) -> Vec<Option<f64>> {
    classes
        .iter()
        .map(|c| {
            target.reset(true);
            let w = isolated_workload(c, num_requests.max(1), seed);
            let run = driver::drive(target, &w);
            let mut s = Summary::build(&run.records, run.wall_s, c.slo);
            s.e2e.percentile(0.50)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::testing::FakeTarget;
    use std::time::{Duration, Instant};

    fn rec(id: u64, class: usize, e2e_ms: u64) -> ReqRecord {
        let base = Instant::now();
        let mut r = ReqRecord::new(id, class, 8, false, base);
        r.admit = Some(base);
        r.first = Some(base);
        r.artifacts = vec![base, base + Duration::from_millis(e2e_ms)];
        r.done = Some(base + Duration::from_millis(e2e_ms));
        r
    }

    #[test]
    fn default_mix_spans_latency_sensitive_and_background_classes() {
        let cs = default_classes();
        assert_eq!(cs.len(), 4);
        assert!(cs[0].slo.ttfa_ms.is_some(), "interactive must have a TTFA contract");
        assert_eq!(cs[3].slo, Slo::NONE, "batch work must have no latency contract");
    }

    #[test]
    fn records_are_split_by_class() {
        let classes = default_classes();
        let recs = vec![rec(0, 0, 10), rec(1, 0, 20), rec(2, 1, 30)];
        let out = split(&recs, 1.0, &classes, &[None, None, None, None]);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].summary.requests, 2);
        assert_eq!(out[1].summary.requests, 1);
        assert_eq!(out[3].summary.requests, 0, "an unused class reports zero, not a panic");
    }

    #[test]
    fn slowdown_is_loaded_over_alone() {
        let classes = default_classes();
        let recs = vec![rec(0, 0, 100)];
        let mut out = split(&recs, 1.0, &classes, &[Some(25.0), None, None, None]);
        let (blocks, _) = to_json(&mut out);
        // 100ms under load vs 25ms alone => 4x
        assert_eq!(blocks[0]["performance"]["normalised_slowdown"], 4.0);
    }

    #[test]
    fn slowdown_is_null_without_a_baseline() {
        let classes = default_classes();
        let recs = vec![rec(0, 0, 100)];
        let mut out = split(&recs, 1.0, &classes, &[None, None, None, None]);
        let (blocks, _) = to_json(&mut out);
        assert!(blocks[0]["performance"]["normalised_slowdown"].is_null());
    }

    #[test]
    fn fairness_is_computed_over_goodput() {
        let classes = default_classes();
        // Class 0 served, class 1 starved.
        let recs = vec![rec(0, 0, 10), rec(1, 0, 10)];
        let mut out = split(&recs, 1.0, &classes, &[None; 4]);
        let (_, fair) = to_json(&mut out);
        let j = fair["jain_fairness"].as_f64().unwrap();
        assert!(j < 0.6, "one class getting everything must read as unfair, got {j}");
    }

    #[test]
    fn baselines_run_each_class_alone() {
        let mut t = FakeTarget::new(8, 1);
        let classes = scaled_classes(1024);
        let b = baselines(&mut t, &classes, 2, 1);
        assert_eq!(b.len(), classes.len());
        assert!(b.iter().all(|x| x.is_some()), "every class needs an alone baseline");
    }

    #[test]
    fn scaled_classes_shrink_but_keep_the_contracts() {
        let full = default_classes();
        let small = scaled_classes(64);
        assert_eq!(small.len(), full.len());
        assert_eq!(small[0].slo, full[0].slo, "scaling must not change the SLO");
        assert!(matches!(small[1].input, Lengths::Fixed(n) if n == 8192 / 64));
    }
}
