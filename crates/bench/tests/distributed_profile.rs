// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The single-machine distributed profile end to end: two ranks of the tiny
//! GPT train over the host collective, the report parses, the cross-run ratios
//! are present, and the replicas agree bit for bit.
//!
//! It selects the CPU backend itself, so it runs anywhere and never skips.

use bench::distributed::{run_profile, ProfileConfig, Transport};
use serde_json::Value;

#[test]
fn profile_reports_speedup_and_bit_identical_replicas() {
    let cfg = ProfileConfig {
        steps: 2,
        worlds: vec![1, 2],
        transports: vec![Transport::Host],
        dataset_examples: 800,
        ..Default::default()
    };
    let report = run_profile(&cfg).expect("the profile runs on the CPU backend");
    report_config_is_complete(&report);

    let runs = report["runs"].as_array().expect("the report has a run list");
    assert_eq!(runs.len(), 2, "one run per configured transport/world pair");
    let world1 = find_run(runs, 1);
    let world2 = find_run(runs, 2);

    for run in runs {
        assert_eq!(run["replicas_bit_identical"], Value::Bool(true), "replicas must end bit-identical");
        let times = run["step_times_ms"].as_array().expect("per-step times");
        assert_eq!(times.len(), cfg.steps as usize, "one interval per configured step");
        assert!(times.iter().all(|t| t.as_f64().is_some_and(|t| t > 0.0)), "step intervals must be positive");
        assert!(run["tokens_per_s"].as_f64().is_some_and(|v| v > 0.0), "throughput must be reported");
    }

    // A world-1 run is its own baseline; the world-2 run is measured against it.
    assert_eq!(world1["speedup"].as_f64(), Some(1.0));
    let speedup = world2["speedup"].as_f64().expect("world 2 has a speedup against world 1");
    assert!(speedup.is_finite() && speedup > 0.0, "speedup must be a positive finite number, got {speedup}");
    let efficiency = world2["parallel_efficiency"].as_f64().expect("parallel efficiency");
    assert!((efficiency - speedup / 2.0).abs() < 1e-9, "efficiency is speedup / world");

    // Communication is accounted for the distributed run.
    let comm = &world2["comm"];
    assert!(comm["bytes"].as_u64().is_some_and(|bytes| bytes > 0), "the reduction moves gradient bytes");
    assert!(comm["ops"]["all_reduce"]["count"].as_u64().is_some_and(|count| count > 0), "gradients are all-reduced");
    assert!(comm["time_ms"].as_f64().is_some_and(|ms| ms >= 0.0));
}

fn find_run(runs: &[Value], world: u64) -> &Value {
    runs.iter().find(|run| run["world"].as_u64() == Some(world)).unwrap_or_else(|| panic!("no world {world} run"))
}

fn report_config_is_complete(report: &Value) {
    let config = &report["config"];
    assert_eq!(config["backend"], Value::from("cpu"));
    assert_eq!(config["steps"], Value::from(2));
    assert_eq!(config["model"]["d_model"], Value::from(32));
    assert_eq!(config["model"]["d_ff"], Value::from(64));
    assert!(config["link"]["one_way_latency_us"].as_f64().is_some());
}
