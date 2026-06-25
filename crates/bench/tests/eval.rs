// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Architecture-eval harness integration test — the turn-key path end-to-end.
//!
//! It runs `eval` for the `gpt` architecture in **smoke** mode (every benchmark
//! at a slashed step/corpus budget — see `bench::registry_smoke`) so it finishes
//! in a couple of minutes on the CPU backend, then asserts the results artifact:
//!   * is written to disk and re-parses,
//!   * carries every populated capability axis with a **finite** score,
//!   * and has the per-benchmark / gating fields the `compare` leaderboard reads.
//!
//! Smoke scores are NOT meaningful as architecture quality (the budget is tiny);
//! this test guards the *harness*, not the model. Skipped when `MOE_SKIP_GPU_TESTS`
//! is set, like the rest of the suite.

use std::path::PathBuf;

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

#[test]
fn eval_gpt_smoke_writes_valid_artifact() {
    if skip() {
        return;
    }

    let seed = 4242u64;
    // Run the WHOLE battery against `gpt` in smoke mode.
    let report = bench::eval::run("gpt", seed, true).expect("eval run");

    // Every axis that has a benchmark must be populated with a finite score.
    assert!(!report.axis_scores.is_empty(), "no capability axes populated");
    for ax in bench::axes() {
        // Every canonical axis maps to ≥1 registered benchmark, so all are present.
        let score = report
            .axis_scores
            .get(ax)
            .copied()
            .unwrap_or_else(|| panic!("axis '{ax}' missing from eval report"));
        assert!(score.is_finite(), "axis '{ax}' score not finite: {score}");
    }

    // Gating summary is sane.
    assert!(report.gating_total > 0, "no gating benchmarks");
    assert!(report.gating_passed <= report.gating_total);

    // Write + re-load the artifact.
    let out = std::env::temp_dir().join(format!("brain_eval_test_{}.json", std::process::id()));
    bench::eval::write_artifact(&report, &out).expect("write artifact");
    let loaded = bench::eval::load_artifact(&out).expect("reload artifact");
    std::fs::remove_file(&out).ok();

    assert_eq!(loaded.arch, "gpt");
    assert_eq!(loaded.seed, seed);
    assert!(loaded.param_count > 0, "param_count not recorded");
    assert!(loaded.gating_pass_rate.is_finite());
    // Round-trip preserved every axis.
    for ax in bench::axes() {
        let v = loaded.axis_scores.get(ax).copied();
        assert!(v.map(|x| x.is_finite()).unwrap_or(false), "axis '{ax}' lost on reload");
    }
    // Per-benchmark scores survived for the compare leaderboard.
    assert!(loaded.bench_scores.contains_key("mqar"), "mqar score missing");
    assert_eq!(loaded.bench_scores.len(), report.benchmarks.len());
}

#[test]
fn compare_two_artifacts_runs() {
    if skip() {
        return;
    }
    // Two cheap smoke runs (gpt at two seeds) is enough to exercise compare.
    let r1 = bench::eval::run("gpt", 1, true).expect("eval 1");
    let r2 = bench::eval::run("gpt-small", 2, true).expect("eval 2");
    let p1 = std::env::temp_dir().join(format!("brain_cmp1_{}.json", std::process::id()));
    let p2 = std::env::temp_dir().join(format!("brain_cmp2_{}.json", std::process::id()));
    bench::eval::write_artifact(&r1, &p1).unwrap();
    bench::eval::write_artifact(&r2, &p2).unwrap();
    bench::eval::compare(&[p1.clone(), p2.clone()]).expect("compare ok");
    std::fs::remove_file(&p1).ok();
    std::fs::remove_file(&p2).ok();
}

#[test]
fn compare_needs_two_artifacts() {
    // Pure-CPU guard (no training): one path is an error.
    let one = vec![PathBuf::from("results/does-not-matter.json")];
    assert!(bench::eval::compare(&one).is_err());
}
