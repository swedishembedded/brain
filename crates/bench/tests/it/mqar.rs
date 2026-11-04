// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MQAR benchmark integration test — a *learnability* guard for the benchmark
//! suite, mirroring `crates/gpt/tests/convergence.rs`.
//!
//! It runs the real MQAR benchmark end-to-end (synthesize data → train the GPT
//! engine on `BRAIN_DEVICE` → score associative recall on held-out sequences)
//! and asserts recall clears a calibrated threshold far above chance. Skipped
//! when `MOE_SKIP_GPU_TESTS` is set, so the suite stays runnable with no
//! accelerator. Sized to finish in a couple of minutes on the CPU backend.

use bench::mqar::Mqar;
use bench::Benchmark;

/// Skip the whole test when no accelerator is wanted (same gate as the rest).
fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain_bench_test_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create temp dataset dir");
    d
}

#[test]
fn mqar_recall_above_threshold() {
    if skip() {
        return;
    }
    let bench = Mqar::default();
    let dir = tmpdir("mqar");
    bench.prepare(&dir, 1337).expect("prepare mqar dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate mqar");
    std::fs::remove_dir_all(&dir).ok();

    let recall = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "MQAR: recall={recall:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    // Must be far above chance and clear the calibrated bar.
    assert!(recall > chance * 3.0, "recall {recall:.4} not far above chance {chance:.4}");
    assert!(
        recall >= bench.threshold(),
        "MQAR recall {recall:.4} below threshold {:.4} — associative recall regressed",
        bench.threshold()
    );
}
