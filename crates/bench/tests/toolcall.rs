// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tool-calling benchmark integration test — a *learnability* guard for the
//! benchmark suite, mirroring `tests/mqar.rs`.
//!
//! It runs the real `toolcall` benchmark end-to-end (synthesize data → train the
//! GPT engine on `BRAIN_DEVICE` → score exact-match of the full tool call on
//! held-out sequences) and asserts accuracy clears a calibrated threshold far
//! above chance. Skipped when `MOE_SKIP_GPU_TESTS` is set, so the suite stays
//! runnable with no accelerator. Sized to finish in minutes on the CPU backend.

use bench::toolcall::Toolcall;
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
fn toolcall_exact_match_above_threshold() {
    if skip() {
        return;
    }
    let bench = Toolcall::default();
    let dir = tmpdir("toolcall");
    bench.prepare(&dir, 1337).expect("prepare toolcall dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate toolcall");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "TOOLCALL: exact_match={acc:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    // Must be far above chance and clear the calibrated bar.
    assert!(acc > chance * 3.0, "exact_match {acc:.4} not far above chance {chance:.4}");
    assert!(
        acc >= bench.threshold(),
        "TOOLCALL exact_match {acc:.4} below threshold {:.4} — tool-call accuracy regressed",
        bench.threshold()
    );
}
