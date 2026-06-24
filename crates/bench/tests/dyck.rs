// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dyck-k benchmark integration test — a *learnability* guard mirroring
//! `tests/mqar.rs`.
//!
//! Runs the Dyck benchmark end-to-end (synthesize balanced bracket words → train
//! the GPT engine on `BRAIN_DEVICE` → score close-bracket next-token accuracy on
//! held-out words) and asserts accuracy clears its threshold far above the `1/k`
//! chance. Skipped when `MOE_SKIP_GPU_TESTS` is set. A reduced config (fewer
//! steps / sequences / width than the registered default) keeps it ~1-2 min on
//! the CPU backend while still landing at ~0.99.

use bench::dyck::Dyck;
use bench::Benchmark;

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
fn dyck_accuracy_above_threshold() {
    if skip() {
        return;
    }
    // Lighter than the registered default (600 steps / 4000 words / d_model 64)
    // so the test stays fast; measured ~0.99 at this size on the CPU backend.
    let bench = Dyck { steps: 600, n_sequences: 4000, d_model: 64, eval_sequences: 150, ..Dyck::default() };
    let dir = tmpdir("dyck");
    bench.prepare(&dir, 1337).expect("prepare dyck dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate dyck");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "dyck: accuracy={acc:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    assert!(acc > chance * 1.5, "accuracy {acc:.4} not clearly above chance {chance:.4}");
    assert!(
        acc >= bench.threshold(),
        "dyck accuracy {acc:.4} below threshold {:.4} — hierarchical state tracking regressed",
        bench.threshold()
    );
}
