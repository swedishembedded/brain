// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity benchmark integration test — a *learnability* guard mirroring
//! `tests/mqar.rs`.
//!
//! Runs the parity benchmark end-to-end (synthesize bit strings → train the GPT
//! engine on `BRAIN_DEVICE` → score running-parity next-token accuracy on
//! held-out sequences) and asserts accuracy clears its threshold far above the
//! 0.5 coin flip. Skipped when `MOE_SKIP_GPU_TESTS` is set. A reduced config
//! (fewer steps / sequences than the registered default) keeps it to about a
//! minute on the CPU backend while still landing at ~1.0.

use bench::parity::Parity;
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
fn parity_accuracy_above_threshold() {
    if skip() {
        return;
    }
    // Lighter than the registered default (500 steps / 4000 sequences) so the
    // test stays fast; measured ~1.0 at this size on the CPU backend.
    let bench = Parity { steps: 500, n_sequences: 4000, eval_sequences: 150, ..Parity::default() };
    let dir = tmpdir("parity");
    bench.prepare(&dir, 1337).expect("prepare parity dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate parity");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "parity: accuracy={acc:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    assert!(acc > chance * 1.5, "accuracy {acc:.4} not clearly above chance {chance:.4}");
    assert!(
        acc >= bench.threshold(),
        "parity accuracy {acc:.4} below threshold {:.4} — state tracking regressed",
        bench.threshold()
    );
}
