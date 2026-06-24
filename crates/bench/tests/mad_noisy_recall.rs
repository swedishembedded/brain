// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD noisy in-context recall learnability guard — mirrors `tests/mqar.rs`.
//!
//! Trains the real `mad_noisy_recall` benchmark end-to-end on `BRAIN_DEVICE` and
//! asserts recall amid distractor tokens clears its calibrated threshold, well
//! above chance. Skipped under `MOE_SKIP_GPU_TESTS`. ~1-2 min on the CPU backend.

use bench::mad_noisy_recall::MadNoisyRecall;
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
fn mad_noisy_recall_above_threshold() {
    if skip() {
        return;
    }
    let bench = MadNoisyRecall::default();
    let dir = tmpdir("mad_noisy_recall");
    bench.prepare(&dir, 1337).expect("prepare mad_noisy_recall dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate mad_noisy_recall");
    std::fs::remove_dir_all(&dir).ok();

    let recall = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "MAD noisy recall: recall={recall:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    assert!(recall > chance * 3.0, "recall {recall:.4} not far above chance {chance:.4}");
    assert!(
        recall >= bench.threshold(),
        "MAD noisy recall {recall:.4} below threshold {:.4} — selective recall regressed",
        bench.threshold()
    );
}
