// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD selective copying learnability guard — mirrors `tests/mqar.rs`.
//!
//! Trains the real `mad_selective_copy` benchmark end-to-end on `BRAIN_DEVICE`
//! and asserts whole-group copy exact-match clears its calibrated threshold, far
//! above chance. Skipped under `MOE_SKIP_GPU_TESTS`. Minutes on CPU.

use bench::mad_selective_copy::MadSelectiveCopy;
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
fn mad_selective_copy_above_threshold() {
    if skip() {
        return;
    }
    let bench = MadSelectiveCopy::default();
    let dir = tmpdir("mad_selective_copy");
    bench.prepare(&dir, 1337).expect("prepare mad_selective_copy dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate mad_selective_copy");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "MAD selective copy: exact_match={acc:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    assert!(acc > chance * 3.0, "exact_match {acc:.4} not far above chance {chance:.4}");
    assert!(
        acc >= bench.threshold(),
        "MAD selective copy {acc:.4} below threshold {:.4} — selective copying regressed",
        bench.threshold()
    );
}
