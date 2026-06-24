// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD fuzzy in-context recall learnability guard — mirrors `tests/mqar.rs`.
//!
//! Trains the real `mad_fuzzy_recall` benchmark end-to-end on `BRAIN_DEVICE` and
//! asserts whole-group (multi-token) exact-match clears its calibrated threshold,
//! well above chance. Skipped under `MOE_SKIP_GPU_TESTS`. ~2-3 min on CPU.

use bench::mad_fuzzy_recall::MadFuzzyRecall;
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
fn mad_fuzzy_recall_above_threshold() {
    if skip() {
        return;
    }
    let bench = MadFuzzyRecall::default();
    let dir = tmpdir("mad_fuzzy_recall");
    bench.prepare(&dir, 1337).expect("prepare mad_fuzzy_recall dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate mad_fuzzy_recall");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "MAD fuzzy recall: group_em={acc:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    assert!(acc > chance * 3.0, "group_em {acc:.4} not far above chance {chance:.4}");
    assert!(
        acc >= bench.threshold(),
        "MAD fuzzy recall {acc:.4} below threshold {:.4} — group recall regressed",
        bench.threshold()
    );
}
