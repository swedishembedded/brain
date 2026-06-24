// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD memorization learnability guard — mirrors `tests/mqar.rs`.
//!
//! Trains the real `mad_memorize` benchmark end-to-end on `BRAIN_DEVICE` and
//! asserts recall of the fixed weight-stored key->value map clears its calibrated
//! threshold, far above chance. Skipped under `MOE_SKIP_GPU_TESTS`. ~20-40 s on
//! the CPU backend.

use bench::mad_memorize::MadMemorize;
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
fn mad_memorize_above_threshold() {
    if skip() {
        return;
    }
    let bench = MadMemorize::default();
    let dir = tmpdir("mad_memorize");
    bench.prepare(&dir, 1337).expect("prepare mad_memorize dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate mad_memorize");
    std::fs::remove_dir_all(&dir).ok();

    let recall = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "MAD memorize: recall={recall:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    assert!(recall > chance * 3.0, "recall {recall:.4} not far above chance {chance:.4}");
    assert!(
        recall >= bench.threshold(),
        "MAD memorize {recall:.4} below threshold {:.4} — weight memorization regressed",
        bench.threshold()
    );
}
