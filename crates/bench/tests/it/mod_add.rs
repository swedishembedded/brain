// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Modular-addition benchmark integration test — a *learnability* guard
//! mirroring `tests/mqar.rs`.
//!
//! Runs the mod_add benchmark end-to-end (synthesize a held-out partition of the
//! `a+b=c (mod p)` fact table → train the GPT engine on `BRAIN_DEVICE` → score
//! test-fact accuracy) and asserts accuracy clears its threshold an order of
//! magnitude above the `1/p` chance. Skipped when `MOE_SKIP_GPU_TESTS` is set.
//!
//! mod_add is the grokking task and is the slowest of the formal-language
//! benchmarks: generalizing on a held-out fact partition needs the full
//! d_model-128 model and a few thousand steps. The test uses a smaller modulus
//! (`p=17`) and fewer steps than the registered default to keep it to minutes on
//! the CPU backend while still generalizing to ~0.79 test accuracy.

use bench::mod_add::ModAdd;
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
fn mod_add_test_accuracy_above_threshold() {
    if skip() {
        return;
    }
    // Lighter than the registered default (p=17 / 2000 steps) so the test stays
    // within budget; measured ~0.79 test accuracy at this size on the CPU
    // backend. The d_model-128 width is load-bearing — shrinking it stops the
    // model generalizing to held-out facts at all (it only memorizes).
    let bench = ModAdd { p: 17, steps: 2000, d_model: 128, eval_facts: 120, ..ModAdd::default() };
    let dir = tmpdir("mod_add");
    bench.prepare(&dir, 1337).expect("prepare mod_add dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate mod_add");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "mod_add: test_accuracy={acc:.4} chance={chance:.4} train_ce={:.4} threshold={:.4}",
        metrics.get("train_ce").unwrap(),
        bench.threshold()
    );
    // Generalization, not memorization: must be far above the 1/p chance.
    assert!(acc > chance * 3.0, "test accuracy {acc:.4} not far above chance {chance:.4}");
    assert!(
        acc >= bench.threshold(),
        "mod_add test accuracy {acc:.4} below threshold {:.4} — modular structure not learned",
        bench.threshold()
    );
}
