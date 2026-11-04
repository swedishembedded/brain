// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD compression benchmark integration test — a *learnability* guard for the
//! bottleneck-autoencoder task (ADR §6 / PR-10), mirroring `tests/mqar.rs`.
//!
//! It runs the real `mad_compress` benchmark end-to-end (synthesize token corpus
//! → map tokens to codebook features → train the autoencoder with the MSE
//! `Regression` head on `BRAIN_DEVICE` → score nearest-codebook reconstruction on
//! held-out sequences) and asserts reconstruction accuracy clears a calibrated
//! threshold far above chance. Skipped when `MOE_SKIP_GPU_TESTS` is set, so the
//! suite stays runnable with no accelerator. Sized to finish in a couple of
//! minutes on the CPU backend.

use bench::mad_compress::MadCompress;
use bench::{known_names, Benchmark};

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
fn mad_compress_is_registered() {
    // The task now has a real model (autoencoder) and is part of the live suite.
    assert!(known_names().contains(&"mad_compress".to_string()));
}

#[test]
fn mad_compress_reconstruction_above_threshold() {
    if skip() {
        return;
    }
    let bench = MadCompress::default();
    let dir = tmpdir("mad_compress");
    bench.prepare(&dir, 1337).expect("prepare mad_compress dataset");
    let metrics = bench.evaluate(&dir, 1337).expect("evaluate mad_compress");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "mad_compress: recon_acc={acc:.4} chance={chance:.4} final_mse={:.4} threshold={:.4}",
        metrics.get("final_mse").unwrap(),
        bench.threshold(),
    );
    assert!(acc > chance * 2.0, "reconstruction not above chance: {acc} vs chance {chance}");
    assert!(
        acc >= bench.threshold(),
        "reconstruction {acc} below threshold {}",
        bench.threshold()
    );
}
