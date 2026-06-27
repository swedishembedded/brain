// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-decoder learnability guard: trains the Qwen architecture (GQA + QK-norm
//! + RoPE + SwiGLU) from scratch on the `toolcall` exact-match benchmark through
//! the architecture-agnostic `DecoderLm` seam, and asserts held-out tool-call
//! exact-match clears the calibrated threshold — objective proof the engine both
//! trains and infers the Qwen architecture correctly on concrete tasks.
//! Skipped when `MOE_SKIP_GPU_TESTS` is set. ~1-2 min on the CPU backend.

use bench::toolcall::Toolcall;
use bench::{Benchmark, QwenDecoder};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain_bench_qwen_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create temp dataset dir");
    d
}

#[test]
fn qwen_learns_toolcall_above_threshold() {
    if skip() {
        return;
    }
    let bench = Toolcall::default();
    let dir = tmpdir("toolcall");
    bench.prepare(&dir, 1337).expect("prepare toolcall dataset");
    let metrics = bench.evaluate_with(&QwenDecoder, &dir, 1337).expect("evaluate qwen toolcall");
    std::fs::remove_dir_all(&dir).ok();

    let acc = metrics.score;
    let chance = metrics.get("chance").unwrap();
    println!(
        "QWEN/TOOLCALL: exact_match={acc:.4} chance={chance:.4} threshold={:.4}",
        bench.threshold()
    );
    assert!(acc > chance * 3.0, "exact_match {acc:.4} not far above chance {chance:.4}");
    assert!(
        acc >= bench.threshold(),
        "QWEN toolcall exact_match {acc:.4} below threshold {:.4}",
        bench.threshold()
    );
}
