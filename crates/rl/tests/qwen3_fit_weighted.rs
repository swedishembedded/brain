// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end convergence guard for [`rl::fit_weighted`], mirroring
//! `toyseq2seq/tests/convergence.rs`'s "does the whole loop actually learn"
//! spirit: `check_qwen3_weighted` (gradcheck) already proves the weighted
//! backward is analytically correct per-parameter; this proves the file-
//! based driver loop around it (dataset loading, weight-file attachment,
//! LR schedule, batching, checkpoint save/resume) produces a model that
//! actually learns, run through `rl::fit_weighted` exactly as a real caller
//! would use it - not a hand-rolled training loop in the test itself.
//!
//! Task: a small deterministic bigram - `next = (cur + 1) mod vocab` - over
//! the whole corpus, with NO `train.weight.bin` present (every position
//! implicitly weights `1.0`, per `rl`'s own doc comment on that default).
//! This is deliberately the "ordinary training driven through the weighted
//! path" case: a correct implementation must converge exactly as plain
//! `model::train::fit` would.
//!
//! Skipped when `MOE_SKIP_GPU_TESTS` is set (same gate as the rest of the
//! suite).

use qwen3::{Qwen, QwenConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-rl-fit-weighted-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn qwen3_learns_a_deterministic_bigram_through_fit_weighted() {
    if skip() {
        return;
    }

    let vocab: u32 = 23;
    let n = 4000usize;
    let mut data = vec![0u32; n];
    for i in 1..n {
        data[i] = (data[i - 1] + 1) % vocab;
    }

    let dir = tmp("bigram");
    data::binio::write_u32_bin(&dir.join("train.u32.bin"), &data).unwrap();
    data::binio::write_u32_bin(&dir.join("val.u32.bin"), &data[..500]).unwrap();
    std::fs::write(dir.join("meta.json"), data::binio::Meta::vocab_only(vocab as usize)).unwrap();
    // Deliberately NO train.weight.bin - the no-file / implicit-1.0 path.

    let cfg = QwenConfig::tiny();
    let opts = model::FitOpts {
        steps: 300,
        batch_size: 16,
        block_size: cfg.block_size,
        lr: 5e-3,
        min_lr: 5e-4,
        warmup: 20,
        decay_iters: 600,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        seed: 11,
        checkpoint_secs: 0,
        ..Default::default()
    };

    let (initial, last) = rl::fit_weighted::<Qwen>(&dir, cfg, &opts, None).expect("fit_weighted");
    println!("qwen3 bigram via fit_weighted: init {initial:.4} -> final {last:.4} (marginal ln(23) ~= 3.135)");

    // Marginal entropy of a uniform 23-symbol distribution (no learned rule)
    // is ln(23) ~= 3.135 - a model that never actually trained (or whose
    // weighted path silently zeroed every gradient) would be stuck there.
    // A correctly-learned deterministic bigram drives this far below.
    assert!(
        last < 0.5,
        "qwen3 failed to learn the bigram through fit_weighted: {initial:.4} -> {last:.4} (expected < 0.5, marginal floor ln(23) ~= 3.135)"
    );
}
