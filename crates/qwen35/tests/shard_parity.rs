// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic pipeline-parallel sharding (`model::Pipeline<Qwen35>`) is bit-exact
//! against the single-device model. Mirrors `qwen35moe/tests/shard_parity.rs`
//! exactly, adapted for this crate's dense-MLP config.
//!
//! `Qwen35Config::tiny()` (`full_attention_interval=4`, `n_layers=4`)
//! exercises BOTH layer types (Gated DeltaNet at 0,1,2, GQA at 3) on both
//! sides of whatever cut `plan_balanced` picks. The sharded forward loss and
//! every per-parameter gradient must match the whole-model reference; `tok.
//! weight`/`lm_head.weight` are untied for this model, so no
//! replicated-gradient summing is exercised here. A sharded overfit run must
//! reduce the loss - end-to-end forward+backward+optimiser across stages,
//! with the cut placed automatically by `plan_balanced`.
//!
//! Two stages on GPUs 0 and 1 by default; `SHARD_TEST_GPUS=1,1` pins both to
//! one card. Skipped under `MOE_SKIP_GPU_TESTS` or on a box without enough
//! discrete GPUs.

use model::{Batch, Pipeline};
use qwen35::config::Qwen35Config;
use qwen35::model::Qwen35;

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    let need = stage_gpus().iter().copied().max().unwrap_or(0) + 1;
    let have = gpu_core::discrete_gpu_count();
    if have < need {
        brain_testutil::skip_unavailable(&format!("needs {need} discrete GPU(s), found {have}"));
        return true;
    }
    false
}

fn stage_gpus() -> Vec<usize> {
    std::env::var("SHARD_TEST_GPUS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0, 1])
}

fn batch(cfg: &Qwen35Config, b: u32, t: u32, seed: u32) -> (Vec<u32>, Vec<u32>) {
    let n = b * t;
    let x = (0..n).map(|i| (i * 3 + seed) % cfg.vocab).collect();
    let y = (0..n).map(|i| (i * 3 + 1 + seed) % cfg.vocab).collect();
    (x, y)
}

#[test]
fn shard_forward_and_grad_parity() {
    if gpu_disabled() {
        return;
    }
    let cfg = Qwen35Config::tiny(); // both layer types; untied head
    let init = qwen35::init::init_weights(&cfg, 7);
    let (b, t) = (1u32, cfg.block_size);
    let (x, y) = batch(&cfg, b, t, 0);

    // Single-device reference.
    let single = Qwen35::new_train_on(gpu_core::Gpu::new(qwen35::model::PIPELINES), cfg.clone(), b, t, &init);
    single.set_batch(&x, &y);
    single.zero_grads();
    let l_single = single.forward();
    single.backward();
    single.poll_wait();

    // Auto-placed two-stage pipeline from the SAME weights.
    let pipe = Pipeline::<Qwen35>::new(cfg.clone(), b, t, &init, &stage_gpus());
    assert_eq!(pipe.n_stages(), 2);
    eprintln!("auto-placed shards: {:?}", pipe.shards());
    pipe.zero_grads();
    let l_pipe = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
    pipe.backward();

    let dl = (l_single - l_pipe).abs() / l_single.abs().max(1e-6);
    eprintln!("loss  single={l_single:.6}  pipe={l_pipe:.6}  rel={dl:.2e}");
    assert!(dl < 1e-4, "forward loss mismatch: {l_single} vs {l_pipe}");

    // Every parameter gradient.
    let mut worst = 0f32;
    let mut worst_name = String::new();
    for (name, _) in cfg.param_list() {
        let a = single.read_grad(&name);
        let g = pipe.reduced_grad(&name);
        assert_eq!(a.len(), g.len(), "grad len mismatch for {name}");
        let (mut num, mut den) = (0f32, 1e-6f32);
        for (p, q) in a.iter().zip(&g) {
            num = num.max((p - q).abs());
            den = den.max(p.abs());
        }
        let rel = num / den;
        if rel > worst {
            worst = rel;
            worst_name = name.clone();
        }
        assert!(rel < 1e-3, "grad mismatch for {name}: rel {rel:.2e}");
    }
    eprintln!("worst grad rel {worst:.2e} ({worst_name})");
}

#[test]
fn shard_overfit_reduces_loss() {
    if gpu_disabled() {
        return;
    }
    let cfg = Qwen35Config::tiny();
    let init = qwen35::init::init_weights(&cfg, 11);
    let (b, t) = (1u32, cfg.block_size);
    let (x, y) = batch(&cfg, b, t, 5);

    let mut pipe = Pipeline::<Qwen35>::new(cfg, b, t, &init, &stage_gpus());
    let before = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
    for step in 1..=50 {
        pipe.zero_grads();
        pipe.forward(Batch::Lm { tokens: &x, targets: &y });
        pipe.backward();
        pipe.adamw_step(step, 1e-2, 0.0, Some(1.0), 1.0);
    }
    let after = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
    eprintln!("sharded overfit  loss {before:.4} -> {after:.4}");
    assert!(after < before, "sharded overfit did not reduce loss: {before} -> {after}");
}
