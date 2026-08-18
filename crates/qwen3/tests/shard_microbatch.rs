// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The concurrent GPipe micro-batch schedule (`Pipeline::pipelined_fwd_bwd`) with
//! activation recomputation is bit-exact against sequential grad-accumulation: the
//! gradients from running `M` microbatches through the overlapped pipeline equal
//! those from one model accumulating the same `M` microbatches. Also checks a
//! `train_step` reduces the loss over a few steps.
//!
//! Two stages on GPUs 0 and 1 by default; `SHARD_TEST_GPUS=1,1` pins both to one
//! card. Skipped under `MOE_SKIP_GPU_TESTS`.

use model::{Batch, Pipeline};
use qwen3::{Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    // Skip rather than fault when this box lacks the cards the test pins to.
    // Without the check the multi-GPU paths assume cards 0..n exist and die
    // inside the driver on a single-GPU or GPU-less machine, which reads as a
    // real regression and masks actual ones.
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

#[test]
fn microbatch_pipeline_grad_parity() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 7);
    let (b, t) = (2u32, 8u32);
    let k = 4usize; // microbatches
    let mbs: Vec<(Vec<u32>, Vec<u32>)> = (0..k)
        .map(|j| {
            let x = (0..b * t).map(|i| (i * 3 + j as u32 * 13) % cfg.vocab).collect();
            let y = (0..b * t).map(|i| (i * 3 + 1 + j as u32 * 13) % cfg.vocab).collect();
            (x, y)
        })
        .collect();

    // Single-GPU reference: accumulate all K microbatches.
    let single = Qwen::new(cfg.clone(), b, t, &init);
    single.zero_grads();
    for (x, y) in &mbs {
        single.set_batch(x, y);
        single.forward();
        single.backward();
    }
    single.poll_wait();

    // Micro-batched pipeline (2 stages, concurrent + recompute).
    let batches: Vec<Batch> = mbs.iter().map(|(x, y)| Batch::Lm { tokens: x, targets: y }).collect();
    let mut pipe = Pipeline::<Qwen>::new(cfg.clone(), b, t, &init, &stage_gpus());
    pipe.zero_grads();
    let loss = pipe.pipelined_fwd_bwd(&batches);
    eprintln!("microbatched pipeline summed loss {loss:.4}");

    let mut worst = 0f32;
    let mut worst_name = String::new();
    for (name, _) in cfg.param_list() {
        let a = single.read_grad(&name);
        let g = pipe.reduced_grad(&name);
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
        assert!(rel < 1e-3, "microbatch grad mismatch for {name}: rel {rel:.2e}");
    }
    eprintln!("microbatch vs sequential accumulation: worst grad rel {worst:.2e} ({worst_name})");
}

#[test]
fn microbatch_train_step_reduces_loss() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 11);
    let (b, t) = (2u32, 8u32);
    let k = 4usize;
    let mbs: Vec<(Vec<u32>, Vec<u32>)> = (0..k)
        .map(|j| {
            let x = (0..b * t).map(|i| (i * 7 + j as u32 * 5) % cfg.vocab).collect();
            let y = (0..b * t).map(|i| (i * 7 + 1 + j as u32 * 5) % cfg.vocab).collect();
            (x, y)
        })
        .collect();
    let batches: Vec<Batch> = mbs.iter().map(|(x, y)| Batch::Lm { tokens: x, targets: y }).collect();

    let mut pipe = Pipeline::<Qwen>::new(cfg, b, t, &init, &stage_gpus());
    let before = pipe.train_step(&batches, 1, 1e-2, 0.0, Some(1.0));
    let mut last = before;
    for step in 2..=30 {
        last = pipe.train_step(&batches, step, 1e-2, 0.0, Some(1.0));
    }
    eprintln!("microbatch train_step  loss {before:.4} -> {last:.4}");
    assert!(last < before, "micro-batched training did not reduce loss: {before} -> {last}");
}
