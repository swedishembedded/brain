// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Data-parallel training is bit-exact against single-GPU grad accumulation.
//!
//! Running `K` micro-batches split across 2 replicas + all-reduce must produce
//! the identical accumulated gradient as one GPU running the same `K`
//! micro-batches with grad-accum `K`. (This also confirms the backward pass
//! accumulates into the grad buffers, which grad-accum depends on.)
//!
//! Two replicas on the first two selected cards (BRAIN_DEVICE) by default; `SHARD_TEST_GPUS=1,1` pins both to
//! one card. Skipped under `MOE_SKIP_GPU_TESTS`.

use model::{Batch, DataParallel};
use qwen3::{Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return true;
    }
    // Skip rather than fault when this box lacks the cards the test pins to.
    // Without the check the multi-GPU paths assume cards 0..n exist and die
    // inside the driver on a single-GPU or GPU-less machine, which reads as a
    // real regression and masks actual ones.
    if let Err(why) = stage_cards() {
        brain_testutil::skip_unavailable(&why);
        return true;
    }
    false
}

fn stage_gpus() -> Vec<usize> {
    stage_cards().unwrap_or_default()
}

/// The two cards the stages go on: `SHARD_TEST_GPUS` when set (each one a card the selection names), else the first two
/// cards `BRAIN_DEVICE` selects; never cards the selection leaves to someone else.
fn stage_cards() -> Result<Vec<usize>, String> {
    let requested = std::env::var("SHARD_TEST_GPUS").ok().map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<usize>>()).filter(|v| !v.is_empty());
    gpu_core::testgpu::stage_cards(requested.as_deref(), 2)
}

#[test]
fn dp_grad_parity() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen3::init_weights(&cfg, 7);
    let (b, t) = (2u32, 8u32);

    // K micro-batches (distinct data each).
    let k = 4usize;
    let mbs: Vec<(Vec<u32>, Vec<u32>)> = (0..k)
        .map(|j| {
            let x = (0..b * t).map(|i| (i * 3 + j as u32 * 13) % cfg.vocab).collect();
            let y = (0..b * t).map(|i| (i * 3 + 1 + j as u32 * 13) % cfg.vocab).collect();
            (x, y)
        })
        .collect();

    // Single-GPU reference: accumulate all K micro-batches.
    let single = Qwen::new(cfg.clone(), b, t, &init);
    single.zero_grads();
    for (x, y) in &mbs {
        single.set_batch(x, y);
        single.forward();
        single.backward();
    }
    single.poll_wait();

    // Data-parallel over 2 GPUs (generic over Model): same K micro-batches.
    let batches: Vec<Batch> = mbs.iter().map(|(x, y)| Batch::Lm { tokens: x, targets: y }).collect();
    let mut dp = DataParallel::<Qwen>::new(cfg.clone(), b, t, &init, &stage_gpus());
    assert_eq!(dp.n_replicas(), 2);
    dp.zero_grads();
    dp.forward_backward(&batches);

    let mut worst = 0f32;
    let mut worst_name = String::new();
    for (name, _) in cfg.param_list() {
        let a = single.read_grad(&name);
        let g = dp.reduced_grad(&name);
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
        assert!(rel < 1e-3, "dp grad mismatch for {name}: rel {rel:.2e}");
    }
    eprintln!("dp vs single-GPU accumulation: worst grad rel {worst:.2e} ({worst_name})");
}
