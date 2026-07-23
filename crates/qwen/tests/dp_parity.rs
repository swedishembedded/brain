// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Data-parallel training is bit-exact against single-GPU grad accumulation.
//!
//! Running `K` micro-batches split across 2 replicas + all-reduce must produce
//! the identical accumulated gradient as one GPU running the same `K`
//! micro-batches with grad-accum `K`. (This also confirms the backward pass
//! accumulates into the grad buffers, which grad-accum depends on.)
//!
//! Two replicas on GPUs 0 and 1 by default; `SHARD_TEST_GPUS=1,1` pins both to
//! one card. Skipped under `MOE_SKIP_GPU_TESTS`.

use qwen::{DataParallel, Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn stage_gpus() -> Vec<usize> {
    std::env::var("SHARD_TEST_GPUS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![0, 1])
}

#[test]
fn dp_grad_parity() {
    if gpu_disabled() {
        return;
    }
    let cfg = QwenConfig::tiny();
    let init = qwen::init_weights(&cfg, 7);
    let (b, t) = (2u32, 8u32);

    // K micro-batches (distinct data each).
    let k = 4usize;
    let mbs: Vec<(Vec<u32>, Vec<u32>)> = (0..k)
        .map(|j| {
            let x = (0..b * t).map(|i| ((i * 3 + j as u32 * 13) % cfg.vocab) as u32).collect();
            let y = (0..b * t).map(|i| ((i * 3 + 1 + j as u32 * 13) % cfg.vocab) as u32).collect();
            (x, y)
        })
        .collect();

    // Single-GPU reference: accumulate all K micro-batches.
    std::env::remove_var("BRAIN_OFFLOAD_ADAM");
    let single = Qwen::new(cfg.clone(), b, t, &init);
    single.zero_grads();
    for (x, y) in &mbs {
        single.set_batch(x, y);
        single.forward();
        single.backward();
    }
    single.poll_wait();

    // Data-parallel over 2 GPUs: same K micro-batches, then all-reduce.
    let mut dp = DataParallel::new(cfg.clone(), b, t, &init, &stage_gpus());
    assert_eq!(dp.n_replicas(), 2);
    dp.zero_grads();
    dp.forward_backward(&mbs);
    dp.all_reduce();

    let mut worst = 0f32;
    let mut worst_name = String::new();
    for (name, _) in cfg.param_list() {
        let a = single.read_grad(&name);
        let g = dp.read_grad(&name);
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
