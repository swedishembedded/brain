// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generic `model::DataParallel` works for the **autoencoder** — a non-LM
//! model whose batch is `Batch::Tensor` (float inputs/targets, no tokens). This
//! proves multi-GPU data-parallel training is architecture-agnostic, not
//! token-model-specific: running `K` micro-batches across 2 replicas produces the
//! identical accumulated gradient as one GPU running the same `K`.
//!
//! `SHARD_TEST_GPUS=1,1` pins both replicas to one card. Skipped under
//! `MOE_SKIP_GPU_TESTS`.

use toyautoencoder::{Autoencoder, AutoencoderConfig};
use model::{Batch, DataParallel, Model};

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
fn dp_grad_parity_autoencoder() {
    if gpu_disabled() {
        return;
    }
    let cfg = AutoencoderConfig::tiny(); // in_dim 12, hidden 16, z_dim 4
    let init = toyautoencoder::init_weights(&cfg, 7);
    let (b, t) = (4u32, 1u32);
    let n = (b * cfg.in_dim) as usize;
    let k = 4usize;
    let mbs: Vec<(Vec<f32>, Vec<f32>)> = (0..k)
        .map(|j| {
            let x: Vec<f32> = (0..n).map(|i| ((i + j * 5) as f32 * 0.1).sin()).collect();
            (x.clone(), x) // reconstruction target = input
        })
        .collect();

    // Single-GPU reference: accumulate all K micro-batches.
    std::env::remove_var("BRAIN_OFFLOAD_ADAM");
    let single = <Autoencoder as Model>::new(cfg.clone(), b, t, &init);
    single.zero_grads();
    for (x, y) in &mbs {
        single.set_batch(x, y);
        single.forward();
        single.backward();
    }
    single.poll_wait();

    // Data-parallel over 2 GPUs.
    let batches: Vec<Batch> =
        mbs.iter().map(|(x, y)| Batch::Tensor { tokens: None, inputs: x, targets: y }).collect();
    let mut dp = DataParallel::<Autoencoder>::new(cfg.clone(), b, t, &init, &stage_gpus());
    assert_eq!(dp.n_replicas(), 2);
    dp.zero_grads();
    dp.forward_backward(&batches);

    let mut worst = 0f32;
    let mut worst_name = String::new();
    for name in single.param_names() {
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
        assert!(rel < 1e-3, "autoencoder dp grad mismatch for {name}: rel {rel:.2e}");
    }
    eprintln!("autoencoder dp vs single-GPU accumulation: worst grad rel {worst:.2e} ({worst_name})");
}
