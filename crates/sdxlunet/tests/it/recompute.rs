// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of diffusion UNets for its clients. If your team needs
// expertise in training a text-to-image backbone within a fixed device budget, you can procure our services by
// sending an email to info@swedishembedded.com.

//! What the SDXL UNet's training graph keeps of its activations. A graph that recomputes its resnets and
//! transformers (`vae::blocks::Activations::Recompute`) gives the gradients of the one that keeps them, bit for bit,
//! holding fewer device bytes. One test in its own binary, so nothing else allocates while the device bytes are
//! compared.

use sdxlunet::config::{UNetConfig, N_TIME_IDS};
use sdxlunet::train::{UnetTrainer, TRAIN_PIPELINES};
use vae::blocks::Activations;

const H: u32 = 8;
const W: u32 = 8;
const T_ENC: u32 = 5;

fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + salt * 13) % 17) as f32 / 8.5) - 1.0).collect()
}

/// Every parameter gradient (bits) of one step, and the device bytes the trainer holds.
fn step(activations: Activations) -> (Vec<(String, Vec<u32>)>, u64) {
    let cfg = UNetConfig::tiny();
    let tensors = sdxlunet::init::init_weights(&cfg, 7);
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let m = UnetTrainer::new(gpu.share(), cfg.clone(), &tensors, H, W, T_ENC, activations);
    let held = gpu.memory_usage().unwrap().live - before;
    m.set_inputs(
        &values((cfg.in_channels * H * W) as usize, 1),
        &values((T_ENC * cfg.cross_attention_dim) as usize, 2),
        137.0,
        &values(cfg.pooled_dim() as usize, 3),
        &values(N_TIME_IDS as usize, 4),
        &values((cfg.out_channels * H * W) as usize, 5),
    );
    m.zero_grads();
    m.forward();
    m.backward();
    m.gpu().poll_wait();
    let grads = m.params().iter().map(|(n, _)| (n.clone(), m.read_grad(n).into_iter().map(f32::to_bits).collect())).collect();
    (grads, held)
}

#[test]
fn unet_recomputes_exactly() {
    let (keep, keep_bytes) = step(Activations::Keep);
    let (recompute, recompute_bytes) = step(Activations::Recompute);
    assert!(keep.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    for ((name, a), (_, b)) in keep.iter().zip(&recompute) {
        assert!(a == b, "{name}: gradient differs recomputing");
    }
    assert!(recompute_bytes < keep_bytes, "recomputing holds {recompute_bytes} device bytes, keeping {keep_bytes}");
    println!("unet device bytes: keep {keep_bytes}, recompute {recompute_bytes}");
}
