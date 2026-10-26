// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of control adapters for diffusion models for its clients.
// If your team needs expertise in fine-tuning a ControlNet within a fixed device budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What ControlNet's training graph keeps of its activations. A graph that recomputes the trainable copy's blocks and
//! the frozen backbone's (`vae::blocks::Activations::Recompute`) gives the gradients of the one that keeps them, bit
//! for bit, holding fewer device bytes. One test in its own binary, so nothing else allocates while the device bytes
//! are compared.

use controlnet::config::ControlNetConfig;
use controlnet::train::{tensors_for, ControlNetTrainer, TRAIN_PIPELINES};
use sdxlunet::config::N_TIME_IDS;
use vae::blocks::Activations;

const H: u32 = 8;
const W: u32 = 8;
const T_ENC: u32 = 5;

fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + salt * 13) % 17) as f32 / 8.5) - 1.0).collect()
}

/// Every parameter gradient (bits) of one step, and the device bytes the trainer holds.
fn step(activations: Activations) -> (Vec<(String, Vec<u32>)>, u64) {
    let cfg = ControlNetConfig::tiny();
    let tensors = tensors_for(&cfg, 7);
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let m = ControlNetTrainer::new(gpu.share(), cfg.clone(), &tensors, H, W, T_ENC, activations);
    let held = gpu.memory_usage().unwrap().live - before;
    let bb = &cfg.backbone;
    let ds = cfg.cond_downscale();
    m.set_inputs(
        &values((bb.in_channels * H * W) as usize, 1),
        601.0,
        &values((T_ENC * bb.cross_attention_dim) as usize, 2),
        &values(bb.pooled_dim() as usize, 3),
        &values(N_TIME_IDS as usize, 4),
        &values((cfg.conditioning_channels * (H * ds) * (W * ds)) as usize, 5),
        0.7,
        &values((bb.out_channels * H * W) as usize, 6),
    );
    m.zero_grads();
    m.forward();
    m.backward();
    m.gpu().poll_wait();
    let grads = m.params().iter().map(|(n, _)| (n.clone(), m.read_grad(n).into_iter().map(f32::to_bits).collect())).collect();
    (grads, held)
}

#[test]
fn controlnet_recomputes_exactly() {
    let (keep, keep_bytes) = step(Activations::Keep);
    let (recompute, recompute_bytes) = step(Activations::Recompute);
    assert!(keep.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    for ((name, a), (_, b)) in keep.iter().zip(&recompute) {
        assert!(a == b, "{name}: gradient differs recomputing");
    }
    assert!(recompute_bytes < keep_bytes, "recomputing holds {recompute_bytes} device bytes, keeping {keep_bytes}");
    println!("controlnet device bytes: keep {keep_bytes}, recompute {recompute_bytes}");
}
