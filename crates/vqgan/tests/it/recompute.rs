// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of image tokenizers for its clients. If your team needs
// expertise in fine-tuning a VQGAN within a fixed device budget, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A VQGAN trainer that recomputes its encoder's and generator's blocks (`vae::blocks::Activations::Recompute`) gives
//! the gradients of the one that keeps them, bit for bit, holding fewer device bytes. One test in its own binary, so
//! nothing else allocates while the device bytes are compared.

use vae::blocks::{Activations, Tensors};
use vqgan::train::VqganTrainer;
use vqgan::{BetaTerm, VqganConfig};

fn config() -> VqganConfig {
    VqganConfig {
        in_channels: 3,
        out_channels: 3,
        nf: 4,
        ch_mult: vec![1, 2],
        res_blocks: 1,
        dec_res_blocks: 1,
        attn_resolutions: vec![4],
        img_size: 8,
        codebook_size: 6,
        emb_dim: 4,
        z_channels: None,
        head_act: false,
        codebook_l2_norm: false,
        beta: 0.25,
        beta_on: BetaTerm::Codebook,
        norm_groups: 2,
        norm_eps: 1e-6,
    }
}

fn fill(n: usize, k: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| scale * (((i + 7 * k) as f32) * 0.754_877_7).sin()).collect()
}

fn weights(cfg: &VqganConfig) -> Tensors {
    cfg.tensor_manifest()
        .into_iter()
        .enumerate()
        .map(|(k, (name, shape))| {
            let n: usize = shape.iter().product();
            let data = match shape.len() {
                1 if name.ends_with(".weight") => fill(n, k, 0.1).iter().map(|v| 1.0 + v).collect(),
                1 => fill(n, k, 0.1),
                2 => fill(n, k, 0.6),
                _ => fill(n, k, 1.0 / ((n / shape[0]) as f32).sqrt()),
            };
            (name, (shape, data))
        })
        .collect()
}

/// Every parameter gradient (bits) of one step and the device bytes the trainer holds.
type Measured = (Vec<(String, Vec<u32>)>, u64);

fn step(activations: Activations) -> Measured {
    let cfg = config();
    let gpu = gpu_core::testgpu::dev(vqgan::train::TRAIN_PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let s = cfg.img_size;
    let m = VqganTrainer::new(cfg.clone(), &weights(&cfg), s, s, gpu.share(), activations);
    let held = gpu.memory_usage().unwrap().live - before;
    let n = (cfg.in_channels * s * s) as usize;
    m.set_batch(&fill(n, 1, 1.0), &fill(n, 2, 1.0));
    m.zero_grads();
    m.loss();
    m.backward();
    m.gpu().poll_wait();
    let grads = m.param_names().iter().map(|(name, _)| (name.clone(), m.read_grad(name).into_iter().map(f32::to_bits).collect())).collect();
    (grads, held)
}

#[test]
fn vqgan_recomputes_its_blocks_exactly() {
    let (keep, keep_bytes) = step(Activations::Keep);
    let (recompute, recompute_bytes) = step(Activations::Recompute);
    assert!(keep.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    assert_eq!(keep.len(), recompute.len());
    for ((name, a), (_, b)) in keep.iter().zip(&recompute) {
        assert!(a == b, "{name}: gradient differs recomputing");
    }
    assert!(recompute_bytes < keep_bytes, "recomputing holds {recompute_bytes} device bytes, keeping {keep_bytes}");
}
