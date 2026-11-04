// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of image restoration networks for its clients. If your team
// needs expertise in fine-tuning a super-resolution model within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! What RRDBNet's graphs keep of their activations. Every dense block frees its intermediates after their last read,
//! so the inference graph reuses them and computes exactly the training graph's output; and a training graph that
//! recomputes the dense blocks (`vae::blocks::Activations::Recompute`) gives the gradients of the one that keeps them,
//! bit for bit, holding fewer device bytes. One test in its own binary, so nothing else allocates while the device
//! bytes are compared.

use rrdbnet::train::{RrdbTrainer, TRAIN_PIPELINES};
use rrdbnet::{Rrdb, RrdbConfig};
use vae::blocks::{Activations, Tensors};

const H: u32 = 4;
const W: u32 = 6;

fn config() -> RrdbConfig {
    RrdbConfig { in_channels: 3, out_channels: 3, num_feat: 8, num_grow_ch: 4, num_block: 2, scale: 4 }
}

fn weights(cfg: &RrdbConfig) -> Tensors {
    cfg.param_list()
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let n: usize = shape.iter().product();
            let s = 1.0 / ((n / shape[0]).max(1) as f32).sqrt();
            let data = (0..n).map(|j| (((j * 7 + i * 13) % 17) as f32 / 8.5 - 1.0) * s).collect();
            (name, (shape, data))
        })
        .collect()
}

fn inputs(cfg: &RrdbConfig) -> (Vec<f32>, Vec<f32>) {
    let n_in = (cfg.in_channels * H * W) as usize;
    let n_out = n_in * (cfg.scale * cfg.scale) as usize;
    ((0..n_in).map(|i| (i % 9) as f32 / 9.0).collect(), (0..n_out).map(|i| (i % 7) as f32 / 7.0).collect())
}

/// Every parameter gradient (bits) of one step, the unclamped output, and the device bytes the trainer holds.
type Measured = (Vec<(String, Vec<u32>)>, Vec<f32>, u64);

fn step(activations: Activations) -> Measured {
    let cfg = config();
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let m = RrdbTrainer::new(gpu.share(), cfg.clone(), &weights(&cfg), H, W, activations);
    let held = gpu.memory_usage().unwrap().live - before;
    let (image, target) = inputs(&cfg);
    m.set_inputs(&image, &target);
    m.zero_grads();
    m.forward();
    m.backward();
    m.gpu().poll_wait();
    let grads = m.params().iter().map(|(n, _)| (n.clone(), m.read_grad(n).into_iter().map(f32::to_bits).collect())).collect();
    let (oh, ow) = m.out_hw();
    let out = m.gpu().read(m.out(), (cfg.out_channels * oh * ow) as usize);
    (grads, out, held)
}

#[test]
fn rrdbnet_graphs_reuse_and_recompute_exactly() {
    let (keep, out, keep_bytes) = step(Activations::Keep);
    let (recompute, _, recompute_bytes) = step(Activations::Recompute);
    assert!(keep.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    for ((name, a), (_, b)) in keep.iter().zip(&recompute) {
        assert!(a == b, "{name}: gradient differs recomputing");
    }
    assert!(recompute_bytes < keep_bytes, "recomputing holds {recompute_bytes} device bytes, keeping {keep_bytes}");

    // The inference graph reuses freed intermediates; its (clamped) output is the training graph's.
    let cfg = config();
    let eval = Rrdb::new(gpu_core::testgpu::dev(&rrdbnet::KERNELS), cfg.clone(), &weights(&cfg), H, W, false);
    let got = eval.run(&inputs(&cfg).0);
    let want: Vec<f32> = out.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "the inference graph's output differs from the training forward");
}
