// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of image restoration models for its clients. If your
// team needs expertise in training a SUPIR-style restorer within a fixed device budget, you can procure our services
// by sending an email to info@swedishembedded.com.

//! What SUPIR's graphs keep of their activations. Its trunk, backbone and adaptor blocks free their intermediates
//! after their last read, so the inference graph reuses them and computes the training graph's output; and a training
//! graph that recomputes those blocks (`vae::blocks::Activations::Recompute`) gives the gradients of the one that
//! keeps them, bit for bit, holding fewer device bytes. One test in its own binary, so nothing else allocates while
//! the device bytes are compared.

use sdxlunet::config::N_TIME_IDS;
use supir::config::SupirConfig;
use supir::model::{GraphShape, Supir, KERNELS};
use supir::train::{SupirTrainer, TRAIN_PIPELINES};
use vae::blocks::{Activations, Tensors};

const SHAPE: GraphShape = GraphShape { h: 8, w: 8, t_enc: 5, control_scale: 0.7 };
const TIMESTEP: f32 = 601.0;

fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7 + salt * 13) % 17) as f32 / 8.5) - 1.0).collect()
}

fn weights(cfg: &SupirConfig) -> Tensors {
    let mut tensors = sdxlunet::init::init_weights(&cfg.backbone, 7);
    tensors.extend(supir::init::init_weights(cfg, 7 ^ 0x5350_4952));
    tensors
}

struct Batch {
    sample: Vec<f32>,
    hint: Vec<f32>,
    enc: Vec<f32>,
    pooled: Vec<f32>,
    time_ids: Vec<f32>,
    target: Vec<f32>,
}

fn batch(cfg: &SupirConfig) -> Batch {
    let c = &cfg.backbone;
    let GraphShape { h, w, t_enc, .. } = SHAPE;
    Batch {
        sample: values((c.in_channels * h * w) as usize, 1),
        hint: values((4 * h * w) as usize, 2),
        enc: values((t_enc * c.cross_attention_dim) as usize, 3),
        pooled: values(c.pooled_dim() as usize, 4),
        time_ids: values(N_TIME_IDS as usize, 5),
        target: values((c.out_channels * h * w) as usize, 6),
    }
}

/// Every parameter gradient (bits) of one step, the graph's output, and the device bytes the trainer holds.
type Measured = (Vec<(String, Vec<u32>)>, Vec<f32>, u64);

fn step(activations: Activations) -> Measured {
    let cfg = SupirConfig::tiny();
    let gpu = gpu_core::testgpu::dev(TRAIN_PIPELINES);
    let before = gpu.memory_usage().expect("the test device reports its memory").live;
    let m = SupirTrainer::new(gpu.share(), cfg.clone(), &weights(&cfg), SHAPE, activations);
    let held = gpu.memory_usage().unwrap().live - before;
    let b = batch(&cfg);
    m.set_inputs(&b.sample, &b.hint, &b.enc, TIMESTEP, &b.pooled, &b.time_ids, &b.target);
    m.zero_grads();
    m.forward();
    m.backward();
    m.gpu().poll_wait();
    let grads = m.params().iter().map(|(n, _)| (n.clone(), m.read_grad(n).into_iter().map(f32::to_bits).collect())).collect();
    let out = m.gpu().read(m.out(), b.target.len());
    (grads, out, held)
}

#[test]
fn supir_graphs_reuse_and_recompute_exactly() {
    let (keep, out, keep_bytes) = step(Activations::Keep);
    let (recompute, _, recompute_bytes) = step(Activations::Recompute);
    assert!(keep.iter().any(|(_, g)| g.iter().any(|b| f32::from_bits(*b) != 0.0)), "no parameter received a gradient");
    for ((name, a), (_, b)) in keep.iter().zip(&recompute) {
        assert!(a == b, "{name}: gradient differs recomputing");
    }
    assert!(recompute_bytes < keep_bytes, "recomputing holds {recompute_bytes} device bytes, keeping {keep_bytes}");
    println!("supir device bytes: keep {keep_bytes}, recompute {recompute_bytes}");

    // The inference graph reuses freed intermediates (taps pin every buffer, so a tapped build reuses none): the
    // two compute the same bits. The training graph differs from both only by pinning the materialised attention
    // where inference runs the flash kernel, a rounding-level difference.
    let cfg = SupirConfig::tiny();
    let GraphShape { h, w, t_enc, control_scale } = SHAPE;
    let b = batch(&cfg);
    let infer = |taps: bool| {
        let m = Supir::new(gpu_core::testgpu::dev(&KERNELS), cfg.clone(), &weights(&cfg), h, w, t_enc, taps, control_scale);
        m.run(&b.sample, &b.hint, TIMESTEP, &b.enc, &b.pooled, &b.time_ids)
    };
    let (reusing, pinned) = (infer(false), infer(true));
    assert!(reusing.iter().zip(&pinned).all(|(a, b)| a.to_bits() == b.to_bits()), "buffer reuse changes the inference output");
    let drift = reusing.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(drift < 1e-5, "the inference graph's output differs from the training forward by {drift}");
}
