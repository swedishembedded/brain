// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM learnability tests: the real MLA-MoE engine must actually *learn* tiny
//! tasks, not just pass the gradient check. Gated by `MOE_SKIP_GPU_TESTS` (these
//! need a working backend — CPU JIT or GPU). Mirrors `gpt/tests/convergence.rs`.

use std::collections::HashMap;

use glmdsa::{Glm, GlmConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// A small config that trains quickly but still exercises MLA + a MoE layer.
fn small_cfg(vocab: u32, block: u32) -> GlmConfig {
    GlmConfig { vocab, block_size: block, ..GlmConfig::tiny() }
}

fn train_batch(model: &Glm, x: &[u32], y: &[u32], steps: u32, lr: f32) {
    model.set_batch(x, y);
    for step in 1..=steps {
        model.zero_grads();
        model.forward();
        model.backward();
        model.adamw_step(step, lr, 0.0, Some(1.0), 1.0);
        model.poll_wait();
    }
}

/// Overfitting a single fixed batch must drive the loss well down: this is the
/// end-to-end check that MLA + the sigmoid MoE router + shared expert + the
/// dense/MoE schedule all train together (forward *and* backward wired right).
#[test]
fn glm_overfits_fixed_batch() {
    if skip() {
        return;
    }
    let cfg = small_cfg(23, 16);
    let init = glmdsa::init_weights(&cfg, 11);
    let model = Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg, 2, 8, &init);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
    model.set_batch(&x, &y);
    let before = model.forward();
    train_batch(&model, &x, &y, 60, 1e-2);
    let after = model.forward();
    assert!(after < before * 0.5, "overfit did not reduce loss enough: {before} -> {after}");
}

/// A short cyclic sequence is exactly memorisable; the model should reach a low
/// loss floor. This mirrors `gpt`'s `engine_memorizes_cyclic_sequence`.
#[test]
fn glm_memorizes_cyclic_sequence() {
    if skip() {
        return;
    }
    let vocab = 7u32;
    let cfg = small_cfg(vocab, 16);
    let init = glmdsa::init_weights(&cfg, 3);
    let model = Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg, 2, 8, &init);
    // Two sequences over the cycle 0,1,2,3,4,5,6,0,1,... (predict next).
    let cyc: Vec<u32> = (0..17).map(|i| (i % vocab as usize) as u32).collect();
    let x: Vec<u32> = [&cyc[0..8], &cyc[4..12]].concat();
    let y: Vec<u32> = [&cyc[1..9], &cyc[5..13]].concat();
    train_batch(&model, &x, &y, 300, 1e-2);
    let loss = model.forward();
    assert!(loss < 0.20, "cyclic memorization loss too high: {loss}");
}

/// A GLM with the MTP head enabled trains (the added t+2 auxiliary loss and the
/// shared-head/embedding grad accumulation don't break optimisation).
#[test]
fn glm_mtp_overfits_fixed_batch() {
    if skip() {
        return;
    }
    let cfg = GlmConfig { mtp: true, ..small_cfg(23, 16) };
    let init = glmdsa::init_weights(&cfg, 13);
    let model = Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg, 2, 8, &init);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
    model.set_batch(&x, &y);
    let before = model.forward();
    train_batch(&model, &x, &y, 60, 1e-2);
    let after = model.forward();
    assert!(after < before * 0.6, "MTP model did not learn: {before} -> {after}");
}

/// More capacity should fit the same fixed batch at least as well — a basic
/// scaling-sanity signal that the architecture uses its parameters.
#[test]
fn glm_more_capacity_fits_better() {
    if skip() {
        return;
    }
    let x: Vec<u32> = (0..16).map(|i| (i * 5 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 5 + 3) % 23) as u32).collect();

    let run = |d_model: u32, moe_ff: u32| -> f32 {
        let cfg = GlmConfig { d_model, moe_intermediate_size: moe_ff, intermediate_size: moe_ff, ..small_cfg(23, 16) };
        let init: HashMap<_, _> = glmdsa::init_weights(&cfg, 5);
        let model = Glm::new_on(gpu_core::testgpu::dev(glmdsa::model::PIPELINES), cfg, 2, 8, &init);
        train_batch(&model, &x, &y, 120, 1e-2);
        model.forward()
    };
    let small = run(8, 8);
    let large = run(32, 32);
    assert!(large < small + 0.10, "larger model did not fit at least as well: small={small} large={large}");
}
