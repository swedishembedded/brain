// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Learnability tests (M8): the real hybrid GDN/GQA engine must actually
//! *learn* tiny tasks, not just pass the gradient check. Mirrors
//! `glmdsa/tests/convergence.rs`. Gated by `MOE_SKIP_GPU_TESTS` (these need a
//! working backend - CPU JIT or GPU).

use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn train_batch(m: &Qwen35, x: &[u32], y: &[u32], steps: u32, lr: f32) {
    m.set_batch(x, y);
    for step in 1..=steps {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, lr, 0.0, Some(1.0), 1.0);
        m.poll_wait();
    }
}

/// Overfitting a single fixed batch (full, all-parameter finetune) must drive
/// the loss well down - the end-to-end check that GDN + GQA + the dense MLP
/// all train together (forward *and* backward wired right).
#[test]
fn qwen35_full_finetune_overfits_fixed_batch() {
    if skip() {
        return;
    }
    let cfg = Qwen35Config::tiny();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 11);
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 2, t, &init);
    let x: Vec<u32> = (0..2 * t).map(|i| (i * 7) % cfg.vocab).collect();
    let y: Vec<u32> = (0..2 * t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    m.set_batch(&x, &y);
    let before = m.forward();
    train_batch(&m, &x, &y, 60, 1e-2);
    let after = m.forward();
    assert!(after < before * 0.3, "full finetune did not reduce loss enough: {before} -> {after}");
}

/// A short cyclic sequence is exactly memorisable; the model should reach a
/// low loss floor. Mirrors `glmdsa::glm_memorizes_cyclic_sequence`.
#[test]
fn qwen35_memorizes_cyclic_sequence() {
    if skip() {
        return;
    }
    let vocab = 7u32;
    let cfg = Qwen35Config { vocab, ..Qwen35Config::tiny() };
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 3);
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 2, t, &init);
    // Two overlapping windows over the cycle 0,1,2,...,6,0,1,... (predict next).
    let half = (t / 2) as usize;
    let cyc: Vec<u32> = (0..(t as usize + half + 1)).map(|i| (i as u32) % vocab).collect();
    let x: Vec<u32> = [&cyc[0..t as usize], &cyc[half..half + t as usize]].concat();
    let y: Vec<u32> = [&cyc[1..t as usize + 1], &cyc[half + 1..half + t as usize + 1]].concat();
    train_batch(&m, &x, &y, 300, 1e-2);
    let loss = m.forward();
    assert!(loss < 0.20, "cyclic memorization loss too high: {loss}");
}

/// LoRA (adapters only, base frozen) must also drive a fixed batch's loss
/// down - the added `.lora_a`/`.lora_b` machinery doesn't break optimisation.
#[test]
fn qwen35_lora_overfits_fixed_batch() {
    if skip() {
        return;
    }
    let cfg = Qwen35Config { lora: Some(lora_cfg(4, 8.0)), ..Qwen35Config::tiny() };
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 13);
    let m = Qwen35::new_train_on(Gpu::new(pipelines()), cfg.clone(), 2, t, &init);
    let x: Vec<u32> = (0..2 * t).map(|i| (i * 7) % cfg.vocab).collect();
    let y: Vec<u32> = (0..2 * t).map(|i| (i * 7 + 1) % cfg.vocab).collect();
    m.set_batch(&x, &y);
    let before = m.forward();
    train_batch(&m, &x, &y, 250, 3e-2);
    let after = m.forward();
    assert!(after < before * 0.8, "LoRA finetune did not learn: {before} -> {after}");
}
