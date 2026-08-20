// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Convergence + load-bearing checks for the MTP head. No reference
//! oracle exists for `mtp.*` on this box (`transformers` discards it on
//! load), so these two checks are the head's end-to-end correctness gate,
//! complementing
//! `gradcheck::check_qwen35_mtp`'s per-tensor finite-difference check:
//! (1) a real training loop with `cfg.mtp` on must still reduce the combined
//! loss (the added aux loss and the shared-head/embedding gradient
//! accumulation don't break optimisation); (2) zeroing `mtp.fc_e`/`mtp.fc_h`
//! (collapsing the one extra decoder layer's input to a constant zero for
//! every token) must move the loss, proving the head actually participates
//! in the forward computation rather than being wired but inert.

use gpu_core::Gpu;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

fn mtp_cfg() -> Qwen35Config {
    Qwen35Config { mtp: true, ..Qwen35Config::tiny() }
}

fn batch(cfg: &Qwen35Config, t: u32) -> (Vec<u32>, Vec<u32>) {
    let x: Vec<u32> = (0..t).map(|i| (i * 3 + 1) % cfg.vocab).collect();
    let y: Vec<u32> = (0..t).map(|i| (i * 3 + 2) % cfg.vocab).collect();
    (x, y)
}

fn run_overfit(gpu: Gpu) {
    let cfg = mtp_cfg();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 11);
    let m = Qwen35::new_train_on(gpu, cfg.clone(), 1, t, &init);
    let (x, y) = batch(&cfg, t);
    m.set_batch(&x, &y);

    let before = m.forward();
    for step in 1..=60 {
        m.zero_grads();
        m.forward();
        m.backward();
        m.adamw_step(step, 1e-2, 0.0, Some(1.0), 1.0);
        m.poll_wait();
    }
    let after = m.forward();
    assert!(after < before * 0.6, "MTP model did not learn: {before} -> {after}");
}

#[test]
fn qwen35_mtp_overfits_fixed_batch_cpu() {
    run_overfit(Gpu::new_cpu(pipelines()));
}

#[test]
fn qwen35_mtp_overfits_fixed_batch_default_backend() {
    run_overfit(Gpu::new(pipelines()));
}

/// `Qwen35::new_on` (frozen, no training) is enough here - a plain forward
/// pass comparison before/after mutating two weights, no gradient needed.
fn run_mutation_check(gpu: Gpu) {
    let cfg = mtp_cfg();
    let t = cfg.block_size;
    let init = qwen35::init::init_weights(&cfg, 13);
    let m = Qwen35::new_on(gpu, cfg.clone(), 1, t, &init);
    let (x, y) = batch(&cfg, t);
    m.set_batch(&x, &y);

    let before = m.forward();

    // The main decoder branch never reads `mtp.*` (it's a strict side branch
    // off `res[last]`), so any change below is entirely the MTP head's own
    // contribution to the combined loss.
    for name in ["mtp.fc_e.weight", "mtp.fc_h.weight"] {
        let zeros = vec![0.0f32; m.read_weight(name).len()];
        m.write_weight(name, &zeros);
    }
    let after = m.forward();

    assert_ne!(before, after, "zeroing mtp.fc_e/mtp.fc_h did not change the loss -- the MTP head is wired but inert");
}

#[test]
fn qwen35_mtp_head_is_load_bearing_cpu() {
    run_mutation_check(Gpu::new_cpu(pipelines()));
}

#[test]
fn qwen35_mtp_head_is_load_bearing_default_backend() {
    run_mutation_check(Gpu::new(pipelines()));
}
