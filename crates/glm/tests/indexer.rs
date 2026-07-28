// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DSA sparse-indexer (Phase 2) tests. The indexer is detached from the LM loss
//! (Frozen params, forward-only); these validate its *forward* behaviour:
//!   1. at `index_topk >= seq` the top-k mask is all-pass ⇒ output is bit-for-bit
//!      the dense-MLA (Phase-1) model — the invariant that keeps Phase 1 correct;
//!   2. at `index_topk < seq` the mask actually restricts attention (output
//!      differs, stays finite);
//!   3. IndexShare (`Full` then `Shared` layers) trains without breaking.
//! Gated by `MOE_SKIP_GPU_TESTS`.

use std::collections::HashMap;

use glm::{Glm, GlmConfig};

fn skip() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn batch() -> (Vec<u32>, Vec<u32>) {
    let x: Vec<u32> = (0..12).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 5 + 2) % 23).collect();
    (x, y)
}

/// Indexer at `index_topk >= seq` (all keys selected) is exactly the dense model:
/// the additive mask is 0 everywhere causal. Build an indexer model (layer 0
/// `Full`, layer 1 `Shared`) and a dense model sharing the same MLA/MoE weights;
/// their losses must match.
#[test]
fn indexer_allpass_equals_dense() {
    if skip() {
        return;
    }
    let cfg_idx = GlmConfig { indexer_full: vec![true, false], index_topk: 999, ..GlmConfig::tiny() };
    let init = glm::init_weights(&cfg_idx, 5);
    let a = Glm::new_on(gpu_core::testgpu::dev(glm::model::PIPELINES), cfg_idx, 2, 6, &init);
    let (x, y) = batch();
    a.set_batch(&x, &y);
    let la = a.forward();

    // Same dims, no indexer; reuse the shared weights (a subset of `init`).
    let cfg_dense = GlmConfig { indexer_full: vec![], ..GlmConfig::tiny() };
    let init_dense: HashMap<String, Vec<f32>> =
        cfg_dense.param_list().into_iter().map(|(n, _)| (n.clone(), init[&n].clone())).collect();
    let b = Glm::new_on(gpu_core::testgpu::dev(glm::model::PIPELINES), cfg_dense, 2, 6, &init_dense);
    b.set_batch(&x, &y);
    let lb = b.forward();

    assert!((la - lb).abs() < 1e-4, "indexer all-pass ({la}) != dense ({lb})");
}

/// Indexer at `index_topk < seq` restricts attention to the top-k keys per query,
/// so the output differs from the all-pass run (same weights) and stays finite.
#[test]
fn indexer_sparse_restricts_attention() {
    if skip() {
        return;
    }
    let cfg_sparse = GlmConfig { indexer_full: vec![true, false], index_topk: 2, ..GlmConfig::tiny() };
    let cfg_full = GlmConfig { index_topk: 999, ..cfg_sparse.clone() };
    let init = glm::init_weights(&cfg_sparse, 9); // param_list identical for both
    let (x, y) = batch();

    let ms = Glm::new_on(gpu_core::testgpu::dev(glm::model::PIPELINES), cfg_sparse, 2, 6, &init);
    ms.set_batch(&x, &y);
    let ls = ms.forward();
    let mf = Glm::new_on(gpu_core::testgpu::dev(glm::model::PIPELINES), cfg_full, 2, 6, &init);
    mf.set_batch(&x, &y);
    let lf = mf.forward();

    assert!(ls.is_finite() && ls > 0.0, "sparse loss not finite: {ls}");
    assert!((ls - lf).abs() > 1e-5, "top-k=2 mask did not change the output vs all-pass ({ls} vs {lf})");
}

/// DSA indexer distillation integration: `Glm::distill_step` runs a dense forward,
/// reads the cached attention + indexer inputs, computes the host-side gradient,
/// and updates the `idx.*` params (leaving the backbone untouched). This checks
/// the end-to-end wiring; the *math* is gradient-checked and the *convergence* is
/// validated on a controlled target in `glm::distill` unit tests.
#[test]
fn indexer_distillation_updates_weights() {
    if skip() {
        return;
    }
    let cfg = GlmConfig { indexer_full: vec![true, true], index_topk: 999, block_size: 16, ..GlmConfig::tiny() };
    let init = glm::init_weights(&cfg, 5);
    let model = Glm::new_on(gpu_core::testgpu::dev(glm::model::PIPELINES), cfg, 2, 8, &init);
    let x: Vec<u32> = (0..16).map(|i| (i * 5 + 1) % 23).collect();
    let y: Vec<u32> = (0..16).map(|i| (i * 5 + 2) % 23).collect();
    model.set_batch(&x, &y);
    // Train the backbone briefly so attention (the distillation target) is non-trivial.
    for step in 1..=40 {
        model.zero_grads();
        model.forward();
        model.backward();
        model.adamw_step(step, 1e-2, 0.0, Some(1.0), 1.0);
        model.poll_wait();
    }
    let loss = model.distill_step(0.0);
    assert!(loss.is_finite() && loss > 0.0, "distill loss not finite: {loss}");

    // A real step must move the indexer weights and leave the backbone unchanged.
    let idx0 = model.read_weight("blocks.0.idx.weights_proj.weight");
    let bb0 = model.read_weight("blocks.0.attn.q_a.weight");
    model.distill_step(0.1);
    model.poll_wait();
    let idx1 = model.read_weight("blocks.0.idx.weights_proj.weight");
    let bb1 = model.read_weight("blocks.0.attn.q_a.weight");
    assert!(idx0.iter().zip(&idx1).any(|(a, b)| (a - b).abs() > 1e-6), "distill_step did not update the indexer");
    assert_eq!(bb0, bb1, "distill_step must not touch the backbone");
}

/// An indexer-active model (IndexShare: `Full` + `Shared`) still trains — the
/// extra indexer dispatches don't break forward/backward.
#[test]
fn indexer_model_still_learns() {
    if skip() {
        return;
    }
    let cfg = GlmConfig { indexer_full: vec![true, false], index_topk: 999, block_size: 16, ..GlmConfig::tiny() };
    let init = glm::init_weights(&cfg, 11);
    let model = Glm::new_on(gpu_core::testgpu::dev(glm::model::PIPELINES), cfg, 2, 8, &init);
    let x: Vec<u32> = (0..16).map(|i| (i * 7 % 23) as u32).collect();
    let y: Vec<u32> = (0..16).map(|i| ((i * 7 + 1) % 23) as u32).collect();
    model.set_batch(&x, &y);
    let before = model.forward();
    for step in 1..=60 {
        model.zero_grads();
        model.forward();
        model.backward();
        model.adamw_step(step, 1e-2, 0.0, Some(1.0), 1.0);
        model.poll_wait();
    }
    let after = model.forward();
    assert!(after < before * 0.6, "indexer model did not learn: {before} -> {after}");
}
