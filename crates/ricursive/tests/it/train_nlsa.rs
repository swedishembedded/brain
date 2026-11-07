// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native sparse attention (N-LSA) on Ripple's global layers, trained through the shared kernels.
//!
//! Swedish Embedded AB implements long-context models trained with sparse attention for clients whose sequences
//! outgrow quadratic attention. If your team needs expertise in continuing the training of a dense model with
//! sparse attention then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The kernels need workgroup barriers, which the CPU JIT does not run: these tests skip by name there and run on
//! a GPU or the software Vulkan adapter (`BRAIN_DEVICE=software`).

use model::lsa::LsaConfig;
use model::{Adam, Batch, Model};
use ricursive::{Ripple, TrainConfig};

const ROWS: u32 = 2;
const SEQ: u32 = 64;

fn runs_here() -> bool {
    let runs = gpu_core::testgpu::dev(&[]).caps().workgroup_reductions;
    if !runs {
        brain_testutil::skip_unavailable("N-LSA: this backend does not run workgroup-barrier kernels (GPU-only)");
    }
    runs
}

/// The tiny Ripple with rows long enough to outgrow a budget: its global layer is the last of three.
fn config() -> TrainConfig {
    TrainConfig { block_size: SEQ, ..TrainConfig::tiny() }
}

fn lsa(top_k: usize, top_p: usize, window: usize) -> LsaConfig {
    LsaConfig { index_dim: 0, pool_size: 2, bucket_size: 4, bucket_cap: 8, cluster_block: 8, top_p, top_k, sink: 2, window, early: None }
}

fn batch(vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let n = ROWS * SEQ;
    let x = (0..n).map(|i| (i * 5 + i / 7 + 1) % vocab).collect();
    let y = (0..n).map(|i| (i * 5 + i / 7 + 6) % vocab).collect();
    (x, y)
}

fn step(m: &Ripple, x: &[u32], y: &[u32]) -> (f32, Vec<f32>) {
    m.set_batch(Batch::Lm { tokens: x, targets: y });
    m.zero_grads();
    let loss = m.forward();
    m.backward();
    m.poll_wait();
    (loss, m.param_names().iter().flat_map(|n| m.read_grad(n)).collect())
}

fn model(native: Option<(LsaConfig, bool)>) -> Ripple {
    let cfg = config();
    let weights = cfg.tiny_weights(3);
    let mut m = Ripple::new(cfg, ROWS, SEQ, &weights);
    if let Some((lsa, long_range)) = native {
        m.enable_native_lsa(lsa, long_range).expect("native sparse attention is enabled");
    }
    m
}

/// Spec: with a budget past the row every query selects every causal key, so the sparse step has dense
/// attention's loss and every parameter's gradient; a window-only selection does not.
#[test]
fn a_full_budget_trains_like_dense_attention() {
    if !runs_here() {
        return;
    }
    let (x, y) = batch(config().model.vocab_size as u32);
    let (dense_loss, dense) = step(&model(None), &x, &y);
    let (full_loss, full) = step(&model(Some((lsa(1024, 1024, 16), true))), &x, &y);
    let (window_loss, _) = step(&model(Some((lsa(1024, 1024, 16), false))), &x, &y);

    let scale = dense.iter().fold(0f32, |m, g| m.max(g.abs()));
    let worst = dense.iter().zip(&full).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
    eprintln!("loss dense {dense_loss:.6}, full budget {full_loss:.6}, window only {window_loss:.6}; worst gradient error {worst:.2e} of {scale:.2e}");
    assert!((dense_loss - full_loss).abs() < 1e-5 * dense_loss.abs(), "loss {full_loss} vs dense {dense_loss}");
    assert!(worst < 1e-4, "gradients differ by {worst:.2e} of the largest");
    assert!((dense_loss - window_loss).abs() > 1e-4, "a window-only selection changed nothing: the selection is not applied");
}

/// Spec: a sparse budget (each query reads a fraction of its causal keys) trains: AdamW on a fixed batch brings the
/// loss down.
#[test]
fn a_sparse_budget_trains() {
    if !runs_here() {
        return;
    }
    let (x, y) = batch(config().model.vocab_size as u32);
    let m = model(Some((lsa(16, 8, 8), true)));
    let (first, _) = step(&m, &x, &y);
    let mut last = first;
    for t in 1..=40 {
        m.set_batch(Batch::Lm { tokens: &x, targets: &y });
        m.zero_grads();
        last = m.forward();
        m.backward();
        m.adamw_step(t, 1e-2, 0.0, Adam::default(), Some(1.0), 1.0);
    }
    m.poll_wait();
    eprintln!("sparse budget loss {first:.4} -> {last:.4}");
    assert!(last < 0.5 * first, "the loss {first} -> {last} did not come down");
}

/// Spec: the sparse step really reads fewer keys than the dense step on the same weights (the budget bites).
#[test]
fn a_sparse_budget_changes_the_loss_from_dense() {
    if !runs_here() {
        return;
    }
    let (x, y) = batch(config().model.vocab_size as u32);
    let (dense, _) = step(&model(None), &x, &y);
    let (sparse, _) = step(&model(Some((lsa(16, 8, 8), true))), &x, &y);
    assert!((dense - sparse).abs() > 1e-4, "dense {dense}, sparse {sparse}");
}
