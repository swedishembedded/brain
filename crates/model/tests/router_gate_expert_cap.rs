// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Regression test for `router_gate.wgsl`/`router_gate_train.wgsl`'s
//! array-free rewrite (see those files' header docs): correctness at
//! `n_experts` values that exceed the kernels' former hard-coded
//! `array<f32, 128>`/`array<bool, 128>` scratch -- the exact wall
//! Qwen3.5-35B-A3B's 256 experts hits, and the same failure shape
//! `router_bwd_expert_cap.rs` already proved for the backward kernel
//! (docs/lessons.md #35b).
//!
//! The host oracle is a real top-k selection (softmax, greedy argmax without
//! replacement, renormalise), not just "sums to something plausible" -- it
//! independently re-derives which experts get selected and compares
//! index-for-index against the GPU kernel's selected set, plus the exact
//! gate values.

use data::rng::Lcg;
use gpu_core::Gpu;

const PIPES: &[(&str, &str)] = &[("router_gate", kernels::ROUTER_GATE), ("router_gate_train", kernels::ROUTER_GATE_TRAIN)];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Host oracle: softmax -> greedy top-k (ties broken by first occurrence,
/// matching the kernel's `>` comparison and ascending scan order) ->
/// renormalise the selected set, zero the rest. Independent of the kernel's
/// internal pass structure -- this is "what top-k routing means", not a
/// transcription of the WGSL.
fn host_router_gate(n_rows: usize, n_experts: usize, top_k: usize, logits: &[f32]) -> Vec<f32> {
    let e = n_experts;
    let mut out = vec![0.0f32; n_rows * e];
    for t in 0..n_rows {
        let base = t * e;
        let mx = logits[base..base + e].iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = logits[base..base + e].iter().map(|&x| (x - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

        let mut selected = vec![false; e];
        let mut sel_sum = 0.0f32;
        for _ in 0..top_k {
            let mut best = 0usize;
            let mut best_v = -1.0f32;
            for k in 0..e {
                if !selected[k] && probs[k] > best_v {
                    best_v = probs[k];
                    best = k;
                }
            }
            selected[best] = true;
            sel_sum += best_v;
        }
        let inv = 1.0 / sel_sum.max(1e-9);
        for k in 0..e {
            out[base + k] = if selected[k] { probs[k] * inv } else { 0.0 };
        }
    }
    out
}

fn assert_matches_oracle(got: &[f32], want: &[f32], n_experts: usize) {
    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-4,
        "router_gate diverged from the host oracle at n_experts={n_experts}: max_abs={max_abs} \
         got[..4]={:?} want[..4]={:?}",
        &got[..4.min(got.len())],
        &want[..4.min(want.len())],
    );
}

fn assert_well_formed(got: &[f32], n_rows: usize, n_experts: usize, top_k: usize) {
    for t in 0..n_rows {
        let row = &got[t * n_experts..(t + 1) * n_experts];
        let nonzero = row.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero, top_k, "row {t}: expected exactly top_k={top_k} nonzero gates, got {nonzero}");
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {t}: selected gate weights must renormalise to 1.0, got {sum}");
    }
}

fn run_gate_case(n_experts: u32) {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = idx(&g, "router_gate");

    let n_rows: u32 = 5;
    let top_k = (n_experts / 4).max(1).min(32);
    let mut rng = Lcg::new(0xBEEF ^ n_experts as u64);
    let logits = rng.vec_scaled((n_rows * n_experts) as usize, 3.0);

    let logits_buf = g.storage_init("logits", &logits);
    let gate_buf = g.storage((n_rows * n_experts) as u64);
    g.submit(&[], &[g.step(kernel, &[&logits_buf, &gate_buf], &[n_rows, n_experts, top_k], n_rows)]);
    g.poll_wait();
    let got = g.read(&gate_buf, (n_rows * n_experts) as usize);

    let want = host_router_gate(n_rows as usize, n_experts as usize, top_k as usize, &logits);
    assert_matches_oracle(&got, &want, n_experts as usize);
    assert_well_formed(&got, n_rows as usize, n_experts as usize, top_k as usize);
}

fn run_train_case(n_experts: u32) {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = idx(&g, "router_gate_train");

    let n_rows: u32 = 4;
    let top_k = (n_experts / 4).max(1).min(32);
    let mut rng = Lcg::new(0xCAFE ^ n_experts as u64);
    let logits = rng.vec_scaled((n_rows * n_experts) as usize, 3.0);

    let logits_buf = g.storage_init("logits", &logits);
    let gate_buf = g.storage((n_rows * n_experts) as u64);
    let probs_buf = g.storage((n_rows * n_experts) as u64);
    g.submit(&[], &[g.step(kernel, &[&logits_buf, &gate_buf, &probs_buf], &[n_rows, n_experts, top_k], n_rows)]);
    g.poll_wait();
    let got_gate = g.read(&gate_buf, (n_rows * n_experts) as usize);
    let got_probs = g.read(&probs_buf, (n_rows * n_experts) as usize);

    let want_gate = host_router_gate(n_rows as usize, n_experts as usize, top_k as usize, &logits);
    assert_matches_oracle(&got_gate, &want_gate, n_experts as usize);
    assert_well_formed(&got_gate, n_rows as usize, n_experts as usize, top_k as usize);

    // probs must be the full (un-masked) softmax, summing to 1 per row.
    for t in 0..n_rows as usize {
        let row = &got_probs[t * n_experts as usize..(t + 1) * n_experts as usize];
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "row {t}: full softmax probs must sum to 1.0, got {sum}");
        assert!(row.iter().all(|&v| v >= 0.0), "row {t}: softmax probs must be non-negative");
    }
}

/// Below the former cap -- must keep working exactly as before the rewrite.
#[test]
fn router_gate_matches_host_oracle_at_8_experts() {
    run_gate_case(8);
}

/// One past the former `array<f32, 128>` cap -- the exact boundary #35b
/// corrupts silently on the pre-rewrite kernel.
#[test]
fn router_gate_matches_host_oracle_at_129_experts() {
    run_gate_case(129);
}

/// Qwen3.5-35B-A3B's real scale (256 experts, top-8).
#[test]
fn router_gate_matches_host_oracle_at_256_experts() {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = idx(&g, "router_gate");
    let (n_rows, n_experts, top_k) = (3u32, 256u32, 8u32);
    let mut rng = Lcg::new(0x51DE_1234);
    let logits = rng.vec_scaled((n_rows * n_experts) as usize, 3.0);

    let logits_buf = g.storage_init("logits", &logits);
    let gate_buf = g.storage((n_rows * n_experts) as u64);
    g.submit(&[], &[g.step(kernel, &[&logits_buf, &gate_buf], &[n_rows, n_experts, top_k], n_rows)]);
    g.poll_wait();
    let got = g.read(&gate_buf, (n_rows * n_experts) as usize);

    let want = host_router_gate(n_rows as usize, n_experts as usize, top_k as usize, &logits);
    assert_matches_oracle(&got, &want, n_experts as usize);
    assert_well_formed(&got, n_rows as usize, n_experts as usize, top_k as usize);
}

#[test]
fn router_gate_train_matches_host_oracle_at_8_experts() {
    run_train_case(8);
}

#[test]
fn router_gate_train_matches_host_oracle_at_129_experts() {
    run_train_case(129);
}

#[test]
fn router_gate_train_matches_host_oracle_at_256_experts() {
    run_train_case(256);
}
