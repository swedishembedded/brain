// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Regression test for `router_gate_sigmoid.wgsl`'s array-free rewrite (see
//! that file's header doc, and lessons.md #35c): correctness
//! at `n_experts` values that exceed the kernel's former hard-coded
//! `array<f32/bool, 64>` scratch (`s`/`choice`/`used`) - the exact bound
//! `router_bwd_expert_cap.rs`/`router_gate_expert_cap.rs` already gate for
//! their own sibling kernels, and #35c's own writeup named this file's
//! kernel as the one instance the earlier pass deliberately left behind
//! its `assert!` rather than bump.
//!
//! The host oracle mirrors the kernel's own structure (sigmoid -> optional
//! group-limited top-2 group scoring -> masked global top-k -> gate), so
//! this also catches a real numerical regression in the rewrite, not just
//! an out-of-bounds write.

use data::rng::Lcg;
use gpu_core::Gpu;

const PIPES: &[(&str, &str)] = &[("router_gate_sigmoid", kernels::ROUTER_GATE_SIGMOID)];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// Host oracle: the same structure as `router_gate_sigmoid.wgsl`'s `main`.
#[allow(clippy::too_many_arguments)]
fn host_router_gate_sigmoid(
    n_rows: usize,
    e: usize,
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    norm: bool,
    scale: f32,
    logits: &[f32],
    bias: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut gate = vec![0.0f32; n_rows * e];
    let mut probs = vec![0.0f32; n_rows * e];
    for t in 0..n_rows {
        let base = t * e;
        let s: Vec<f32> = (0..e).map(|i| 1.0 / (1.0 + (-logits[base + i]).exp())).collect();
        probs[base..base + e].copy_from_slice(&s);

        let ng = n_group.max(1);
        let per = e / ng;
        let mut group_keep = vec![ng == 1; ng];
        if ng > 1 {
            let mut gscore = vec![0.0f32; ng];
            for g in 0..ng {
                let (mut b1, mut b2) = (f32::MIN, f32::MIN);
                for m in 0..per {
                    let cv = s[g * per + m] + bias[g * per + m];
                    if cv > b1 {
                        b2 = b1;
                        b1 = cv;
                    } else if cv > b2 {
                        b2 = cv;
                    }
                }
                gscore[g] = b1 + b2;
            }
            let mut gused = vec![false; ng];
            for _ in 0..topk_group {
                let (mut best, mut bestv) = (0usize, f32::MIN);
                for g in 0..ng {
                    if !gused[g] && gscore[g] > bestv {
                        bestv = gscore[g];
                        best = g;
                    }
                }
                gused[best] = true;
                group_keep[best] = true;
            }
        }

        let mut sel: Vec<usize> = Vec::new();
        let mut sel_sum = 0.0f32;
        for _ in 0..top_k {
            let (mut best, mut bestv) = (0usize, f32::MIN);
            for ee in 0..e {
                if !group_keep[ee / per] || sel.contains(&ee) {
                    continue;
                }
                let cv = s[ee] + bias[ee];
                if cv > bestv {
                    bestv = cv;
                    best = ee;
                }
            }
            sel.push(best);
            sel_sum += s[best];
        }
        let denom = if norm { 1.0 / sel_sum.max(1e-20) } else { 1.0 };
        for ee in 0..e {
            gate[base + ee] = if sel.contains(&ee) { s[ee] * denom * scale } else { 0.0 };
        }
    }
    (gate, probs)
}

fn run_case(n_experts: u32, n_group: u32, topk_group: u32) {
    let g = gpu_core::testgpu::dev(PIPES);
    let kernel = idx(&g, "router_gate_sigmoid");

    let n_rows: u32 = 5;
    let top_k = (n_experts / 8).max(1);
    let mut rng = Lcg::new(0xC0FFEE ^ (n_experts as u64) << 8 ^ n_group as u64);

    let logits = rng.vec_scaled((n_rows * n_experts) as usize, 3.0);
    let bias = rng.vec_scaled(n_experts as usize, 1.0);
    let (norm, scale) = (true, 2.5f32);

    let logits_buf = g.storage_init("logits", &logits);
    let bias_buf = g.storage_init("bias", &bias);
    let gate_buf = g.storage((n_rows * n_experts) as u64);
    let probs_buf = g.storage((n_rows * n_experts) as u64);

    let params = [n_rows, n_experts, top_k, n_group, topk_group, norm as u32, gpu_core::f(scale)];
    g.submit(
        &[],
        &[g.step(kernel, &[&logits_buf, &bias_buf, &gate_buf, &probs_buf], &params, n_rows)],
    );
    g.poll_wait();
    let got_gate = g.read(&gate_buf, (n_rows * n_experts) as usize);
    let got_probs = g.read(&probs_buf, (n_rows * n_experts) as usize);

    let (want_gate, want_probs) = host_router_gate_sigmoid(
        n_rows as usize,
        n_experts as usize,
        top_k as usize,
        n_group as usize,
        topk_group as usize,
        norm,
        scale,
        &logits,
        &bias,
    );

    let mut max_abs = 0.0f32;
    for (a, b) in got_gate.iter().zip(want_gate.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    for (a, b) in got_probs.iter().zip(want_probs.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-4,
        "router_gate_sigmoid diverged from the host oracle at n_experts={n_experts}, n_group={n_group}: \
         max_abs={max_abs} got_gate[..4]={:?} want_gate[..4]={:?}",
        &got_gate[..4.min(got_gate.len())],
        &want_gate[..4.min(want_gate.len())],
    );

    // Exactly `top_k` nonzero gates per row - a vacuous all-zero (or
    // all-selected) result would pass a `max_abs` check trivially if the
    // oracle had the same bug, so check the SHAPE of the answer too.
    for t in 0..n_rows as usize {
        let nz = (0..n_experts as usize).filter(|&i| got_gate[t * n_experts as usize + i] != 0.0).count();
        assert_eq!(nz, top_k as usize, "row {t} at n_experts={n_experts}: expected exactly {top_k} nonzero gates");
    }
}

/// Below the former cap -- must keep working exactly as before the rewrite.
#[test]
fn router_gate_sigmoid_matches_host_oracle_at_8_experts() {
    run_case(8, 1, 1);
}

/// One past the former `array<f32/bool, 64>` cap -- the exact boundary
/// #35c's own writeup names as the one left unfixed.
#[test]
fn router_gate_sigmoid_matches_host_oracle_at_65_experts() {
    run_case(65, 1, 1);
}

/// `GlmConfig::glm5_2()`'s real scale (256 routed experts) -- the config
/// this fix unblocks. `n_group=4`/`topk_group=2` also exercises the
/// group-limited masking path (glm5_2 itself ships `n_group=1`, but the
/// masking arrays are bounded by `n_group`, a genuinely different quantity
/// from `n_experts`, and need their own coverage above 1).
#[test]
fn router_gate_sigmoid_matches_host_oracle_at_256_experts_grouped() {
    run_case(256, 4, 2);
}
