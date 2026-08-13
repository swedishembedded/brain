// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chunked-training equivalence: the long-context training regime (query-
//! chunked attention fwd + per-chunk-recompute bwd via the accumulating
//! dk/dv kernels, gathered supervised-row MLM head) must reproduce the
//! materialized regime's LOSS and every parameter GRADIENT on the same
//! weights and batch — with a chunk genuinely smaller than the sequence and
//! supervision on a strict subset of rows (exercising the row scatter's
//! zero-clear).

use std::collections::HashMap;

use lfm2::config::LfmConfig;
use lfm2::model::{Lfm, IGNORE};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn chunked_training_matches_materialized() {
    let cfg = LfmConfig::tiny();
    let (b, t) = (2u32, 200u32);
    let n = (b * t) as usize;
    let init: HashMap<String, Vec<f32>> = lfm2::init::init_weights(&cfg, 11);
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    // Supervise ~1/3 of positions (MLM-shaped), unshifted targets.
    let targets: Vec<u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, &tk)| if i % 3 == 0 { tk } else { IGNORE })
        .collect();
    let names: Vec<(String, usize)> = cfg.param_list();

    let (loss_m, grads_m) = {
        let m = Lfm::new_train(cfg.clone(), b, t, &init);
        m.set_batch(&tokens, &targets);
        let loss = m.forward();
        m.zero_grads();
        m.backward();
        let grads: Vec<Vec<f32>> = names.iter().map(|(nm, _)| m.read_grad(nm)).collect();
        (loss, grads)
    }; // materialized model (and its device) dropped here

    // Slab budget forcing chunk == 64 (< t == 200 ⇒ 4 chunks incl. a tail);
    // head cap just above the supervised count.
    let budget = cfg.n_heads as u64 * 64 * t as u64 * 4;
    let n_sup = targets.iter().filter(|&&v| v != IGNORE).count() as u32;
    let m = Lfm::new_train_chunked(cfg.clone(), b, t, &init, budget, n_sup + 5);
    m.set_batch(&tokens, &targets);
    let loss_c = m.forward();
    m.zero_grads();
    m.backward();

    eprintln!("loss materialized {loss_m:.6} vs chunked {loss_c:.6}");
    assert!((loss_m - loss_c).abs() <= 1e-4, "loss diverges: {loss_m} vs {loss_c}");
    for ((nm, _), want) in names.iter().zip(&grads_m) {
        let got = m.read_grad(nm);
        let d = max_abs_diff(&got, want);
        let scale = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(
            d <= 1e-4 + 1e-3 * scale,
            "grad {nm}: max|Δ| {d:.2e} (scale {scale:.2e})"
        );
    }
    eprintln!("all {} parameter grads match", names.len());
}
