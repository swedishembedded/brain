// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chunked-vs-materialized equivalence: the query-chunked inference regime
//! must reproduce the materialized forward's hidden states and (probe-row)
//! logits on the same weights and tokens — with a chunk genuinely smaller than
//! the sequence so multiple chunks + the tail chunk are exercised.
//!
//! Random (deterministic LCG) weights on a tiny config; runs on any device
//! (`BRAIN_DEVICE=cpu` in CI). Models are built strictly sequentially — the
//! first is dropped before the second exists (one live device at a time).

use std::collections::HashMap;

use lfm::config::LfmConfig;
use lfm::model::Lfm;

/// Deterministic pseudo-random init (small values keep fp32 paths well-scaled).
fn lcg_init(cfg: &LfmConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    cfg.param_list()
        .into_iter()
        .map(|(name, numel)| {
            let scale = if name.ends_with("norm.weight") || name.contains("ln") { 1.0 } else { 0.08 };
            let base = if name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "norm.weight" || name.contains("_norm") { 1.0 } else { 0.0 };
            (name, (0..numel).map(|_| base + scale * next()).collect())
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn chunked_matches_materialized() {
    let cfg = LfmConfig::tiny();
    let (b, t) = (2u32, 200u32);
    let n = (b * t) as usize;
    let init = lcg_init(&cfg, 0x1f2e3d4c);
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let probe_rows: Vec<u32> = vec![0, 63, 64, 199, 200, 399]; // chunk edges + both sequences

    let (hidden_m, logit_rows_m) = {
        let m = Lfm::new(cfg.clone(), b, t, &init);
        m.set_tokens(&tokens);
        m.forward();
        let hidden = m.read_hidden();
        let logits = m.read_logits();
        let v = cfg.vocab as usize;
        let rows: Vec<Vec<f32>> = probe_rows.iter().map(|&r| logits[r as usize * v..(r as usize + 1) * v].to_vec()).collect();
        (hidden, rows)
    }; // materialized model (and its device) dropped here

    // Slab budget forcing chunk == 64 (< t == 200 ⇒ 4 chunks incl. a tail of 8).
    let budget = cfg.n_heads as u64 * 64 * t as u64 * 4;
    let m = Lfm::new_chunked(cfg.clone(), b, t, &init, budget, probe_rows.len() as u32);
    assert_eq!(m.chunk(), Some(64), "budget arithmetic drifted");
    m.set_tokens(&tokens);
    m.set_probe_rows(&probe_rows);
    m.forward();

    let hidden_c = m.read_hidden();
    let d_h = max_abs_diff(&hidden_m, &hidden_c);
    eprintln!("hidden max|Δ| = {d_h:.2e}");
    assert!(d_h <= 1e-5, "chunked hidden diverges: {d_h:.2e}");

    let probe = m.read_probe_logits();
    let v = cfg.vocab as usize;
    for (i, want) in logit_rows_m.iter().enumerate() {
        let got = &probe[i * v..(i + 1) * v];
        let dl = max_abs_diff(got, want);
        assert!(dl <= 1e-4, "probe row {i} (abs row {}): max|Δ| {dl:.2e}", probe_rows[i]);
    }
}
