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

use data::rng::Lcg;
use std::collections::HashMap;

use lfm2::config::LfmConfig;
use lfm2::model::Lfm;

/// `Lfm::new`/`new_chunked` have no `_on(gpu, ...)` variant - every constructor
/// builds its own real device via `Gpu::new` internally (see
/// `crates/lfm2/src/model.rs`'s `new_impl_alloc_inner`), so this file's two
/// tests cannot share one via `gpu_core::testgpu::dev` the way most GPU tests
/// do. Each test already builds/drops sequentially WITHIN itself (this file's
/// own module doc), but under `cargo test`'s default multi-threaded run nothing
/// stopped the two `#[test]` functions running concurrently on separate threads
/// and racing their own independent device builds against each other - several
/// concurrent devices on one card are hostile to the driver
/// (`gpu_core::Gpu::share`'s own doc comment), the same hazard
/// `crates/gpu-core/tests/it/device_sharing.rs`'s `DEVICE_SERIAL` and
/// `device_churn.rs`'s own copy of it exist to prevent. Same fix here.
static DEVICE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Deterministic pseudo-random init (small values keep fp32 paths well-scaled).
fn lcg_init(cfg: &LfmConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Lcg::new(seed | 1);
    cfg.param_list()
        .into_iter()
        .map(|(name, numel)| {
            let scale = if name.ends_with("norm.weight") || name.contains("ln") { 1.0 } else { 0.08 };
            let base = if name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "norm.weight" || name.contains("_norm") { 1.0 } else { 0.0 };
            (name, (0..numel).map(|_| base + scale * rng.scaled(0.5)).collect())
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn chunked_matches_materialized() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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

/// Same shape as [`chunked_matches_materialized`] (heads=4, head_dim=4, t=200,
/// chunk=64 — head_dim isn't a multiple of 64 words, so gemm_bidir_fwd's
/// per-head packed strides are unaligned before padding) but with b=1: the
/// `qkv`/`ctx_pack` scratch buffers then have zero slack over the raw
/// (unpadded) per-span size, so this also regresses the buffer-size audit —
/// padding without enlarging those allocations would overflow them.
#[test]
fn chunked_matches_materialized_b1() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = LfmConfig::tiny();
    let (b, t) = (1u32, 200u32);
    let n = (b * t) as usize;
    let init = lcg_init(&cfg, 0x9e3779b9);
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let probe_rows: Vec<u32> = vec![0, 63, 64, 199];

    let (hidden_m, logit_rows_m) = {
        let m = Lfm::new(cfg.clone(), b, t, &init);
        m.set_tokens(&tokens);
        m.forward();
        let hidden = m.read_hidden();
        let logits = m.read_logits();
        let v = cfg.vocab as usize;
        let rows: Vec<Vec<f32>> = probe_rows.iter().map(|&r| logits[r as usize * v..(r as usize + 1) * v].to_vec()).collect();
        (hidden, rows)
    };

    let budget = cfg.n_heads as u64 * 64 * t as u64 * 4;
    let m = Lfm::new_chunked(cfg.clone(), b, t, &init, budget, probe_rows.len() as u32);
    assert_eq!(m.chunk(), Some(64), "budget arithmetic drifted");
    m.set_tokens(&tokens);
    m.set_probe_rows(&probe_rows);
    m.forward();

    let hidden_c = m.read_hidden();
    let d_h = max_abs_diff(&hidden_m, &hidden_c);
    eprintln!("b=1 hidden max|Δ| = {d_h:.2e}");
    assert!(d_h <= 1e-5, "chunked hidden diverges (b=1): {d_h:.2e}");

    let probe = m.read_probe_logits();
    let v = cfg.vocab as usize;
    for (i, want) in logit_rows_m.iter().enumerate() {
        let got = &probe[i * v..(i + 1) * v];
        let dl = max_abs_diff(got, want);
        assert!(dl <= 1e-4, "b=1 probe row {i} (abs row {}): max|Δ| {dl:.2e}", probe_rows[i]);
    }
}

/// The tiny config (`d_model` 16) batched: the second sequence starts at row 42, so its q rows start at 42 * 16 words
/// (2688 bytes) and its k and v rows at 42 * 8, neither on the 256-byte storage-binding grain. Whichever accelerated
/// attention (flash or GEMM) the device selects for the chunked regime, the batched forward must bind only aligned
/// views and reproduce the materialized forward. The cross trio a device without workgroup reductions runs requires
/// widths on the grain (`block::chunked_bidir_fwd`) and refuses this shape at the call.
#[test]
fn chunked_matches_materialized_for_a_narrow_batch() {
    let _serial = DEVICE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let cfg = LfmConfig::tiny();
    let (b, t) = (2u32, 42u32);
    let n = (b * t) as usize;
    let init = lcg_init(&cfg, 0x2a2a_1616);
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 11 + 2) % cfg.vocab).collect();

    let hidden_m = {
        let m = Lfm::new(cfg.clone(), b, t, &init);
        m.set_tokens(&tokens);
        m.forward();
        m.read_hidden()
    };
    // The materialized model built the process's device, so its caps are known now.
    if !gpu_core::device_caps().is_some_and(|caps| caps.workgroup_reductions) {
        brain_testutil::skip_unavailable("narrow batched chunked forward: flash and GEMM attention need workgroup reductions (a GPU)");
        return;
    }

    let budget = cfg.n_heads as u64 * 64 * t as u64 * 4;
    let m = Lfm::new_chunked(cfg.clone(), b, t, &init, budget, 1);
    assert_eq!(m.chunk(), Some(t), "one chunk covers each sequence");
    m.set_tokens(&tokens);
    m.set_probe_rows(&[0]);
    m.forward();
    let d_h = max_abs_diff(&hidden_m, &m.read_hidden());
    assert!(d_h <= 1e-5, "narrow batched chunked hidden diverges: {d_h:.2e}");
}
