// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end parity vs the reference FinCast forward.
//!
//! The committed golden (`tests/golden/golden_meta.json`, from
//! `tools/goldens/fincast_dump_reference.py` on the real `v1.pth` with stochastic MoE
//! routing neutralized to deterministic top-2) carries the full fixed context
//! and the reference's denormalized head output for the last patch
//! (`[horizon_len, num_outputs]`). This test loads the real weights
//! (`FINCAST_CKPT` → the converted safetensors), runs brain's first AR-decode
//! step over the same context, and asserts **cosine > 0.99 AND Pearson > 0.99**
//! plus a bounded relative RMS — the continuous-output parity gate from the plan
//! (Pearson because it is scale-invariant).
//!
//! Env-gated: skips (does not fail) when `FINCAST_CKPT` is unset.

use fincast::{Fincast, FincastConfig};
use std::collections::HashMap;

fn read_golden() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/golden_meta.json");
    let bytes = std::fs::read(path).expect("read golden_meta.json");
    serde_json::from_slice(&bytes).unwrap()
}

fn farr(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn end_to_end_matches_reference() {
    let Ok(ckpt) = std::env::var("FINCAST_CKPT") else {
        eprintln!("FINCAST_CKPT unset; skipping FinCast end-to-end parity");
        return;
    };
    let g = read_golden();
    let context = farr(&g["context_full"]);
    let golden = farr(&g["output_full"]); // [horizon_len * num_outputs], step-major
    let freq = g["freq"].as_u64().unwrap() as usize;
    let horizon = g["horizon_len"].as_u64().unwrap() as usize;

    let cfg = FincastConfig::default();
    let weights: HashMap<String, Vec<f32>> = fincast::import::load_hf(&cfg, &ckpt).expect("load weights");
    let model = Fincast::from_weights_on(gpu_core::testgpu::dev(fincast::model::PIPELINES), cfg.clone(), &weights).unwrap();

    // brain's first AR step: forecast_full over the same context, one patch of
    // horizon (== horizon_len). Layout: [horizon, num_outputs] step-major — same
    // as the reference's last-patch output.
    let out = model.forecast_full(&context, freq, horizon);
    assert_eq!(out.len(), golden.len(), "output length mismatch");

    // cosine + Pearson over the full flattened output.
    let n = out.len() as f32;
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f32, 0f32, 0f32, 0f32, 0f32);
    let mut se = 0f32;
    for (a, b) in out.iter().zip(&golden) {
        dot += a * b;
        na += a * a;
        nb += b * b;
        sa += a;
        sb += b;
        saa += a * a;
        sbb += b * b;
        sab += a * b;
        se += (a - b) * (a - b);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
    let cov = sab - sa * sb / n;
    let va = saa - sa * sa / n;
    let vb = sbb - sb * sb / n;
    let pearson = cov / (va.sqrt() * vb.sqrt() + 1e-12);
    let rrms = (se / nb).sqrt(); // rms error relative to golden magnitude

    eprintln!("FinCast parity: cosine={cos:.6} pearson={pearson:.6} rel_rms={rrms:.6}");
    assert!(cos > 0.99, "cosine {cos} <= 0.99");
    assert!(pearson > 0.99, "pearson {pearson} <= 0.99");
    assert!(rrms < 0.02, "relative rms {rrms} too large");
}
