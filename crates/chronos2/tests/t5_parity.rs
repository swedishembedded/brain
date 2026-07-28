// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T5 — end-to-end forward parity against the reference Chronos-2.
//!
//! Compares brain's `forecast_quantiles` to a golden dump from the official
//! `Chronos2Pipeline` on the *identical* context (both read `t5_context.f32`).
//! The continuous-output gate is **cosine > 0.99 AND Pearson > 0.99** (scale- and
//! offset-robust), plus a sane relative max-abs error.
//!
//! Env-gated: runs only when `CHRONOS2_WEIGHTS` points at an imported
//! `chronos2.weights` and the golden files exist (regenerate them with
//! `tools/chronos2_dump_reference.py`). Skips (does not fail) otherwise so CI
//! without the 478 MB checkpoint stays green.

use chronos2::Chronos2;
use std::path::Path;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    cov / (va.sqrt() * vb.sqrt() + 1e-12)
}

#[test]
fn full_forward_matches_the_reference() {
    let Ok(weights) = std::env::var("CHRONOS2_WEIGHTS") else {
        eprintln!("CHRONOS2_WEIGHTS unset; skipping T5 Chronos-2 parity");
        return;
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let ctx_path = golden.join("t5_context.f32");
    let q_path = golden.join("t5_quantiles.f32");
    let meta_path = golden.join("t5_meta.json");
    if !ctx_path.exists() || !q_path.exists() {
        eprintln!("golden dump missing; run tools/chronos2_dump_reference.py — skipping");
        return;
    }

    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
    let horizon = meta["horizon"].as_u64().unwrap() as usize;
    let nq = meta["n_quantiles"].as_u64().unwrap() as usize;

    let context = read_f32(&ctx_path);
    let reference = read_f32(&q_path); // [nq, horizon] row-major (quantile-major)
    assert_eq!(reference.len(), nq * horizon, "golden shape mismatch");

    let model = Chronos2::load_on(gpu_core::testgpu::dev(chronos2::model::PIPELINES), &weights).expect("load chronos2.weights");
    let brain = model.forecast_quantiles(&context, horizon); // [nq, horizon] quantile-major
    assert_eq!(brain.len(), nq * horizon, "brain forecast shape");

    let cos = cosine(&brain, &reference);
    let pear = pearson(&brain, &reference);
    // relative max-abs over the value range
    let (mut max_abs, mut lo, mut hi) = (0.0f32, f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..reference.len() {
        max_abs = max_abs.max((brain[i] - reference[i]).abs());
        lo = lo.min(reference[i]);
        hi = hi.max(reference[i]);
    }
    let rel = max_abs / (hi - lo).max(1e-6);
    eprintln!("T5 chronos-2 parity: cosine={cos:.6} pearson={pear:.6} rel_max_abs={rel:.4} (range {lo:.3}..{hi:.3})");

    assert!(cos > 0.99, "cosine {cos:.6} must exceed 0.99 (forecast diverges from reference)");
    assert!(pear > 0.99, "pearson {pear:.6} must exceed 0.99 (structure diverges)");
    assert!(rel < 0.05, "relative max-abs {rel:.4} too large (values off by > 5% of range)");
}
