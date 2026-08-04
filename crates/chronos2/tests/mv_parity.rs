// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multivariate parity — brain's `forecast_quantiles_mv([target, cov], h)` vs a
//! reference Chronos-2 run with both series in one group (`group_ids=[0,0]`,
//! unknown futures), keeping the target's quantile row.
//!
//! This validates the B>1 group-attention path: the target's forecast is
//! informed by the covariate through group attention at every patch position.
//! Gate: **cosine > 0.99 AND Pearson > 0.99** plus a sane relative max-abs.
//!
//! Env-gated on `CHRONOS2_WEIGHTS` + the golden dump from
//! `tools/goldens/chronos2_dump_mv_reference.py`; skips otherwise.

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
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    cov / (va.sqrt() * vb.sqrt() + 1e-12)
}

#[test]
fn multivariate_forward_matches_the_reference() {
    let Ok(weights) = std::env::var("CHRONOS2_WEIGHTS") else {
        eprintln!("CHRONOS2_WEIGHTS unset; skipping multivariate parity");
        return;
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let (tp, cp, qp, mp) = (
        golden.join("mv_target.f32"),
        golden.join("mv_cov.f32"),
        golden.join("mv_quantiles.f32"),
        golden.join("mv_meta.json"),
    );
    if !tp.exists() || !cp.exists() || !qp.exists() {
        eprintln!("mv golden missing; run tools/goldens/chronos2_dump_mv_reference.py — skipping");
        return;
    }

    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
    let horizon = meta["horizon"].as_u64().unwrap() as usize;
    let nq = meta["n_quantiles"].as_u64().unwrap() as usize;

    let target = read_f32(&tp);
    let cov = read_f32(&cp);
    let reference = read_f32(&qp); // target row [nq, horizon]
    assert_eq!(reference.len(), nq * horizon, "golden shape mismatch");

    let model = Chronos2::load_on(gpu_core::testgpu::dev(chronos2::model::PIPELINES), &weights).expect("load chronos2.safetensors");
    let brain = model.forecast_quantiles_mv(&[&target, &cov], horizon);
    assert_eq!(brain.len(), nq * horizon, "brain mv forecast shape");

    let cos = cosine(&brain, &reference);
    let pear = pearson(&brain, &reference);
    let (mut max_abs, mut lo, mut hi) = (0.0f32, f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..reference.len() {
        max_abs = max_abs.max((brain[i] - reference[i]).abs());
        lo = lo.min(reference[i]);
        hi = hi.max(reference[i]);
    }
    let rel = max_abs / (hi - lo).max(1e-6);
    eprintln!("MV chronos-2 parity: cosine={cos:.6} pearson={pear:.6} rel_max_abs={rel:.4} (range {lo:.3}..{hi:.3})");

    assert!(cos > 0.99, "cosine {cos:.6} must exceed 0.99");
    assert!(pear > 0.99, "pearson {pear:.6} must exceed 0.99");
    assert!(rel < 0.05, "relative max-abs {rel:.4} too large");
}
