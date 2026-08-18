// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Known-future covariate parity — brain's `forecast_quantiles_mv_kf` (target +
//! a covariate whose future is provided) vs the reference Chronos-2 run with
//! `future_covariates` carrying the covariate's future (target future = NaN).
//!
//! Gate: cosine > 0.99 AND Pearson > 0.99. Env-gated on `CHRONOS2_WEIGHTS` + the
//! golden dump from `tools/goldens/chronos2_dump_kf_reference.py`; skips otherwise.

use chronos2::Chronos2;
use std::path::Path;

use brain_testutil::testdata_path;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p:?}: {e}"))
        .chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12)
}
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db; va += da * da; vb += db * db;
    }
    cov / (va.sqrt() * vb.sqrt() + 1e-12)
}

#[test]
fn known_future_forward_matches_the_reference() {
    let Ok(weights) = std::env::var("CHRONOS2_WEIGHTS") else {
        brain_testutil::skip("CHRONOS2_WEIGHTS unset");
        return;
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let g = testdata_path("golden/chronos2");
    let (tp, cp, fp, qp, mp) = (
        g.join("kf_target.f32"), g.join("kf_cov.f32"), g.join("kf_cov_future.f32"),
        g.join("kf_quantiles.f32"), g.join("kf_meta.json"),
    );
    if !tp.exists() || !fp.exists() || !qp.exists() {
        brain_testutil::skip("kf golden missing; run tools/goldens/chronos2_dump_kf_reference.py");
        return;
    }
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&mp).unwrap()).unwrap();
    let horizon = meta["horizon"].as_u64().unwrap() as usize;
    let nq = meta["n_quantiles"].as_u64().unwrap() as usize;

    let target = read_f32(&tp);
    let cov = read_f32(&cp);
    let cov_future = read_f32(&fp);
    let reference = read_f32(&qp);
    assert_eq!(reference.len(), nq * horizon);

    let model = Chronos2::load_on(gpu_core::testgpu::dev(chronos2::model::PIPELINES), &weights).expect("load weights");
    // target future unknown; covariate future known.
    let series: Vec<&[f32]> = vec![&target, &cov];
    let futures: Vec<Option<&[f32]>> = vec![None, Some(&cov_future)];
    let brain = model.forecast_quantiles_mv_kf(&series, &futures, horizon);
    assert_eq!(brain.len(), nq * horizon);

    let (cos, pear) = (cosine(&brain, &reference), pearson(&brain, &reference));
    let (mut mx, mut lo, mut hi) = (0.0f32, f32::INFINITY, f32::NEG_INFINITY);
    for i in 0..reference.len() {
        mx = mx.max((brain[i] - reference[i]).abs());
        lo = lo.min(reference[i]); hi = hi.max(reference[i]);
    }
    let rel = mx / (hi - lo).max(1e-6);
    eprintln!("KF chronos-2 parity: cosine={cos:.6} pearson={pear:.6} rel_max_abs={rel:.4}");
    assert!(cos > 0.99, "cosine {cos:.6}");
    assert!(pear > 0.99, "pearson {pear:.6}");
    assert!(rel < 0.05, "rel_max_abs {rel:.4}");
}
