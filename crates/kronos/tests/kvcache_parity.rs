// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! KV-cache parity + speed: the cached rollout must reproduce the un-cached
//! decoder's forecast (cosine > 0.999) while being much faster.
//!
//! Env-gated on `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` (the HF checkpoint
//! dirs); skips otherwise so CI without the weights stays green.

use kronos::{import, GenOpts, KronosDecoder};
use std::time::Instant;

fn write_f32(path: &str, a: &[f32]) {
    let mut b = Vec::with_capacity(a.len() * 4);
    for &v in a {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &b).unwrap();
}

/// Env-gated: dump the host embedding tables (emb_s1/emb_s2/fusion_proj) the NPU
/// AR-loop driver needs to build `decode_s1`'s input on the host.
#[test]
fn dump_embedding_tables() {
    let Ok(dec) = std::env::var("BRAIN_KRONOS_DECODER") else {
        return brain_testutil::skip("BRAIN_KRONOS_DECODER unset; no embedding tables to dump");
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
    }
    let (cfg, w) = import::load_decoder(&dec).expect("load kronos decoder");
    let d = cfg.d_model;
    let decoder = KronosDecoder::from_weights(cfg, &w).expect("build decoder");
    let hw = decoder.host_weights();
    let dir = std::env::var("KRONOS_PARITY_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    write_f32(&format!("{dir}/k_emb_s1.f32"), &hw.emb_s1);
    write_f32(&format!("{dir}/k_emb_s2.f32"), &hw.emb_s2);
    write_f32(&format!("{dir}/k_fusion_w.f32"), &hw.fusw);
    write_f32(&format!("{dir}/k_fusion_b.f32"), &hw.fusb);
    eprintln!("dumped emb tables: emb_s1[{}] emb_s2[{}] fus_w[{}] d={}", hw.emb_s1.len(), hw.emb_s2.len(), hw.fusw.len(), d);
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

#[test]
fn cached_matches_uncached_and_is_faster() {
    let (Ok(tok), Ok(dec)) = (std::env::var("BRAIN_KRONOS_TOKENIZER"), std::env::var("BRAIN_KRONOS_DECODER")) else {
        return brain_testutil::skip("BRAIN_KRONOS_TOKENIZER / BRAIN_KRONOS_DECODER unset; no KV-cache parity");
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
    }
    let m = import::load_model(&tok, &dec).expect("load kronos model");
    let feat = m.feat();
    let t = 256usize; // context bars
    let pred_len = 20usize;
    // deterministic synthetic OHLCV (coherent-ish; the tokenizer just needs values)
    let mut bars = vec![0f32; t * feat];
    for i in 0..t {
        let base = 100.0 + 10.0 * (i as f32 * 0.05).sin();
        for c in 0..feat {
            bars[i * feat + c] = base + c as f32 * 0.3;
        }
    }
    let ctx_stamp = vec![0u32; t * 5];
    let fut_stamp = vec![0u32; pred_len * 5];
    let opts = GenOpts::default(); // argmax — deterministic

    let t0 = Instant::now();
    let uncached = m.forecast(&bars, &ctx_stamp, &fut_stamp, pred_len, &opts);
    let dt_u = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let cached = m.forecast_cached(&bars, &ctx_stamp, &fut_stamp, pred_len, &opts);
    let dt_c = t0.elapsed().as_secs_f64();

    assert_eq!(uncached.len(), cached.len());
    let cos = cosine(&uncached, &cached);
    let mx = uncached
        .iter()
        .zip(&cached)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let rng = uncached.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - uncached.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!(
        "KV-cache parity: cosine={cos:.6} rel_max_abs={:.5} | uncached {:.2}s  cached {:.2}s  speedup {:.1}x",
        mx / (rng.abs() + 1e-6),
        dt_u,
        dt_c,
        dt_u / dt_c.max(1e-9)
    );

    assert!(cos > 0.999, "cached vs uncached cosine {cos:.6} must exceed 0.999");
    // Speed: the cached path is host scalar f32, so it only wins in an optimized
    // build (debug host code is ~100x slower than the GPU-kernel uncached path).
    if !cfg!(debug_assertions) {
        assert!(dt_c < dt_u, "release: cached ({dt_c:.2}s) should beat uncached ({dt_u:.2}s)");
    }
}
