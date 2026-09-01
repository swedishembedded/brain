// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `core_forward` parity vs the real `google-research/timesfm` reference.
//!
//! Two rungs, per the parity ladder: a checkpoint-free tiny-config rung that
//! runs in CI with no download (weights, input and expected output are all
//! embedded in the committed `tests/golden/manifest.json`), and a
//! real-checkpoint rung gated on `BRAIN_TIMESFM3` (skips, does not fail, when
//! unset). Both gate **cosine AND relative L2** - cosine alone cannot see a
//! dropped or doubled scale factor, which is exactly the class of bug this
//! model's attention-scale fold (see `model.rs`'s module docs) is most at
//! risk of.

use std::collections::HashMap;
use timesfm3::{Timesfm3, Timesfm3Config};

fn read_golden() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/manifest.json");
    let bytes = std::fs::read(path).expect("read manifest.json");
    serde_json::from_slice(&bytes).unwrap()
}

fn farr(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn shape(v: &serde_json::Value) -> Vec<usize> {
    v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

/// cosine similarity and relative L2 error, `‖got-want‖ / ‖want‖` - the
/// scale-sensitive complement cosine alone cannot provide (lesson: `got =
/// 1.05*want` scores cosine 1.0).
fn cos_and_rel_l2(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len());
    let (mut dot, mut na, mut nb, mut se) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        se += (a - b) * (a - b);
    }
    let cos = (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32;
    let rel_l2 = (se.sqrt() / (nb.sqrt() + 1e-12)) as f32;
    (cos, rel_l2)
}

#[test]
fn tiny_config_core_forward_matches_the_reference_end_to_end() {
    let g = read_golden();
    let cfg = Timesfm3Config::from_hf_config_json(&g["tiny_config"]).unwrap();
    // `max_context` is not part of the upstream config.json schema at all (a
    // forecaster-level constant, not a checkpoint hyperparameter - see
    // `from_hf_config_json`'s docs) and does not affect `core_forward`, so it
    // is the one field this round-trip cannot and need not reproduce.
    assert_eq!(Timesfm3Config { max_context: cfg.max_context, ..Timesfm3Config::tiny() }, cfg, "the golden's own config must still describe tiny()");

    let weights: HashMap<String, Vec<f32>> =
        g["tiny_weights"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), farr(v))).collect();
    let model = Timesfm3::from_weights_on(gpu_core::testgpu::dev(timesfm3::model::PIPELINES), cfg.clone(), &weights).unwrap();

    let resblock_input = farr(&g["tiny.resblock_input"]["full"]);
    let s = shape(&g["tiny.resblock_input"]);
    let (b, v, n) = (s[0], s[1], s[2]);
    assert_eq!(s[3], cfg.resblock_in_dim());
    let mask = vec![false; b * v * n]; // no left-padding at this context length - verified empirically against the reference

    let got = model.core_forward(&resblock_input, &mask, b, v, n);
    let want = farr(&g["tiny.raw_logits"]["full"]);
    assert_eq!(got.len(), want.len());

    let (cos, rel_l2) = cos_and_rel_l2(&got, &want);
    eprintln!("timesfm3 tiny parity: cosine={cos:.9} rel_l2={rel_l2:.6}");
    assert!(cos > 0.999_9, "cosine {cos} too low");
    assert!(rel_l2 < 0.01, "relative L2 {rel_l2} too large");
}

#[test]
fn real_checkpoint_core_forward_matches_the_reference_end_to_end() {
    let Ok(path) = std::env::var("BRAIN_TIMESFM3") else {
        brain_testutil::skip("BRAIN_TIMESFM3 unset");
        return;
    };
    let g = read_golden();
    let cfg = timesfm3::import::load_config(&path).unwrap();
    assert_eq!(cfg, Timesfm3Config::default());

    let weights = timesfm3::import::load_hf(&cfg, &path).unwrap();
    let model = Timesfm3::from_weights_on(gpu_core::testgpu::dev(timesfm3::model::PIPELINES), cfg.clone(), &weights).unwrap();

    let resblock_input = farr(&g["real.resblock_input"]["full"]);
    let s = shape(&g["real.resblock_input"]);
    let (b, v, n) = (s[0], s[1], s[2]);
    assert_eq!(s[3], cfg.resblock_in_dim());
    let mask = vec![false; b * v * n];

    let got = model.core_forward(&resblock_input, &mask, b, v, n);
    let want = farr(&g["real.raw_logits"]["full"]);
    assert_eq!(got.len(), want.len());

    let (cos, rel_l2) = cos_and_rel_l2(&got, &want);
    eprintln!("timesfm3 real parity: cosine={cos:.9} rel_l2={rel_l2:.6}");
    assert!(cos > 0.999_9, "cosine {cos} too low");
    assert!(rel_l2 < 0.01, "relative L2 {rel_l2} too large");
}
