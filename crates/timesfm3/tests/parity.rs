// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `core_forward` parity vs the real `google-research/timesfm` reference.
//!
//! Two rungs, per the parity ladder: a checkpoint-free tiny-config rung and a
//! real-checkpoint rung (additionally gated on `BRAIN_TIMESFM3`). Both need
//! `testdata/golden/timesfm3/manifest.json` - a regenerable dump
//! (`tools/goldens/timesfm3_dump_reference.py`), never committed (a golden
//! dump is exactly as regenerable as the checkpoint it was dumped from, and
//! this one is 800+ KB), so both rungs skip cleanly (do not fail) when it is
//! absent rather than only the real-checkpoint one. Both gate **cosine AND
//! relative L2** - cosine alone cannot see a dropped or doubled scale factor,
//! which is exactly the class of bug this model's attention-scale fold (see
//! `model.rs`'s module docs) is most at risk of.

use std::collections::HashMap;
use timesfm3::preprocess::{self, DecodeShape};
use timesfm3::{Timesfm3, Timesfm3Config};

fn read_golden() -> Option<serde_json::Value> {
    let path = brain_testutil::testdata_path("golden/timesfm3/manifest.json");
    if !path.exists() {
        brain_testutil::skip(&format!(
            "{} not found - regenerate with: BRAIN_TIMESFM3_REF=<google-research/timesfm checkout> python3 tools/goldens/timesfm3_dump_reference.py <fetched checkpoint dir> {}",
            path.display(),
            path.parent().unwrap().display()
        ));
        return None;
    }
    let bytes = std::fs::read(&path).expect("read manifest.json");
    Some(serde_json::from_slice(&bytes).unwrap())
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
    let Some(g) = read_golden() else { return; };
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
    let Some(g) = read_golden() else { return; };
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

/// Full `decode()` parity from RAW (unpatched) inputs: `preprocess::build_input`
/// (detrend, patch, running RevIN, future-covariate rolling) ->
/// `core_forward` (already gated above) -> `preprocess::postprocess` (CPM
/// refinement, RevIN-reverse, stitching, trend re-addition). This is the rung
/// that actually matters end to end - the two tests above only prove the
/// transformer core in isolation.
fn decode_end_to_end(g: &serde_json::Value, prefix: &str, cfg: Timesfm3Config, model: &Timesfm3, shape: DecodeShape) {
    let target = farr(&g[format!("{prefix}.input.target")]["full"]);
    let past_only = farr(&g[format!("{prefix}.input.past_only")]["full"]);
    let past_future = farr(&g[format!("{prefix}.input.past_future")]["full"]);

    let built = preprocess::build_input(&cfg, shape, &target, &past_only, &past_future);
    assert!(built.patch_mask.iter().all(|&m| !m), "no left-padding at this context length - nothing should be masked");

    let resblock_want = farr(&g[format!("{prefix}.resblock_input")]["full"]);
    let (cos, rel_l2) = cos_and_rel_l2(&built.resblock_input, &resblock_want);
    eprintln!("timesfm3 {prefix} build_input parity: cosine={cos:.9} rel_l2={rel_l2:.6}");
    assert!(cos > 0.999_9, "{prefix} resblock_input cosine {cos} too low");
    assert!(rel_l2 < 0.01, "{prefix} resblock_input relative L2 {rel_l2} too large");

    let n = built.num_context_patches + built.num_horizon_patches;
    let raw_logits = model.core_forward(&built.resblock_input, &built.patch_mask, shape.batch, shape.num_variates(), n);
    let out = preprocess::postprocess(&cfg, shape, &built, &raw_logits);

    let want = farr(&g[format!("{prefix}.horizon_logits")]["full"]);
    let (cos, rel_l2) = cos_and_rel_l2(&out, &want);
    eprintln!("timesfm3 {prefix} decode end-to-end parity: cosine={cos:.9} rel_l2={rel_l2:.6}");
    assert!(cos > 0.999_9, "{prefix} horizon_logits cosine {cos} too low");
    assert!(rel_l2 < 0.02, "{prefix} horizon_logits relative L2 {rel_l2} too large");
}

#[test]
fn tiny_config_decode_end_to_end_from_raw_inputs_matches_the_reference() {
    let Some(g) = read_golden() else { return; };
    let cfg = Timesfm3Config::from_hf_config_json(&g["tiny_config"]).unwrap();
    let weights: HashMap<String, Vec<f32>> =
        g["tiny_weights"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), farr(v))).collect();
    let model = Timesfm3::from_weights_on(gpu_core::testgpu::dev(timesfm3::model::PIPELINES), cfg.clone(), &weights).unwrap();

    let shape = DecodeShape { batch: 2, num_target: 2, num_past_only: 1, num_past_future: 1, context: 16, horizon: 8 };
    decode_end_to_end(&g, "tiny", cfg, &model, shape);
}

#[test]
fn real_checkpoint_decode_end_to_end_from_raw_inputs_matches_the_reference() {
    let Ok(path) = std::env::var("BRAIN_TIMESFM3") else {
        brain_testutil::skip("BRAIN_TIMESFM3 unset");
        return;
    };
    let Some(g) = read_golden() else { return; };
    let cfg = timesfm3::import::load_config(&path).unwrap();
    let weights = timesfm3::import::load_hf(&cfg, &path).unwrap();
    let model = Timesfm3::from_weights_on(gpu_core::testgpu::dev(timesfm3::model::PIPELINES), cfg.clone(), &weights).unwrap();

    let shape = DecodeShape { batch: 1, num_target: 1, num_past_only: 1, num_past_future: 1, context: 192, horizon: 64 };
    decode_end_to_end(&g, "real", cfg, &model, shape);
}

/// Stage-parity rung, isolated from `core_forward`/stitching/trend-readd:
/// `cpm_iterative_revin_refine` alone,
/// fed the golden's own `raw_logits` and pre-refine running stats, must
/// reproduce the reference's `cpm_refined_mu`/`cpm_refined_sigma` EXACTLY -
/// this caught a real bug the end-to-end test alone could not localize (the
/// anchor prediction grid updates on every patch, context included, not only
/// CPM ones - see `cpm_iterative_revin_refine`'s doc comment).
#[test]
fn cpm_refine_alone_matches_the_golden_exactly() {
    let Some(g) = read_golden() else { return; };
    let want_mu = farr(&g["tiny.cpm_refined_mu"]["full"]);
    let want_sigma = farr(&g["tiny.cpm_refined_sigma"]["full"]);
    let raw_logits = farr(&g["tiny.raw_logits"]["full"]);
    let cfg = Timesfm3Config::from_hf_config_json(&g["tiny_config"]).unwrap();
    let target = farr(&g["tiny.input.target"]["full"]);
    let past_only = farr(&g["tiny.input.past_only"]["full"]);
    let past_future = farr(&g["tiny.input.past_future"]["full"]);
    let shape = DecodeShape { batch: 2, num_target: 2, num_past_only: 1, num_past_future: 1, context: 16, horizon: 8 };
    let built = preprocess::build_input(&cfg, shape, &target, &past_only, &past_future);
    let (b, v, n) = (2usize, 4usize, 6usize);
    let mut patch_cpm_mask = vec![false; n];
    patch_cpm_mask[built.num_context_patches..n].fill(true);
    let (mu, sigma) = preprocess::cpm_iterative_revin_refine(&raw_logits, &built.running_n, &built.running_mu, &built.running_sigma, &patch_cpm_mask, b, v, n, cfg.output_patch_len, cfg.num_quantiles, cfg.rolls(), cfg.value_clip);
    for i in 0..mu.len() {
        assert!((mu[i] - want_mu[i]).abs() < 1e-4, "mu[{i}]: got {} want {}", mu[i], want_mu[i]);
        assert!((sigma[i] - want_sigma[i]).abs() < 1e-4, "sigma[{i}]: got {} want {}", sigma[i], want_sigma[i]);
    }
}
