// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Timesfm3Forecaster` wiring: a `Panel` with all three `Role`s TimesFM-3
//! natively understands (`Target`, `PastCovariate`, `KnownFuture`) produces
//! the same numbers as calling `preprocess::build_input`/`core_forward`/
//! `postprocess` directly with the equivalent raw arrays - this is a
//! composition/wiring check (the math itself is parity-gated in
//! `tests/parity.rs`), reusing the checkpoint-free tiny-config golden so it
//! needs no download. The golden is regenerable numeric data, not checked
//! in; these tests skip (rather than fail) when it hasn't been generated -
//! see `read_golden`.

use std::collections::HashMap;
use forecast::{Capabilities, CovariateSupport, ForecastModel, ForecastSpec, Item, Panel, Representation, Role, Variate};
use timesfm3::preprocess::{self, DecodeShape};
use timesfm3::{Timesfm3, Timesfm3Config, Timesfm3Forecaster};

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

fn load_tiny_model(g: &serde_json::Value) -> (Timesfm3Config, Timesfm3) {
    let cfg = Timesfm3Config::from_hf_config_json(&g["tiny_config"]).unwrap();
    let weights: HashMap<String, Vec<f32>> = g["tiny_weights"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), farr(v))).collect();
    let model = Timesfm3::from_weights_on(gpu_core::testgpu::dev(timesfm3::model::PIPELINES), cfg.clone(), &weights).unwrap();
    (cfg, model)
}

#[test]
fn capabilities_advertise_native_multivariate_full_covariates() {
    let Some(g) = read_golden() else { return; };
    let (_, model) = load_tiny_model(&g);
    let f = Timesfm3Forecaster::new(model);
    let caps: Capabilities = f.capabilities();
    assert_eq!(caps.name, "timesfm3");
    assert!(caps.multivariate);
    assert!(caps.supports_known_future);
    assert_eq!(caps.covariates, CovariateSupport::Full);
    assert_eq!(caps.native_representation, Representation::Quantiles);
}

#[test]
fn forecast_over_a_panel_matches_the_equivalent_direct_decode_call() {
    let Some(g) = read_golden() else { return; };
    let (cfg, model) = load_tiny_model(&g);
    let target = farr(&g["tiny.input.target"]["full"]); // [2,2,16]: 2 batches, 2 targets, context 16
    let past_only = farr(&g["tiny.input.past_only"]["full"]); // [2,1,16]
    let past_future = farr(&g["tiny.input.past_future"]["full"]); // [2,1,24] = context 16 + horizon 8
    let (context, horizon) = (16usize, 8usize);

    // Batch element 0 only - the forecaster processes one `Item` at a time.
    let t0 = target[0..context].to_vec();
    let t1 = target[context..2 * context].to_vec();
    let po = past_only[0..context].to_vec();
    let pf_ctx = past_future[0..context].to_vec();
    let pf_fut = past_future[context..context + horizon].to_vec();

    let item = Item::new(
        "series-0",
        vec![
            Variate::target("target_a", t0.clone()),
            Variate::target("target_b", t1.clone()),
            {
                let mut v = Variate::target("past_cov", po.clone());
                v.role = Role::PastCovariate;
                v
            },
            {
                let mut v = Variate::target("known_future_cov", pf_ctx.clone());
                v.role = Role::KnownFuture;
                v.future = Some(pf_fut.clone());
                v
            },
        ],
    );
    let panel = Panel::single("H", "series-0", item.variates.clone());
    // Requesting exactly the native levels makes the forecaster's
    // interpolation-to-requested-levels step an identity, so the result can
    // be compared directly against `postprocess`'s raw quantile output.
    let spec = ForecastSpec { horizon, quantile_levels: cfg.quantile_levels.clone(), ..ForecastSpec::default() };

    let model2 = {
        let (_, m) = load_tiny_model(&g);
        m
    };
    let forecaster = Timesfm3Forecaster::new(model2);
    let fc = forecaster.forecast(&panel, &spec).expect("forecast");
    assert_eq!(fc.targets.len(), 2, "both target variates must be forecast, in one native multivariate call");

    // Direct equivalent: one decode() call over the same 4 variates.
    let shape = DecodeShape { batch: 1, num_target: 2, num_past_only: 1, num_past_future: 1, context, horizon };
    let mut target_data = t0.clone();
    target_data.extend_from_slice(&t1);
    let built = preprocess::build_input(&cfg, shape, &target_data, &po, &pf_ctx.iter().chain(&pf_fut).copied().collect::<Vec<_>>());
    let n = built.num_context_patches + built.num_horizon_patches;
    let raw_logits = model.core_forward(&built.resblock_input, &built.patch_mask, 1, shape.num_variates(), n);
    let mut want = preprocess::postprocess(&cfg, shape, &built, &raw_logits);
    for row in want.chunks_exact_mut(cfg.num_quantiles) {
        row.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    for (ti, tf) in fc.targets.iter().enumerate() {
        let q = tf.quantiles.as_ref().unwrap();
        assert_eq!(q.shape, vec![horizon, spec.quantile_levels.len()]);
        // The forecaster interpolates to the REQUESTED levels; asking for
        // exactly the native levels makes the interpolation an identity, so
        // this can compare straight against `postprocess`'s raw quantile
        // output for the same target's slice.
        let want_slice = &want[ti * horizon * cfg.num_quantiles..(ti + 1) * horizon * cfg.num_quantiles];
        for t in 0..horizon {
            for (li, &lv) in spec.quantile_levels.iter().enumerate() {
                let qi = cfg.quantile_levels.iter().position(|&x| (x - lv).abs() < 1e-6).unwrap();
                let got = q.data[t * spec.quantile_levels.len() + li];
                let want_v = want_slice[t * cfg.num_quantiles + qi];
                assert!((got - want_v).abs() < 1e-4, "target {ti} step {t} level {lv}: got {got} want {want_v}");
            }
        }
    }
}
