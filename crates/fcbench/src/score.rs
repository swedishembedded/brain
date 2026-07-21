// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared scoring: run one model on one `(context -> future)` split and compute
//! the forecasting metrics. Used by both the scenario harness and the
//! rolling-origin backtester so scoring is defined once.

use forecast::{metrics, ForecastModel, ForecastSpec, Panel, Representation, Variate};
use std::collections::BTreeMap;

/// The quantile grid used for probabilistic scoring (10/50/90 → 80% interval).
pub const LEVELS: [f32; 3] = [0.1, 0.5, 0.9];

/// Number of samples requested for CRPS.
pub const CRPS_SAMPLES: usize = 200;

/// Forecast `context` with `model` and score against `future`. Returns a metric
/// map; an empty map if the model errors or emits nothing usable. `metrics_want`
/// selects which metrics to compute (unknown names are ignored).
pub fn score_split(
    model: &dyn ForecastModel,
    item_id: &str,
    target_name: &str,
    freq: &str,
    context: &[f32],
    future: &[f32],
    season: usize,
    metrics_want: &[String],
    seed: u64,
) -> BTreeMap<String, f32> {
    let want = |m: &str| metrics_want.iter().any(|w| w == m);
    let need_samples = want("crps");
    let spec = ForecastSpec {
        horizon: future.len(),
        representations: {
            let mut r = vec![Representation::Quantiles, Representation::Point];
            if need_samples {
                r.push(Representation::Samples);
            }
            r
        },
        quantile_levels: LEVELS.to_vec(),
        num_samples: if need_samples { CRPS_SAMPLES } else { 0 },
        seed,
    };
    let panel = Panel::single(freq, item_id, vec![Variate::target(target_name, context.to_vec())]);
    let mut out = BTreeMap::new();
    let fc = match model.forecast(&panel, &spec) {
        Ok(f) => f,
        Err(_) => return out,
    };
    let Some(tf) = fc.targets.iter().find(|t| t.name == target_name).or_else(|| fc.targets.first())
    else {
        return out;
    };
    let origin = *context.last().unwrap_or(&0.0);
    let h = future.len();
    let ql = LEVELS.len();

    if let Some(mean) = &tf.mean {
        if want("mase") {
            out.insert("mase".into(), metrics::mase(&mean.data, future, context, season));
        }
        if want("directional") {
            out.insert(
                "directional".into(),
                metrics::directional_accuracy(&mean.data, future, origin),
            );
        }
    }
    if let Some(q) = &tf.quantiles {
        if want("wql") {
            out.insert("wql".into(), metrics::weighted_quantile_loss(&q.data, &LEVELS, future));
        }
        if want("coverage") && ql >= 2 {
            let lo: Vec<f32> = (0..h).map(|t| q.data[t * ql]).collect();
            let hi: Vec<f32> = (0..h).map(|t| q.data[t * ql + ql - 1]).collect();
            out.insert("coverage".into(), metrics::coverage(&lo, &hi, future));
        }
    }
    if want("crps") {
        if let Some(s) = &tf.samples {
            // samples are [n_samples, horizon]; CRPS per step, averaged
            let (n, hh) = (s.shape[0], s.shape[1]);
            let mut acc = 0.0f32;
            let mut col = vec![0.0f32; n];
            for t in 0..hh.min(h) {
                for i in 0..n {
                    col[i] = s.data[i * hh + t];
                }
                acc += metrics::crps_ensemble(&col, future[t]);
            }
            out.insert("crps".into(), acc / hh.max(1) as f32);
        }
    }
    out
}

/// The default metric set for comparisons.
pub fn default_metrics() -> Vec<String> {
    ["mase", "wql", "coverage", "directional"].iter().map(|s| s.to_string()).collect()
}
