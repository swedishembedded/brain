// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared scoring: run one model on one `(context -> future)` split and compute
//! the forecasting metrics. Used by both the scenario harness and the
//! rolling-origin backtester so scoring is defined once.

use forecast::{metrics, ForecastModel, ForecastSpec, Item, Panel, Representation, TargetForecast, Variate};
use std::collections::BTreeMap;

/// The quantile grid used for probabilistic scoring (10/50/90 → 80% interval).
pub const LEVELS: [f32; 3] = [0.1, 0.5, 0.9];

/// Number of samples requested for CRPS.
pub const CRPS_SAMPLES: usize = 200;

fn spec_for(horizon: usize, metrics_want: &[String], seed: u64) -> ForecastSpec {
    let need_samples = metrics_want.iter().any(|w| w == "crps");
    ForecastSpec {
        horizon,
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
    }
}

/// The metric computation shared by `score_split` and `score_windows` - given
/// one target's forecast plus the window it was scored against, compute every
/// requested metric. An empty map for a target with nothing usable is the
/// same "no score" convention both callers already relied on.
fn score_from_target(tf: &TargetForecast, context: &[f32], future: &[f32], season: usize, metrics_want: &[String]) -> BTreeMap<String, f32> {
    let want = |m: &str| metrics_want.iter().any(|w| w == m);
    let origin = *context.last().unwrap_or(&0.0);
    let h = future.len();
    let ql = LEVELS.len();
    let mut out = BTreeMap::new();

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
            for (t, &actual) in future.iter().enumerate().take(hh.min(h)) {
                for (i, c) in col.iter_mut().enumerate() {
                    *c = s.data[i * hh + t];
                }
                acc += metrics::crps_ensemble(&col, actual);
            }
            out.insert("crps".into(), acc / hh.max(1) as f32);
        }
    }
    out
}

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
    let spec = spec_for(future.len(), metrics_want, seed);
    let panel = Panel::single(freq, item_id, vec![Variate::target(target_name, context.to_vec())]);
    let fc = match model.forecast(&panel, &spec) {
        Ok(f) => f,
        Err(_) => return BTreeMap::new(),
    };
    let Some(tf) = fc.targets.iter().find(|t| t.name == target_name).or_else(|| fc.targets.first()) else {
        return BTreeMap::new();
    };
    score_from_target(tf, context, future, season, metrics_want)
}

/// Batched sibling of `score_split`: scores several `(context, future)`
/// windows of ONE series against ONE model in a SINGLE `forecast()` call
/// (one `Item` per window) instead of one call per window, exploiting
/// whatever internal batching the model provides (e.g. `Timesfm3Forecaster`
/// groups same-shaped items into one device call). Every window shares
/// `horizon` (the rolling-origin backtester's own invariant - one horizon per
/// run) and `seed` (unlike `score_split`, which the caller re-seeds per
/// origin; a caller that needs distinct per-window seeds for CRPS sampling
/// should call `score_split` in a loop instead, same as `backtest::run` does
/// today). Returns one metric map per window, in the same order, matching
/// `score_split` called once per window bit-for-bit - including the "empty
/// map" convention for a window whose forecast is unusable; a whole-panel
/// forecast() error yields an empty map for every window, not a partial one,
/// since forecast() itself gives no per-item error to distinguish.
pub fn score_windows(
    model: &dyn ForecastModel,
    item_id: &str,
    target_name: &str,
    freq: &str,
    windows: &[(&[f32], &[f32])],
    season: usize,
    metrics_want: &[String],
    seed: u64,
) -> Vec<BTreeMap<String, f32>> {
    if windows.is_empty() {
        return Vec::new();
    }
    let horizon = windows[0].1.len();
    let spec = spec_for(horizon, metrics_want, seed);
    let items: Vec<Item> = windows
        .iter()
        .enumerate()
        .map(|(i, (context, _))| Item::new(format!("{item_id}#{i}"), vec![Variate::target(target_name, context.to_vec())]))
        .collect();
    let panel = Panel { freq: freq.to_string(), start: None, items };
    let fc = match model.forecast(&panel, &spec) {
        Ok(f) => f,
        Err(_) => return vec![BTreeMap::new(); windows.len()],
    };
    windows
        .iter()
        .enumerate()
        .map(|(i, (context, future))| {
            let id = format!("{item_id}#{i}");
            let Some(tf) = fc.targets.iter().find(|t| t.item_id == id && t.name == target_name) else {
                return BTreeMap::new();
            };
            score_from_target(tf, context, future, season, metrics_want)
        })
        .collect()
}

/// The default metric set for comparisons.
pub fn default_metrics() -> Vec<String> {
    ["mase", "wql", "coverage", "directional"].iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::RandomWalk;

    /// `score_windows` over N origins of one series must equal calling
    /// `score_split` once per origin, in the same order - the same invariant
    /// `timesfm3::Timesfm3Forecaster::forecast`'s own batching is gated on,
    /// checked here at the scoring-harness seam instead of the model seam.
    /// `RandomWalk` needs no checkpoint and already forecasts each panel item
    /// independently, so this is a fast, deterministic host-only test.
    #[test]
    fn score_windows_matches_score_split_called_per_window() {
        let model = RandomWalk;
        let metrics_want = default_metrics();
        let series: Vec<f32> = (0..40).map(|i| (i as f32 * 0.37).sin() * 5.0 + i as f32 * 0.1).collect();
        let h = 4;
        let origins = [20usize, 24, 28, 32];

        let windows: Vec<(&[f32], &[f32])> = origins.iter().map(|&o| (&series[..o], &series[o..o + h])).collect();
        let batched = score_windows(&model, "s", "t", "1d", &windows, 1, &metrics_want, 7);
        assert_eq!(batched.len(), origins.len());

        for (i, &o) in origins.iter().enumerate() {
            let want = score_split(&model, "s", "t", "1d", &series[..o], &series[o..o + h], 1, &metrics_want, 7);
            assert_eq!(batched[i], want, "origin {o}: score_windows must match score_split called individually");
        }
    }

    #[test]
    fn score_windows_of_an_empty_slice_is_empty() {
        let model = RandomWalk;
        let out: Vec<BTreeMap<String, f32>> = score_windows(&model, "s", "t", "1d", &[], 1, &default_metrics(), 0);
        assert!(out.is_empty());
    }
}
