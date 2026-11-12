// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rolling-origin backtester — the honest, cost-agnostic evaluation over a real
//! panel. Walks the origin backward from the end of the series, forecasting and
//! scoring at each step with the resident models, and aggregates per
//! `(model, metric)` to a mean over origins with its standard error.
//!
//! Origins closer together than the horizon score overlapping futures, so
//! their errors are correlated and the origin-to-origin spread understates the
//! uncertainty of the mean. The standard error is therefore Newey-West (a
//! Bartlett-weighted sum of autocovariances) with the lag set to the number of
//! origins one future overlaps: `(horizon - 1) / stride`. Each non-naive row
//! also carries its difference from the `naive` model over the origins both
//! scored, with that difference's own standard error: the same forecast
//! origins are hard for both models, so the paired interval is the one that
//! answers "does this model beat naive".
//!
//! The MASE scale uses the seasonal period of the panel's frequency
//! ([`Panel::seasonal_period`]) and the probabilistic metrics use the spec's
//! `quantile_levels`.
//!
//! Univariate for P0: it backtests the first target of the first item. Panels
//! with covariates or multiple items are handled once a covariate-aware model
//! (Chronos-2) lands.

use crate::score;
use crate::util::{mean_and_se, paired_diff};
use forecast::{BacktestReport, BacktestRow, BacktestSpec, ForecastModel, PairedDiff, Panel};
use std::collections::BTreeMap;

/// Run a rolling-origin backtest of `models` over `panel` per `spec`.
///
/// `models` is `(name, model)` — the caller resolves the spec's model names to
/// resident instances. Origins are taken from the end of the series backward,
/// `spec.stride` apart, each with `spec.horizon` held out as the actual future.
pub fn run(
    models: &[(String, &dyn ForecastModel)],
    panel: &Panel,
    spec: &BacktestSpec,
) -> BacktestReport {
    let mut report = BacktestReport::default();

    // locate the series to backtest: first target of first item.
    let Some(item) = panel.items.first() else { return report };
    let Some(target) = item.targets().next() else { return report };
    let series_values = &target.data;
    let n = series_values.len();
    let h = spec.horizon.max(1);
    let stride = spec.stride.max(1);
    let season = panel.seasonal_period();
    // overlapping futures: origins `k * stride` apart share steps while k * stride < h
    let lag = (h - 1) / stride;

    if n < h + 2 {
        return report;
    }

    // origins: the split point o means context = series[..o], future = series[o..o+h].
    // The latest usable origin is n-h; step backward by stride, up to spec.origins.
    let mut origins: Vec<usize> = Vec::new();
    let mut o = n - h;
    for _ in 0..spec.origins {
        if o < 2 {
            break;
        }
        origins.push(o);
        if o < stride {
            break;
        }
        o -= stride;
    }

    // per (model, metric): one score per origin, None where the model produced none
    let mut series: BTreeMap<(String, String), Vec<Option<f32>>> = BTreeMap::new();

    for (mi, &o) in origins.iter().enumerate() {
        let context = &series_values[..o];
        let future = &series_values[o..o + h];
        for (name, model) in models {
            let scores = score::score_split(
                *model,
                &item.item_id,
                &target.name,
                &panel.freq,
                context,
                future,
                season,
                &spec.quantile_levels,
                &spec.metrics,
                spec.seed.wrapping_add(mi as u64),
            );
            for (metric, v) in scores {
                if v.is_finite() {
                    series.entry((name.clone(), metric)).or_insert_with(|| vec![None; origins.len()])[mi] = Some(v);
                }
            }
        }
    }

    for ((model, metric), scores) in &series {
        let kept: Vec<f32> = scores.iter().flatten().copied().collect();
        if kept.is_empty() {
            continue;
        }
        let (value, std_err) = mean_and_se(&kept, lag);
        let vs_naive = match series.get(&("naive".to_string(), metric.clone())) {
            Some(naive) if model != "naive" => {
                paired_diff(scores, naive, lag).map(|d| PairedDiff { mean: d.mean, std_err: d.std_err })
            }
            _ => None,
        };
        report.rows.push(BacktestRow {
            model: model.clone(),
            metric: metric.clone(),
            value,
            n_origins: kept.len(),
            std_err,
            vs_naive,
        });
    }
    report.rows.sort_by(|a, b| (a.model.as_str(), a.metric.as_str()).cmp(&(&b.model, &b.metric)));
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::{Drift, RandomWalk};
    use crate::rng::Rng;
    use forecast::Variate;

    fn random_walk_series(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        let mut x = 100.0f32;
        (0..n)
            .map(|_| {
                x += r.normal();
                x
            })
            .collect()
    }

    #[test]
    fn backtest_aggregates_over_origins() {
        let series = random_walk_series(300, 5);
        let panel = Panel::single("1d", "SIM", vec![Variate::target("close", series)]);
        let spec = BacktestSpec {
            models: vec!["naive".into(), "drift".into()],
            horizon: 5,
            origins: 20,
            stride: 3,
            metrics: score::default_metrics(),
            quantile_levels: vec![0.1, 0.5, 0.9],
            ..Default::default()
        };
        let rw = RandomWalk;
        let dr = Drift;
        let models: Vec<(String, &dyn ForecastModel)> =
            vec![("naive".into(), &rw), ("drift".into(), &dr)];
        let report = run(&models, &panel, &spec);
        // every (model, metric) present and averaged over multiple origins
        for m in ["naive", "drift"] {
            for metric in ["mase", "wql", "coverage", "directional"] {
                let row = report.rows.iter().find(|r| r.model == m && r.metric == metric);
                assert!(row.is_some(), "missing {m}/{metric}");
                assert!(row.unwrap().n_origins > 1, "should average multiple origins");
            }
        }
    }

    #[test]
    fn mase_is_scaled_by_the_panel_frequencys_seasonal_period() {
        // a weekly pattern with a little noise: one step apart the series moves a
        // lot, a week apart it barely does. Daily data scales by the week, so
        // naive's error reads much larger than under an unrecognised frequency
        // (lag-1 scale).
        let mut r = Rng::new(3);
        let series: Vec<f32> = (0..200).map(|i| [0.0, 3.0, 6.0, 4.0, 1.0, -2.0, -4.0][i % 7] + 0.05 * r.normal()).collect();
        let spec = BacktestSpec { models: vec!["naive".into()], horizon: 3, origins: 30, stride: 1, metrics: vec!["mase".into()], ..Default::default() };
        let rw = RandomWalk;
        let models: Vec<(String, &dyn ForecastModel)> = vec![("naive".into(), &rw)];
        let mase = |freq: &str| {
            let panel = Panel::single(freq, "S", vec![Variate::target("y", series.clone())]);
            run(&models, &panel, &spec).get("naive", "mase").unwrap()
        };
        assert!(mase("1d") > 3.0 * mase("fortnightly"), "daily {} vs unscaled {}", mase("1d"), mase("fortnightly"));
    }

    #[test]
    fn the_backtest_scores_at_the_specs_quantile_levels() {
        let panel = Panel::single("1d", "SIM", vec![Variate::target("close", random_walk_series(300, 8))]);
        let coverage = |levels: Vec<f32>| {
            let spec = BacktestSpec {
                models: vec!["naive".into()],
                horizon: 5,
                origins: 60,
                stride: 2,
                metrics: vec!["coverage".into()],
                quantile_levels: levels,
                ..Default::default()
            };
            let rw = RandomWalk;
            let models: Vec<(String, &dyn ForecastModel)> = vec![("naive".into(), &rw)];
            run(&models, &panel, &spec).get("naive", "coverage")
        };
        let narrow = coverage(vec![0.4, 0.5, 0.6]).unwrap();
        let wide = coverage(vec![0.05, 0.5, 0.95]).unwrap();
        assert!(narrow < wide, "narrow {narrow} wide {wide}");
        assert!(coverage(vec![]).is_none(), "no levels, no coverage row");
    }

    #[test]
    fn rows_carry_a_standard_error_and_a_paired_difference_against_naive() {
        let panel = Panel::single("1d", "SIM", vec![Variate::target("close", random_walk_series(300, 5))]);
        let spec = BacktestSpec {
            models: vec!["naive".into(), "drift".into()],
            horizon: 5,
            origins: 40,
            stride: 1,
            metrics: vec!["mase".into()],
            ..Default::default()
        };
        let (rw, dr) = (RandomWalk, Drift);
        let models: Vec<(String, &dyn ForecastModel)> = vec![("naive".into(), &rw), ("drift".into(), &dr)];
        let report = run(&models, &panel, &spec);
        let row = |m: &str| report.rows.iter().find(|r| r.model == m).unwrap();
        let (naive, drift) = (row("naive"), row("drift"));
        assert!(naive.std_err > 0.0 && naive.vs_naive.is_none());
        let d = drift.vs_naive.expect("drift is paired against naive");
        assert!((d.mean - (drift.value - naive.value)).abs() < 1e-4);
        assert!(d.std_err > 0.0);
    }

    #[test]
    fn naive_is_not_beaten_on_a_random_walk_backtest() {
        // The negative control, now on the backtester: over many origins of a
        // real driftless random walk, naive's MASE must not be materially beaten.
        let series = random_walk_series(400, 11);
        let panel = Panel::single("1d", "SIM", vec![Variate::target("close", series)]);
        let spec = BacktestSpec {
            models: vec!["naive".into(), "drift".into()],
            horizon: 5,
            origins: 40,
            stride: 2,
            metrics: vec!["mase".into()],
            quantile_levels: vec![0.1, 0.5, 0.9],
            ..Default::default()
        };
        let rw = RandomWalk;
        let dr = Drift;
        let models: Vec<(String, &dyn ForecastModel)> =
            vec![("naive".into(), &rw), ("drift".into(), &dr)];
        let report = run(&models, &panel, &spec);
        assert!(
            !report.beats("drift", "naive", "mase"),
            "drift should not beat naive on a random walk"
        );
    }

    #[test]
    fn empty_or_too_short_panel_yields_empty_report() {
        let panel = Panel::single("1d", "X", vec![Variate::target("y", vec![1.0, 2.0])]);
        let spec = BacktestSpec { horizon: 5, ..Default::default() };
        let rw = RandomWalk;
        let models: Vec<(String, &dyn ForecastModel)> = vec![("naive".into(), &rw)];
        assert!(run(&models, &panel, &spec).rows.is_empty());
    }
}
