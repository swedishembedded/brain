// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rolling-origin backtester — the honest, cost-agnostic evaluation over a real
//! panel. Walks the origin backward from the end of the series, forecasting and
//! scoring at each step with the resident models, and aggregates per
//! `(model, metric)`.
//!
//! Univariate for P0: it backtests the first target of the first item. Panels
//! with covariates or multiple items are handled once a covariate-aware model
//! (Chronos-2) lands.

use crate::score;
use forecast::{BacktestReport, BacktestRow, BacktestSpec, ForecastModel, Panel};
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
    let series = &target.data;
    let n = series.len();
    let h = spec.horizon.max(1);
    let stride = spec.stride.max(1);
    let season = 1; // seasonal period is a panel-level concern; default naive scale

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

    // accumulate per (model, metric): (sum, count)
    let mut acc: BTreeMap<(String, String), (f32, usize)> = BTreeMap::new();

    for (mi, &o) in origins.iter().enumerate() {
        let context = &series[..o];
        let future = &series[o..o + h];
        for (name, model) in models {
            let scores = score::score_split(
                *model,
                &item.item_id,
                &target.name,
                &panel.freq,
                context,
                future,
                season,
                &spec.metrics,
                spec.seed.wrapping_add(mi as u64),
            );
            for (metric, v) in scores {
                if v.is_finite() {
                    let e = acc.entry((name.clone(), metric)).or_insert((0.0, 0));
                    e.0 += v;
                    e.1 += 1;
                }
            }
        }
    }

    for ((model, metric), (sum, count)) in acc {
        if count > 0 {
            report.rows.push(BacktestRow {
                model,
                metric,
                value: sum / count as f32,
                n_origins: count,
            });
        }
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
            seed: 0,
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
            seed: 0,
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
