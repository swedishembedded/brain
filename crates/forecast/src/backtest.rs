// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rolling-origin backtest request + report.
//!
//! Backtesting is first-class and server-side: a client sends one
//! [`BacktestSpec`] naming several models and metrics, and brain rolls the
//! origin forward over the panel, forecasting and scoring at each step with the
//! models held resident. The dominant real use case, and the honest one — a
//! single forecast means little without a walk-forward evaluation beside the
//! naive baseline.

/// A rolling-origin backtest request over one panel.
#[derive(Clone, Debug, PartialEq)]
pub struct BacktestSpec {
    /// Models to evaluate (registered names). The naive baseline should always
    /// be among them so results are read relative to it.
    pub models: Vec<String>,
    /// Forecast horizon at each origin.
    pub horizon: usize,
    /// Number of origins to roll through (from the end of the panel backwards).
    pub origins: usize,
    /// Step between successive origins.
    pub stride: usize,
    /// Metric names to compute (`"mase"`, `"crps"`, `"coverage_90"`, …).
    pub metrics: Vec<String>,
    /// Quantile levels used when a metric needs them.
    pub quantile_levels: Vec<f32>,
    /// Seed for any stochastic model.
    pub seed: u64,
}

impl Default for BacktestSpec {
    fn default() -> Self {
        BacktestSpec {
            models: Vec::new(),
            horizon: 1,
            origins: 30,
            stride: 1,
            metrics: vec!["mase".into(), "crps".into()],
            quantile_levels: vec![0.1, 0.5, 0.9],
            seed: 0,
        }
    }
}

/// One scored cell of a backtest: a metric value for a model, aggregated over
/// all origins.
#[derive(Clone, Debug, PartialEq)]
pub struct BacktestRow {
    /// Model name.
    pub model: String,
    /// Metric name.
    pub metric: String,
    /// Aggregate value (mean over origins).
    pub value: f32,
    /// Number of origins that contributed.
    pub n_origins: usize,
}

/// The aggregated result of a backtest — a long-format table of
/// `(model, metric) -> value` the renderer pivots into a comparison grid.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BacktestReport {
    /// One row per `(model, metric)`.
    pub rows: Vec<BacktestRow>,
}

impl BacktestReport {
    /// Look up an aggregated metric value for a model.
    pub fn get(&self, model: &str, metric: &str) -> Option<f32> {
        self.rows.iter().find(|r| r.model == model && r.metric == metric).map(|r| r.value)
    }

    /// True if `model` beats the `baseline` model on `metric` (lower is better).
    /// Returns `false` if either value is missing.
    pub fn beats(&self, model: &str, baseline: &str, metric: &str) -> bool {
        match (self.get(model, metric), self.get(baseline, metric)) {
            (Some(m), Some(b)) => m < b,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_lookup_and_beats() {
        let r = BacktestReport {
            rows: vec![
                BacktestRow { model: "chronos2".into(), metric: "mase".into(), value: 0.9, n_origins: 30 },
                BacktestRow { model: "naive".into(), metric: "mase".into(), value: 1.0, n_origins: 30 },
            ],
        };
        assert_eq!(r.get("chronos2", "mase"), Some(0.9));
        assert!(r.beats("chronos2", "naive", "mase"));
        assert!(!r.beats("naive", "chronos2", "mase"));
        // missing metric -> not a win
        assert!(!r.beats("chronos2", "naive", "crps"));
    }
}
