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
    /// Metric names to compute: `"mase"`, `"wql"`, `"coverage"` (of the
    /// outermost levels' interval), `"directional"`, `"crps"`. A name the scorer
    /// does not know produces no row.
    pub metrics: Vec<String>,
    /// Quantile levels requested of the model and scored: wQL is the mean over
    /// them and `"coverage"` is the interval between the lowest and highest.
    /// Strictly increasing, inside (0, 1); an empty list scores nothing
    /// probabilistic.
    pub quantile_levels: Vec<f32>,
    /// Seed for any stochastic model. [`Default`] draws a random one.
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
            seed: seed::random_seed(),
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
    /// Standard error of [`BacktestRow::value`]. Origins closer together than
    /// the horizon share future steps, so their errors are correlated; the
    /// backtester uses the Newey-West (Bartlett kernel) estimate with the
    /// overlap as its lag. `0` for a single origin.
    pub std_err: f32,
    /// This model minus the `naive` model over the origins both scored, with
    /// its own standard error. `None` for `naive` itself or when no `naive`
    /// row exists for the metric.
    pub vs_naive: Option<PairedDiff>,
}

/// A paired difference of two models' per-origin scores, with the standard
/// error of its mean. Pairing cancels the part of the error that both models
/// share (a hard origin is hard for both), so it is far tighter than the two
/// rows' own standard errors combined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedDiff {
    /// Mean of `model - naive` over the paired origins (negative: model better,
    /// for the lower-is-better metrics).
    pub mean: f32,
    /// Standard error of `mean` (Newey-West at the same lag as the rows).
    pub std_err: f32,
}

impl PairedDiff {
    /// True when the model beats the baseline with `z` standard errors of
    /// confidence: the whole interval `mean +- z * std_err` lies below zero.
    pub fn is_significantly_better(&self, z: f32) -> bool {
        self.mean + z * self.std_err < 0.0
    }
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
                BacktestRow { model: "chronos2".into(), metric: "mase".into(), value: 0.9, n_origins: 30, std_err: 0.0, vs_naive: None },
                BacktestRow { model: "naive".into(), metric: "mase".into(), value: 1.0, n_origins: 30, std_err: 0.0, vs_naive: None },
            ],
        };
        assert_eq!(r.get("chronos2", "mase"), Some(0.9));
        assert!(r.beats("chronos2", "naive", "mase"));
        assert!(!r.beats("naive", "chronos2", "mase"));
        // missing metric -> not a win
        assert!(!r.beats("chronos2", "naive", "crps"));
    }

    #[test]
    fn a_paired_difference_must_clear_its_interval_to_count_as_better() {
        let d = PairedDiff { mean: -0.1, std_err: 0.04 };
        assert!(d.is_significantly_better(1.96)); // -0.1 + 0.078 < 0
        assert!(!PairedDiff { mean: -0.1, std_err: 0.06 }.is_significantly_better(1.96));
        assert!(!PairedDiff { mean: 0.05, std_err: 0.0 }.is_significantly_better(1.96));
    }
}
