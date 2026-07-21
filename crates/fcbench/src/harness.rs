// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The comparison harness: run a set of [`ForecastModel`]s over a battery of
//! [`Scenario`]s, score each with [`forecast::metrics`], and aggregate into a
//! model × scenario × metric grid.
//!
//! This is the engine behind the "definition of done" comparison, and it owns
//! the honesty guarantee: on a negative-control scenario it detects any model
//! that materially beats the naive baseline and reports it as a violation.

use crate::scenarios::{Scenario, Window};
use crate::score;
use forecast::ForecastModel;

/// One aggregated score.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub scenario: String,
    pub model: String,
    pub metric: String,
    pub value: f32,
}

/// The full comparison grid plus the metric names computed.
#[derive(Clone, Debug, Default)]
pub struct Comparison {
    pub cells: Vec<Cell>,
    pub metrics: Vec<String>,
    /// Number of windows averaged per (model, scenario).
    pub windows: usize,
}

impl Comparison {
    pub fn get(&self, scenario: &str, model: &str, metric: &str) -> Option<f32> {
        self.cells
            .iter()
            .find(|c| c.scenario == scenario && c.model == model && c.metric == metric)
            .map(|c| c.value)
    }

    /// Names of models that materially beat the `naive` baseline on a
    /// negative-control scenario (by more than `margin`, e.g. 0.10 = 10% lower
    /// MASE). On a true control the optimal forecast *is* naive, so a non-empty
    /// result signals false skill / a bug — the suite must fail on it.
    pub fn negative_control_violations(
        &self,
        control_scenario: &str,
        margin: f32,
    ) -> Vec<String> {
        let Some(naive) = self.get(control_scenario, "naive", "mase") else {
            return Vec::new();
        };
        let mut bad = Vec::new();
        for c in &self.cells {
            if c.scenario == control_scenario
                && c.metric == "mase"
                && c.model != "naive"
                && c.value < naive * (1.0 - margin)
            {
                bad.push(c.model.clone());
            }
        }
        bad.sort();
        bad.dedup();
        bad
    }
}

/// Score one model on one window via the shared scorer.
fn score_window(
    model: &dyn ForecastModel,
    win: &Window,
    season: usize,
    metrics_want: &[String],
) -> std::collections::BTreeMap<String, f32> {
    score::score_split(
        model,
        "sim",
        "y",
        "1d",
        &win.context,
        &win.future,
        season,
        metrics_want,
        0,
    )
}

/// Run the full comparison: `models` × `scenarios`, `windows` seeded windows per
/// pair, aggregated by mean.
pub fn run(
    models: &[Box<dyn ForecastModel>],
    scenarios: &[Box<dyn Scenario>],
    windows: usize,
    base_seed: u64,
) -> Comparison {
    let metric_names = score::default_metrics();
    let mut cells = Vec::new();

    for sc in scenarios {
        for model in models {
            let name = model.capabilities().name;
            let mut sums: std::collections::BTreeMap<String, (f32, usize)> =
                std::collections::BTreeMap::new();
            for w in 0..windows {
                let win = sc.generate(base_seed.wrapping_add(w as u64).wrapping_mul(2_654_435_761));
                for (m, v) in score_window(model.as_ref(), &win, sc.season(), &metric_names) {
                    if v.is_finite() {
                        let e = sums.entry(m).or_insert((0.0, 0));
                        e.0 += v;
                        e.1 += 1;
                    }
                }
            }
            for (metric, (sum, n)) in sums {
                if n > 0 {
                    cells.push(Cell {
                        scenario: sc.name().to_string(),
                        model: name.clone(),
                        metric,
                        value: sum / n as f32,
                    });
                }
            }
        }
    }

    Comparison { cells, metrics: metric_names, windows }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::{Arima, Drift, Garch11, RandomWalk};
    use crate::scenarios::{Ar1, RandomWalkScenario, SeasonalTrend};

    fn baseline_set() -> Vec<Box<dyn ForecastModel>> {
        vec![
            Box::new(RandomWalk),
            Box::new(Drift),
            Box::new(Arima { p: 2, d: 1 }),
            Box::new(Garch11),
        ]
    }

    #[test]
    fn harness_scores_every_model_and_metric() {
        let models = baseline_set();
        let scenarios: Vec<Box<dyn Scenario>> = vec![Box::new(Ar1::default())];
        let cmp = run(&models, &scenarios, 8, 1);
        // 4 models x 4 metrics all present
        for m in ["naive", "drift", "arima", "garch"] {
            for metric in ["mase", "directional", "wql", "coverage"] {
                assert!(cmp.get("ar1", m, metric).is_some(), "missing {m}/{metric}");
            }
        }
    }

    #[test]
    fn negative_control_holds_no_model_beats_naive_on_random_walk() {
        // THE key honesty test: on a driftless random walk, nothing should beat
        // the naive baseline by a meaningful margin.
        let models = baseline_set();
        let scenarios: Vec<Box<dyn Scenario>> = vec![Box::new(RandomWalkScenario::default())];
        let cmp = run(&models, &scenarios, 40, 7);
        let violations = cmp.negative_control_violations("random_walk", 0.10);
        assert!(
            violations.is_empty(),
            "models falsely beat naive on a random walk (bug): {violations:?}"
        );
    }

    #[test]
    fn coverage_of_naive_is_near_nominal_on_a_random_walk() {
        // naive's random-walk interval is exactly calibrated in the limit; the
        // 10-90 interval (nominal 0.8) should cover roughly 0.8 of the truth.
        let models: Vec<Box<dyn ForecastModel>> = vec![Box::new(RandomWalk)];
        let scenarios: Vec<Box<dyn Scenario>> = vec![Box::new(RandomWalkScenario::default())];
        let cmp = run(&models, &scenarios, 60, 3);
        let cov = cmp.get("random_walk", "naive", "coverage").unwrap();
        assert!((cov - 0.8).abs() < 0.15, "coverage {cov} far from nominal 0.8");
    }

    #[test]
    fn a_structured_scenario_lets_a_real_model_beat_naive() {
        // The positive counterpart: on a strongly seasonal+trend series, a model
        // that extrapolates structure (drift picks up the trend) should beat
        // naive on MASE. This proves the harness can *detect* skill, so the
        // negative control isn't passing just because scoring is inert.
        let models: Vec<Box<dyn ForecastModel>> = vec![Box::new(RandomWalk), Box::new(Drift)];
        let scenarios: Vec<Box<dyn Scenario>> =
            vec![Box::new(SeasonalTrend { slope: 0.3, noise: 0.02, ..Default::default() })];
        let cmp = run(&models, &scenarios, 12, 11);
        let naive = cmp.get("seasonal_trend", "naive", "mase").unwrap();
        let drift = cmp.get("seasonal_trend", "drift", "mase").unwrap();
        assert!(drift < naive, "drift {drift} should beat naive {naive} on a trend");
    }
}
