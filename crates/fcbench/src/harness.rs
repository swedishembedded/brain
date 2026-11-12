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
use crate::util::{mean_and_se, paired_diff, Paired};
use forecast::ForecastModel;
use std::collections::BTreeMap;

/// Standard errors of confidence the negative control demands before it calls
/// a model's edge over naive real: the whole 99% interval of the paired
/// difference must lie on the model's side of zero.
pub const CONTROL_Z: f32 = 2.576;

/// One aggregated score.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub scenario: String,
    pub model: String,
    pub metric: String,
    pub value: f32,
    /// Windows the mean is over: fewer than [`Comparison::windows`] when the
    /// model errored on a window or its metric was not finite there.
    pub scored: usize,
    /// Standard error of [`Cell::value`] over the windows. The windows are
    /// drawn from independent seeds, so this is the plain `sigma / sqrt(n)`.
    pub std_err: f32,
    /// This model minus `naive` on the same windows, with its standard error
    /// (windows share a seed across models, so the pairing cancels the
    /// difficulty of each draw). `None` for `naive` itself.
    pub vs_naive: Option<Paired>,
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

    /// How many windows a cell's mean is over (see [`Cell::scored`]).
    pub fn scored(&self, scenario: &str, model: &str, metric: &str) -> Option<usize> {
        self.cells
            .iter()
            .find(|c| c.scenario == scenario && c.model == model && c.metric == metric)
            .map(|c| c.scored)
    }

    /// A cell's value for a report, with its window count when the mean is
    /// over fewer windows than the run had.
    pub fn display(&self, scenario: &str, model: &str, metric: &str, fmt: impl Fn(f32) -> String) -> Option<String> {
        let v = self.get(scenario, model, metric)?;
        let n = self.scored(scenario, model, metric).unwrap_or(self.windows);
        Some(if n < self.windows { format!("{} ({n}/{})", fmt(v), self.windows) } else { fmt(v) })
    }

    /// Names of models that beat the `naive` baseline on a negative-control
    /// scenario: those whose paired MASE difference against naive has its whole
    /// `z`-standard-error interval below zero. On a true control the optimal
    /// forecast *is* naive, so a non-empty result signals false skill / a bug
    /// the suite must fail on. A model with no paired difference (naive, or no
    /// shared window) is never flagged.
    pub fn negative_control_violations(&self, control_scenario: &str, z: f32) -> Vec<String> {
        let mut bad: Vec<String> = self
            .cells
            .iter()
            .filter(|c| c.scenario == control_scenario && c.metric == "mase" && c.model != "naive")
            .filter(|c| c.vs_naive.is_some_and(|d| d.mean + z * d.std_err < 0.0))
            .map(|c| c.model.clone())
            .collect();
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
        &score::DEFAULT_LEVELS,
        metrics_want,
        0,
    )
}

/// Run the full comparison: `models` × `scenarios`, `windows` seeded windows per
/// pair, aggregated by mean. Every model sees the same windows, so each cell
/// also carries its paired difference against `naive`.
pub fn run(
    models: &[Box<dyn ForecastModel>],
    scenarios: &[Box<dyn Scenario>],
    windows: usize,
    base_seed: u64,
) -> Comparison {
    let metric_names = score::default_metrics();
    let mut cells = Vec::new();

    for sc in scenarios {
        let wins: Vec<Window> = (0..windows)
            .map(|w| sc.generate(base_seed.wrapping_add(w as u64).wrapping_mul(2_654_435_761)))
            .collect();
        // model -> metric -> one score per window (None: errored or not finite)
        let mut series: BTreeMap<String, BTreeMap<String, Vec<Option<f32>>>> = BTreeMap::new();
        for model in models {
            let per_metric = series.entry(model.capabilities().name).or_default();
            for (w, win) in wins.iter().enumerate() {
                for (m, v) in score_window(model.as_ref(), win, sc.season(), &metric_names) {
                    if v.is_finite() {
                        per_metric.entry(m).or_insert_with(|| vec![None; windows])[w] = Some(v);
                    }
                }
            }
        }
        for (name, per_metric) in &series {
            for (metric, scores) in per_metric {
                let kept: Vec<f32> = scores.iter().flatten().copied().collect();
                if kept.is_empty() {
                    continue;
                }
                let (value, std_err) = mean_and_se(&kept, 0);
                let vs_naive = match series.get("naive").and_then(|n| n.get(metric)) {
                    Some(naive) if name != "naive" => paired_diff(scores, naive, 0),
                    _ => None,
                };
                cells.push(Cell {
                    scenario: sc.name().to_string(),
                    model: name.clone(),
                    metric: metric.clone(),
                    value,
                    scored: kept.len(),
                    std_err,
                    vs_naive,
                });
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
    use crate::util::Paired;

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

    fn control_cell(model: &str, value: f32, vs_naive: Option<Paired>) -> Cell {
        let std_err = 0.0;
        Cell { scenario: "random_walk".into(), model: model.into(), metric: "mase".into(), value, scored: 40, std_err, vs_naive }
    }

    /// The control is decided on the paired difference against naive and its
    /// interval, not on a fixed fraction of the mean.
    #[test]
    fn the_control_flags_a_difference_only_when_its_interval_clears_zero() {
        let paired = |mean, std_err| Some(Paired { mean, std_err, n: 40 });
        let cmp = Comparison {
            cells: vec![
                control_cell("naive", 1.0, None),
                // a 2% edge measured tightly is real skill on a random walk
                control_cell("tight_small_edge", 0.98, paired(-0.02, 0.002)),
                // a 50% edge measured loosely is noise
                control_cell("loose_big_edge", 0.5, paired(-0.5, 0.4)),
                control_cell("worse", 1.2, paired(0.2, 0.05)),
            ],
            metrics: vec!["mase".into()],
            windows: 40,
        };
        assert_eq!(cmp.negative_control_violations("random_walk", CONTROL_Z), vec!["tight_small_edge".to_string()]);
    }

    #[test]
    fn every_non_naive_cell_carries_a_paired_standard_error_against_naive() {
        let models = baseline_set();
        let scenarios: Vec<Box<dyn Scenario>> = vec![Box::new(RandomWalkScenario::default())];
        let cmp = run(&models, &scenarios, 20, 5);
        for c in cmp.cells.iter().filter(|c| c.metric == "mase") {
            assert!(c.std_err > 0.0, "{}: no standard error", c.model);
            match c.model.as_str() {
                "naive" => assert!(c.vs_naive.is_none()),
                m => {
                    let d = c.vs_naive.unwrap_or_else(|| panic!("{m}: no paired difference"));
                    let naive = cmp.get("random_walk", "naive", "mase").unwrap();
                    assert!((d.mean - (c.value - naive)).abs() < 1e-4, "{m}: paired mean is the difference of means");
                    // garch's point forecast is naive's, so its difference is exactly 0 with no spread
                    assert!(d.std_err > 0.0 || m == "garch", "{m}: no paired spread");
                }
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
        let violations = cmp.negative_control_violations("random_walk", CONTROL_Z);
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
