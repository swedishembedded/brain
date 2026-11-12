// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scoring a forecaster against data it did not see: an OHLCV series in, the
//! last `horizon` bars held out as truth, the model's forecast judged against
//! them next to three baselines that cost nothing.
//!
//! That is what makes a printed number evidence: the model is judged against a
//! continuation it did not see, on one axis with persistence, drift and the
//! seasonal-naive forecast. Any [`forecast::ForecastModel`] runs it - the model
//! enters as a closure, so a model with request-level knobs of its own (Kronos
//! sampling) binds them without this module knowing.
//!
//! Swedish Embedded AB implements forecast evaluation that judges a model
//! against honest baselines on held-out data for its clients. If your team
//! needs expertise in probabilistic forecast scoring then you can procure our
//! services by sending an email to info@swedishembedded.com.

use forecast::csv::{self, OhlcvSeries, Split};
use forecast::{Forecast, ForecastError, ForecastSpec, Panel, Representation, TargetForecast};

/// One forecaster's score on one held-out window.
///
/// Three numbers rather than one, because a point error alone cannot judge a
/// probabilistic forecast of a near-random-walk: nothing beats persistence on
/// the LEVEL by much, and a model whose value is a calibrated interval would
/// look like a failure on MAE alone.
///
/// - `mae` - the point error of the median path.
/// - `crps` - the Continuous Ranked Probability Score of the whole predicted
///   distribution. For a deterministic forecast it collapses exactly to MAE,
///   which is what makes the model row and the baseline rows comparable.
/// - `pinball` - mean quantile loss over the emitted levels, the same judgement
///   read off the quantiles instead of the sample cloud.
#[derive(Clone, Copy, Debug)]
pub struct Score {
    pub mae: f32,
    pub crps: f32,
    pub pinball: f32,
}

impl Score {
    /// Score a distributional forecast: CRPS from the sample ensemble when there
    /// is one, else from the quantile grid, and pinball from the quantile grid.
    fn probabilistic(pred: &[f32], tf: &TargetForecast, actual: &[f32]) -> Score {
        let h = actual.len();
        let crps = match &tf.samples {
            // samples are [n_samples, horizon], row-major.
            Some(s) if s.shape.len() == 2 && s.data.len() == s.shape[0] * s.shape[1] && s.shape[1] >= h => {
                let (n, hh) = (s.shape[0], s.shape[1]);
                let mut acc = 0.0;
                for (t, &y) in actual.iter().enumerate() {
                    let col: Vec<f32> = (0..n).map(|k| s.data[k * hh + t]).collect();
                    acc += forecast::metrics::crps_ensemble(&col, y);
                }
                acc / h as f32
            }
            // No ensemble but a quantile grid (a quantile-native model): CRPS
            // read off the grid.
            _ if tf.quantiles.as_ref().is_some_and(|q| !tf.levels.is_empty() && q.data.len() >= h * tf.levels.len()) => {
                let q = tf.quantiles.as_ref().expect("checked above");
                forecast::metrics::crps_quantile_grid(&q.data[..h * tf.levels.len()], &tf.levels, actual)
            }
            // Neither: the point path IS the distribution, and its CRPS is its
            // MAE.
            _ => forecast::metrics::mae(pred, actual),
        };
        let pinball = match &tf.quantiles {
            Some(q) if !tf.levels.is_empty() && q.data.len() >= h * tf.levels.len() => forecast::metrics::mean_pinball(&q.data[..h * tf.levels.len()], &tf.levels, actual),
            _ => forecast::metrics::mean_pinball(&repeat_levels(pred, 1), &[0.5], actual),
        };
        Score { mae: forecast::metrics::mae(pred, actual), crps, pinball }
    }

    /// Score a point forecast on the same three axes. A deterministic forecast
    /// is a degenerate distribution: its CRPS is its MAE, and its quantile grid
    /// is the same value at every level.
    fn point(pred: &[f32], actual: &[f32]) -> Score {
        let mae = forecast::metrics::mae(pred, actual);
        let levels = [0.1f32, 0.5, 0.9];
        Score { mae, crps: mae, pinball: forecast::metrics::mean_pinball(&repeat_levels(pred, levels.len()), &levels, actual) }
    }
}

/// A `[H]` point path widened into the `[H, Q]` grid `mean_pinball` wants, with
/// the same value at every level.
fn repeat_levels(pred: &[f32], q: usize) -> Vec<f32> {
    pred.iter().flat_map(|v| std::iter::repeat_n(*v, q)).collect()
}

/// The mean of one field over the per-origin scores.
fn mean_of(scores: &[Score], f: impl Fn(&Score) -> f32) -> f32 {
    scores.iter().map(f).sum::<f32>() / scores.len().max(1) as f32
}

/// Empirical coverage of the outermost predicted interval: the fraction of
/// held-out bars that landed inside it. Read against the nominal level (80% for
/// a 10-90% band) it says whether the model's uncertainty is honest - the one
/// question a point metric cannot answer. `None` when the model emitted no
/// usable band.
fn band_coverage(tf: &TargetForecast, actual: &[f32]) -> Option<f32> {
    let q = tf.quantiles.as_ref()?;
    let n = tf.levels.len();
    if n < 2 || q.data.len() < actual.len() * n {
        return None;
    }
    let lo: Vec<f32> = (0..actual.len()).map(|h| q.data[h * n]).collect();
    let hi: Vec<f32> = (0..actual.len()).map(|h| q.data[h * n + n - 1]).collect();
    Some(forecast::metrics::coverage(&lo, &hi, actual))
}

/// "Carry the context's own mean log return forward" - the drift baseline, and
/// the right one to beat on a series with a trend: persistence forecasts no
/// move at all, which is a different (and weaker) claim.
pub fn drift_path(ctx: &[f32], horizon: usize) -> Vec<f32> {
    let last = *ctx.last().unwrap_or(&0.0);
    if ctx.len() < 2 || last <= 0.0 || ctx.iter().any(|v| *v <= 0.0) {
        return vec![last; horizon];
    }
    let mu = (last.ln() - ctx[0].ln()) / (ctx.len() - 1) as f32;
    (1..=horizon).map(|h| last * (mu * h as f32).exp()).collect()
}

/// The forecast's point path over the horizon: the median quantile when the
/// model produced quantiles (robust for a sampled model), else the mean.
pub fn point_path(tf: &TargetForecast, horizon: usize) -> Option<Vec<f32>> {
    if let (Some(q), Some(mid)) = (&tf.quantiles, tf.levels.iter().position(|l| (*l - 0.5).abs() < 1e-6)) {
        let n = tf.levels.len();
        if q.data.len() >= horizon * n {
            return Some((0..horizon).map(|h| q.data[h * n + mid]).collect());
        }
    }
    tf.mean.as_ref().filter(|m| m.data.len() >= horizon).map(|m| m.data[..horizon].to_vec())
}

/// "The same bar one season ago", extended by repeating the last full season
/// when the horizon runs past it. `None` when the context is shorter than one
/// season, so the baseline is omitted rather than silently degenerating into
/// persistence.
pub fn seasonal_naive(ctx: &[f32], horizon: usize, season: usize) -> Option<Vec<f32>> {
    if season == 0 || ctx.len() < season {
        return None;
    }
    Some((0..horizon).map(|h| ctx[ctx.len() - season + (h % season)]).collect())
}

/// What one held-out evaluation asks for.
#[derive(Clone, Debug)]
pub struct HoldoutRequest<'a> {
    pub series: &'a OhlcvSeries,
    /// Bars held out as truth at each origin.
    pub horizon: usize,
    /// Bars of history the model sees.
    pub context: usize,
    /// Trajectories drawn; `1` asks for the deterministic path.
    pub samples: usize,
    /// Disjoint held-out windows to average over, the latest ending the file.
    pub origins: usize,
    pub seed: u64,
    /// The seasonal-naive baseline's period, bars.
    pub season: usize,
    /// The series' name on the chart and in the panel.
    pub item: &'a str,
    /// The series' sampling frequency, as the model's panel names it.
    pub freq: &'a str,
}

/// The latest origin's window and forecast: what a chart draws.
#[derive(Debug)]
pub struct Latest {
    pub split: Split,
    pub point: Vec<f32>,
    pub forecast: TargetForecast,
}

/// One scored row of the report: a forecaster and its score at every origin.
#[derive(Debug)]
pub struct Row {
    pub name: String,
    pub scores: Vec<Score>,
}

/// A finished evaluation.
#[derive(Debug)]
pub struct HoldoutReport {
    pub model: String,
    pub request_context: usize,
    pub horizon: usize,
    pub origins: usize,
    pub samples: usize,
    pub elapsed_secs: f64,
    /// The model first, then the baselines.
    pub rows: Vec<Row>,
    /// Mean coverage of the outermost predicted band; NaN when none was emitted.
    pub band_coverage: f32,
    /// Mean fraction of steps whose direction the point path called right.
    pub direction_hit: f32,
    pub latest: Latest,
}

impl HoldoutReport {
    /// The model's mean score over the origins.
    pub fn model_score(&self) -> Score {
        let s = &self.rows[0].scores;
        Score { mae: mean_of(s, |x| x.mae), crps: mean_of(s, |x| x.crps), pinball: mean_of(s, |x| x.pinball) }
    }

    /// The report as text: one line of what ran, a score table, the band's
    /// calibration and the model's CRPS against persistence's.
    pub fn render(&self) -> String {
        let n = self.origins;
        let mut out = format!(
            "{} forecast: {} bars of context -> {} held-out bars x {n} rolling origin{}  ({:.1}s, {} sample{})\n",
            self.model,
            self.request_context,
            self.horizon,
            if n == 1 { "" } else { "s" },
            self.elapsed_secs,
            self.samples,
            if self.samples == 1 { "" } else { "s" }
        );
        out.push_str(&format!("  {:<28} {:>10} {:>10} {:>10}\n", "close, vs held-out truth", "mean MAE", "CRPS", "pinball"));
        for row in &self.rows {
            out.push_str(&format!(
                "  {:<28} {:>10.4} {:>10.4} {:>10.4}\n",
                row.name,
                mean_of(&row.scores, |s| s.mae),
                mean_of(&row.scores, |s| s.crps),
                mean_of(&row.scores, |s| s.pinball)
            ));
        }
        // Calibration is the claim a band forecast is actually making, so it is
        // a number rather than left to the eye on the chart. A one-sample run is
        // the deterministic path: there is no band, so a coverage line would
        // read zero and mean nothing.
        if self.band_coverage.is_finite() && self.samples > 1 {
            out.push_str(&format!(
                "  p10-p90 band covers {:.0}% of held-out bars (nominal coverage is 80 percent); direction hit rate {:.0}%\n",
                self.band_coverage * 100.0,
                self.direction_hit * 100.0
            ));
        }
        if let (Some(k), Some(p)) = (self.rows.first(), self.rows.get(1)) {
            // CRPS, not MAE, is the headline: it is the proper score for the
            // distribution the model emits, and it reduces to MAE for the point
            // baselines - so the two rows are comparable on the same axis.
            let wins = k.scores.iter().zip(&p.scores).filter(|(a, b)| a.crps < b.crps).count();
            let (kc, pc) = (mean_of(&k.scores, |s| s.crps), mean_of(&p.scores, |s| s.crps));
            out.push_str(&format!("  vs persistence: {:+.1}% CRPS reduction, better at {wins}/{n} origins\n", (1.0 - kc / pc) * 100.0));
        }
        out
    }

    /// The evidence chart: the tail of the context, the forecast, the held-out
    /// actual, and (when the model produced quantiles) the 10-90% band under
    /// the forecast.
    pub fn chart(&self, item: &str) -> forecast::chart::ForecastChart {
        let (split, pred, tf, horizon) = (&self.latest.split, &self.latest.point, &self.latest.forecast, self.horizon);
        // Four horizons of history, and never less than two days of hourly bars:
        // enough to read the volatility regime the forecast is continuing (a
        // couple of dozen bars of a random walk read as a smooth trend, which is
        // exactly the wrong impression), without squeezing the part under
        // judgement into the right-hand margin.
        let show = (horizon * 4).max(48).min(split.context.len());
        let first = split.context.len() - show;
        let mut chart = forecast::chart::ForecastChart::new(format!("{item}: {} {horizon}-bar forecast vs held-out actual", self.model));
        chart.forecast_label = format!("{} forecast", self.model);
        chart.y_label = "close".to_string();
        chart.history = split.context[first..].iter().enumerate().map(|(i, b)| (i as f64, b.ohlcv[csv::CLOSE] as f64)).collect();
        let origin = show as f64 - 1.0;
        // The forecast and the actual both start at the context's last bar, so
        // the three lines meet at the origin rather than floating apart by one
        // step.
        let last = split.context[split.context.len() - 1].ohlcv[csv::CLOSE] as f64;
        chart.forecast = std::iter::once((origin, last)).chain(pred.iter().enumerate().map(|(h, v)| (origin + 1.0 + h as f64, *v as f64))).collect();
        chart.actual = std::iter::once((origin, last)).chain(split.actual.iter().enumerate().map(|(h, b)| (origin + 1.0 + h as f64, b.ohlcv[csv::CLOSE] as f64))).collect();
        if let Some(q) = &tf.quantiles {
            let n = tf.levels.len();
            if n >= 3 && q.data.len() >= horizon * n {
                chart.band = (0..horizon).map(|h| (origin + 1.0 + h as f64, q.data[h * n] as f64, q.data[h * n + n - 1] as f64)).collect();
            }
        }
        chart
    }
}

/// Run the held-out evaluation of `model` (named `name`; `forecast` is its
/// forecast call).
///
/// Origin 0 is the end of the file; each further origin steps `horizon` bars
/// earlier, so the held-out windows are disjoint and no bar is scored twice.
/// Metrics are averaged over all of them - one origin is a draw, not a
/// measurement. `on_origin` is told `(origins done, origins in all)` after
/// each, so a caller can report progress.
pub fn evaluate(
    name: &str,
    forecast: &mut dyn FnMut(&Panel, &ForecastSpec) -> Result<Forecast, ForecastError>,
    req: &HoldoutRequest,
    on_origin: &mut dyn FnMut(usize, usize),
) -> Result<HoldoutReport, String> {
    let origins = req.origins.max(1);
    let (horizon, season) = (req.horizon, req.season);
    let spec = ForecastSpec {
        horizon,
        representations: vec![Representation::Quantiles, Representation::Point],
        quantile_levels: vec![0.1, 0.5, 0.9],
        num_samples: req.samples.max(1),
        seed: req.seed,
    };
    let mut rows: Vec<Row> = Vec::new();
    let mut calib: Vec<(f32, f32)> = Vec::new(); // per-origin (band coverage, direction hit rate)
    let mut latest: Option<Latest> = None;
    let t0 = std::time::Instant::now();
    for o in 0..origins {
        let split = req.series.split_at_origin(req.context, horizon, o * horizon)?;
        let panel = csv::panel(&split, req.item, req.freq);
        let out = forecast(&panel, &spec).map_err(|e| format!("forecast failed: {} ({})", e.message, e.code))?;
        let tf = out.targets.into_iter().find(|t| t.name == "close").ok_or("the model returned no `close` target")?;
        let pred = point_path(&tf, horizon).ok_or("the model returned no usable point path")?;

        // The baselines. Persistence is the random-walk answer; drift
        // extrapolates the context's own mean log return; seasonal naive is the
        // same bar one period ago. All three are read from the CONTEXT only, so
        // they stay leak-free.
        let ctx_close: Vec<f32> = split.context.iter().map(|b| b.ohlcv[csv::CLOSE]).collect();
        let actual: Vec<f32> = split.actual.iter().map(|b| b.ohlcv[csv::CLOSE]).collect();
        let last = *ctx_close.last().expect("split guarantees a non-empty context");
        let mut row: Vec<(String, Score)> = vec![
            (name.to_string(), Score::probabilistic(&pred, &tf, &actual)),
            ("persistence (last close)".into(), Score::point(&vec![last; horizon], &actual)),
            ("drift (context mean return)".into(), Score::point(&drift_path(&ctx_close, horizon), &actual)),
        ];
        if let Some(s) = seasonal_naive(&ctx_close, horizon, season) {
            row.push((format!("seasonal naive ({season} bars)"), Score::point(&s, &actual)));
        }
        for (i, (row_name, s)) in row.into_iter().enumerate() {
            if rows.len() <= i {
                rows.push(Row { name: row_name, scores: Vec::new() });
            }
            rows[i].scores.push(s);
        }
        calib.push((band_coverage(&tf, &actual).unwrap_or(f32::NAN), forecast::metrics::directional_accuracy(&pred, &actual, last)));
        // Only the most recent origin is charted: it is the one whose held-out
        // window a reader can line up against the end of the input file.
        if o == 0 {
            latest = Some(Latest { split, point: pred, forecast: tf });
        }
        on_origin(o + 1, origins);
    }
    Ok(HoldoutReport {
        model: name.to_string(),
        request_context: req.context,
        horizon,
        origins,
        samples: spec.num_samples,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        rows,
        band_coverage: calib.iter().map(|c| c.0).sum::<f32>() / calib.len() as f32,
        direction_hit: calib.iter().map(|c| c.1).sum::<f32>() / calib.len() as f32,
        latest: latest.expect("at least one origin ran"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use forecast::{Block, Forecast};

    /// A daily series of `n` bars with a gentle trend and a weekly wiggle.
    fn series(n: usize) -> OhlcvSeries {
        let mut text = String::from("Date,open,high,low,close,volume\n");
        for i in 0..n {
            let (month, day) = (1 + i / 28, 1 + i % 28);
            let c = 100.0 + i as f32 * 0.5 + ((i % 7) as f32 - 3.0);
            text.push_str(&format!("2024-{month:02}-{day:02},{c:.2},{:.2},{:.2},{c:.2},1000\n", c + 1.0, c - 1.0));
        }
        csv::parse_ohlcv(&text).expect("a valid series")
    }

    /// A forecaster that repeats the last close at every level: persistence
    /// itself, so the evaluation must score it level with the persistence row.
    fn persistence(panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        let close = panel.items[0].variate("close").expect("close").data.clone();
        let last = *close.last().unwrap();
        let mut fc = Forecast::new("persistence", Representation::Quantiles, spec.horizon, &panel.freq);
        let mut tf = TargetForecast::new(&panel.items[0].item_id, "close");
        let levels = vec![0.1f32, 0.5, 0.9];
        tf.levels = levels.clone();
        tf.quantiles = Some(Block::native(vec![spec.horizon, 3], vec![last; spec.horizon * 3]));
        fc.targets.push(tf);
        Ok(fc)
    }

    fn request(s: &OhlcvSeries, origins: usize) -> HoldoutRequest<'_> {
        HoldoutRequest { series: s, horizon: 6, context: 40, samples: 1, origins, seed: 7, season: 7, item: "x", freq: "1d" }
    }

    #[test]
    fn a_persistence_model_scores_level_with_the_persistence_baseline() {
        let s = series(120);
        let r = evaluate("persist", &mut persistence, &request(&s, 3), &mut |_, _| {}).unwrap();
        assert_eq!(r.rows.len(), 4, "the model, persistence, drift and seasonal naive");
        assert_eq!(r.rows[0].scores.len(), 3, "one score per origin");
        for (m, p) in r.rows[0].scores.iter().zip(&r.rows[1].scores) {
            assert!((m.mae - p.mae).abs() < 1e-5, "model {} vs persistence {}", m.mae, p.mae);
        }
        let text = r.render();
        assert!(text.contains("persist forecast: 40 bars of context -> 6 held-out bars x 3 rolling origins"), "{text}");
        assert!(text.contains("vs persistence: +0.0% CRPS reduction, better at 0/3 origins"), "{text}");
    }

    #[test]
    fn origins_step_back_by_whole_horizons_and_each_reports_progress() {
        let s = series(120);
        let mut seen = Vec::new();
        evaluate("persist", &mut persistence, &request(&s, 2), &mut |done, all| seen.push((done, all))).unwrap();
        assert_eq!(seen, [(1, 2), (2, 2)]);
    }

    #[test]
    fn a_series_too_short_for_the_origins_is_refused_with_the_csv_message() {
        let s = series(50);
        let e = evaluate("persist", &mut persistence, &request(&s, 3), &mut |_, _| {}).unwrap_err();
        assert!(e.contains("too few"), "{e}");
    }

    #[test]
    fn the_baselines_are_read_from_the_context_only() {
        // Drift carries the mean log return; seasonal naive repeats the last season.
        let ctx = [100.0f32, 110.0, 121.0];
        let d = drift_path(&ctx, 2);
        assert!((d[0] - 133.1).abs() < 0.05 && (d[1] - 146.41).abs() < 0.05, "{d:?}");
        assert_eq!(seasonal_naive(&[1.0, 2.0, 3.0], 5, 2), Some(vec![2.0, 3.0, 2.0, 3.0, 2.0]));
        assert_eq!(seasonal_naive(&[1.0], 3, 2), None, "shorter than a season: no baseline, not a silent persistence");
    }

    #[test]
    fn the_chart_meets_the_forecast_and_the_actual_at_the_last_context_bar() {
        let s = series(120);
        let r = evaluate("persist", &mut persistence, &request(&s, 1), &mut |_, _| {}).unwrap();
        let c = r.chart("x");
        assert_eq!(c.forecast[0], c.actual[0]);
        assert_eq!(c.forecast.len(), 7);
        assert_eq!(c.band.len(), 6);
    }
}
