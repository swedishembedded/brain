// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `predict` action every forecasting model serves: an OHLCV CSV in, a
//! scored forecast out - the model against persistence, drift and seasonal
//! naive on bars it never saw, optionally with the evidence chart.
//!
//! The model is a closure, so this one implementation runs for every
//! forecaster and `brain serve`, `brain <model> predict` and the samples all
//! run it.
//!
//! Swedish Embedded AB implements forecast evaluation that judges a model
//! against honest baselines on held-out data for its clients. If your team
//! needs expertise in probabilistic forecast scoring then you can procure our
//! services by sending an email to info@swedishembedded.com.

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use forecast::{Forecast, ForecastError, ForecastSpec, Panel};
use serde_json::json;

use crate::holdout::{self, HoldoutRequest};

/// Held-out bars when the caller names none: two full daily cycles of an
/// hourly series, long enough that a forecast which only copies the last
/// value is visibly wrong.
const DEFAULT_HORIZON: i64 = 48;
/// The naive baseline's period when the caller names none: "the same hour
/// yesterday" on an hourly series, the sharpest cheap baseline there is on
/// anything with a daily cycle and therefore the one worth beating.
const DEFAULT_SEASON: i64 = 24;
/// Longest horizon, history, period or origin count a request may ask for.
const MAX_BARS: f64 = 100_000.0;
/// Most trajectories one forecast may draw.
const MAX_SAMPLES: f64 = 1_000.0;

/// The action's schema. Model-specific knobs (Kronos sampling) are appended by
/// the model that has them.
pub fn predict_spec(summary: &str) -> ActionSpec {
    ActionSpec::new("predict", summary)
        .param(ParamSpec::new("horizon", ParamType::Int, "bars held out as ground truth at each origin").min(1.0).max(MAX_BARS).default(json!(DEFAULT_HORIZON)))
        .param(ParamSpec::new("context", ParamType::Int, "bars of history the model sees; 0 = as many as the model's window allows").min(0.0).max(MAX_BARS).default(json!(0)))
        .param(ParamSpec::new("samples", ParamType::Int, "trajectories drawn: 1 is the deterministic path, more draw a cloud whose median is the point path and whose p10-p90 is the band").min(1.0).max(MAX_SAMPLES).default(json!(1)))
        .param(ParamSpec::new("origins", ParamType::Int, "disjoint held-out windows to average over, the latest ending the file: one origin is a draw, several are a measurement").min(1.0).max(MAX_BARS).default(json!(1)))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the sampled trajectories (omit for random; the outcome echoes it)").min(0.0))
        .param(ParamSpec::new("season", ParamType::Int, "period of the seasonal-naive baseline, bars").min(0.0).max(MAX_BARS).default(json!(DEFAULT_SEASON)))
        .param(ParamSpec::new("item", ParamType::Str, "the series' name on the chart").default(json!("series")))
        .param(ParamSpec::new("freq", ParamType::Str, "the series' sampling frequency as the model's panel names it").default(json!("1h")))
        .param(ParamSpec::new("chart", ParamType::Bool, "also render the chart of the latest origin (needs gnuplot on the machine that runs the model)").default(json!(false)))
        .input(BlobSpec::new("csv", Media::Text, "the OHLCV series: timestamp,open,high,low,close,volume, one validated bar per row").required())
        .output(BlobSpec::new("report", Media::Text, "the score table: the model against the baselines on the held-out bars"))
        .output(BlobSpec::new("chart", Media::Bytes, "the PNG chart of the latest origin; meta {format}"))
}

/// What `predict` needs to know about the model it runs.
pub struct PredictModel<'a> {
    /// The name every printed label and the chart use.
    pub name: &'a str,
    /// The longest history the model's window takes, bars.
    pub max_context: usize,
    /// The history used when the caller names none.
    pub default_context: usize,
    /// The most history after which the forecast is still exact; a longer
    /// request runs but is reported as an approximation. `None` when the model
    /// has no such limit.
    pub exact_context: Option<usize>,
    /// One forecast call.
    pub forecast: &'a mut dyn FnMut(&Panel, &ForecastSpec) -> Result<Forecast, ForecastError>,
}

fn int(inv: &Invocation, key: &str, default: i64) -> usize {
    inv.get_i64(key).unwrap_or(default).max(0) as usize
}

/// Run `predict` for `model` on the invocation's CSV.
pub fn run_predict(model: PredictModel, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let csv_blob = inv.get_blob("csv").ok_or("predict: missing input 'csv'")?;
    let text = std::str::from_utf8(&csv_blob.bytes).map_err(|_| "predict: csv is not UTF-8 text".to_string())?;
    let want_chart = inv.get_bool("chart").unwrap_or(false);
    // Fail on a missing renderer BEFORE any forecast: a run that cannot
    // produce the artifact the caller asked for should say so in milliseconds,
    // not after a minute of rollout.
    if want_chart && !forecast::chart::gnuplot_available() {
        return Err(format!("predict: chart needs the gnuplot CLI, which is not on PATH: {}", forecast::chart::INSTALL_HINT));
    }
    let series = forecast::csv::parse_ohlcv(text).map_err(|e| format!("predict: csv: {e}"))?;

    let horizon = int(inv, "horizon", DEFAULT_HORIZON).max(1);
    let requested = int(inv, "context", 0);
    let context = if requested == 0 { model.default_context } else { requested.min(model.max_context) };
    if let Some(exact) = model.exact_context.filter(|&e| context > e) {
        progress(Progress::step(
            0,
            0,
            format!(
                "context {context} + horizon {horizon} exceeds the model's {}-bar window, so the cached rollout slides it and is an approximation of the reference (use context {exact} or less for exactness)",
                model.max_context
            ),
        ));
    }
    let seed = inv.seed()?.unwrap_or_else(seed::random_seed);
    let item = inv.get_str("item").unwrap_or_else(|| "series".into());
    let freq = inv.get_str("freq").unwrap_or_else(|| "1h".into());
    let req = HoldoutRequest { series: &series, horizon, context, samples: int(inv, "samples", 1).max(1), origins: int(inv, "origins", 1).max(1), seed, season: int(inv, "season", DEFAULT_SEASON), item: &item, freq: &freq };

    let mut on_origin = |done: usize, all: usize| progress(Progress::step(done as u32, all as u32, format!("origin {done}/{all}")));
    let report = holdout::evaluate(model.name, model.forecast, &req, &mut on_origin).map_err(|e| format!("predict: {e}"))?;

    let m = report.model_score();
    let mut out = Outcome::new()
        .set("model", json!(model.name))
        .set("seed", json!(seed))
        .set("horizon", json!(horizon))
        .set("context", json!(context))
        .set("origins", json!(report.origins))
        .set("samples", json!(report.samples))
        .set("mae", json!(m.mae))
        .set("crps", json!(m.crps))
        .set("pinball", json!(m.pinball))
        .set("band_coverage", if report.band_coverage.is_finite() { json!(report.band_coverage) } else { json!(null) })
        .set("direction_hit", json!(report.direction_hit))
        .blob("report", Blob::new(Media::Text, report.render().into_bytes()));
    if want_chart {
        out = out.blob("chart", render_chart(&report, &item)?);
    }
    Ok(out)
}

/// The chart as PNG bytes. gnuplot writes files, so it draws into a private
/// scratch directory that is removed whether or not the render worked.
fn render_chart(report: &holdout::HoldoutReport, item: &str) -> Result<Blob, String> {
    let dir = capability::scratch::scratch_dir("forecast-chart").map_err(|e| format!("predict: a scratch directory: {e}"))?;
    let png = dir.join("chart.png");
    let drawn = forecast::chart::render_png(&report.chart(item), &png).and_then(|p| std::fs::read(&p).map_err(|e| format!("reading {}: {e}", p.display())));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(Blob::new(Media::Bytes, drawn.map_err(|e| format!("predict: {e}"))?).with_meta(json!({"format": "png"})))
}

#[cfg(test)]
mod tests {
    use super::*;
    use forecast::{Block, Representation, TargetForecast};

    fn csv(n: usize) -> String {
        let mut text = String::from("Date,open,high,low,close,volume\n");
        for i in 0..n {
            let (month, day) = (1 + i / 28, 1 + i % 28);
            let c = 100.0 + i as f32 * 0.5 + ((i % 7) as f32 - 3.0);
            text.push_str(&format!("2024-{month:02}-{day:02},{c:.2},{:.2},{:.2},{c:.2},1000\n", c + 1.0, c - 1.0));
        }
        text
    }

    fn persistence(panel: &Panel, spec: &ForecastSpec) -> Result<Forecast, ForecastError> {
        let last = *panel.items[0].variate("close").unwrap().data.last().unwrap();
        let mut fc = Forecast::new("persistence", Representation::Quantiles, spec.horizon, &panel.freq);
        let mut tf = TargetForecast::new(&panel.items[0].item_id, "close");
        tf.levels = vec![0.1, 0.5, 0.9];
        tf.quantiles = Some(Block::native(vec![spec.horizon, 3], vec![last; spec.horizon * 3]));
        fc.targets.push(tf);
        Ok(fc)
    }

    fn model<'a>(f: &'a mut dyn FnMut(&Panel, &ForecastSpec) -> Result<Forecast, ForecastError>) -> PredictModel<'a> {
        PredictModel { name: "persist", max_context: 60, default_context: 50, exact_context: Some(54), forecast: f }
    }

    fn inv(csv_text: String) -> Invocation {
        Invocation::new().blob("csv", Blob::new(Media::Text, csv_text.into_bytes())).set("horizon", json!(6)).set("season", json!(7)).set("seed", json!(3))
    }

    #[test]
    fn predict_scores_the_model_against_the_baselines_and_echoes_the_seed() {
        let out = run_predict(model(&mut persistence), &inv(csv(120)), &mut |_| {}).unwrap();
        assert_eq!(out.outputs["seed"], json!(3));
        assert_eq!(out.outputs["context"], json!(50), "a model's own window when the caller names none");
        let report = String::from_utf8(out.blobs["report"].bytes.clone()).unwrap();
        assert!(report.contains("persistence (last close)") && report.contains("seasonal naive (7 bars)"), "{report}");
        assert!(!out.blobs.contains_key("chart"), "no chart unless asked");
    }

    #[test]
    fn a_context_past_the_window_is_clamped_and_one_past_the_exact_limit_is_reported() {
        let mut said = Vec::new();
        let out = run_predict(model(&mut persistence), &inv(csv(120)).set("context", json!(500)), &mut |p| said.push(p.message)).unwrap();
        assert_eq!(out.outputs["context"], json!(60));
        assert!(said.iter().any(|m| m.contains("approximation")), "{said:?}");
    }

    #[test]
    fn a_malformed_csv_names_its_line_and_a_missing_one_is_refused() {
        let e = run_predict(model(&mut persistence), &inv("Date,open,high,low,close,volume\n2024-01-01,1,2\n".into()), &mut |_| {}).unwrap_err();
        assert!(e.contains("csv") && e.contains("line 2"), "{e}");
        let e = run_predict(model(&mut persistence), &Invocation::new(), &mut |_| {}).unwrap_err();
        assert!(e.contains("missing input 'csv'"), "{e}");
    }
}
