// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain forecast <compare|serve>` — the forecasting comparison + serving CLI.
//!
//! - `compare` runs the scenario battery against the statistical baselines and
//!   renders the model × scenario × metric report (markdown to stdout, optional
//!   self-contained HTML to `--html <path>`). This is the P0 "definition of
//!   done" deliverable; foundation models join the same battery once imported.
//! - `serve` starts the unified JSONL server (stdio, or a `--socket`/`--listen`
//!   TCP endpoint) with the baselines registered (and Chronos-2 via
//!   `--chronos2 <weights>`), so a Python client can drive forecasts over any
//!   transport.
//! - `import --hf <dir> --out chronos2.safetensors` converts an `amazon/chronos-2`
//!   checkpoint into a brain `.safetensors` container.

use crate::args::Args;
use std::sync::Arc;

pub fn run_forecast(argv: &[String]) {
    match argv.first().map(|s| s.as_str()) {
        Some("predict") => predict(&argv[1..]),
        Some("compare") => compare(&argv[1..]),
        Some("serve") => serve(&argv[1..]),
        Some("import") => import(&argv[1..]),
        Some("finetune") => finetune(&argv[1..]),
        other => {
            eprintln!("usage: brain forecast <predict|compare|serve|import|finetune> ...  (got {other:?})");
        }
    }
}

/// Default held-out horizon: 48 hourly bars, two full daily cycles of the
/// example series, long enough that a forecast which only copies the last
/// value is visibly wrong.
const DEFAULT_HORIZON: usize = 48;

/// The seasonal period the naive baseline uses when the caller names none.
/// 24 is "the same hour yesterday" on an hourly series - the sharpest cheap
/// baseline there is on anything with a daily cycle, and therefore the one
/// worth beating.
const DEFAULT_SEASON: usize = 24;

/// `brain forecast predict --csv <file>` - the one-command path: an OHLCV CSV
/// in, a scored forecast (and optionally a chart) out.
///
/// The last `--horizon` rows of the file are **held out**, never shown to the
/// model, and used as ground truth. That is what makes the printed numbers and
/// the chart evidence: the model is judged against a continuation it did not
/// see, next to two baselines that cost nothing, on one axis.
fn predict(args: &[String]) {
    let mut a = Args::new(args);
    let csv_path = a.str_or("--csv", "");
    let horizon = a.usize_or("--horizon", DEFAULT_HORIZON);
    let context = a.usize_or("--context", 0); // 0 = the model's own max_context
    let samples = a.usize_or("--samples", 1);
    let origins = a.usize_or("--origins", 1);
    let seed = a.u64_or("--seed", 7);
    let season = a.usize_or("--season", DEFAULT_SEASON);
    let item = a.str_or("--item", "series");
    let freq = a.str_or("--freq", "1h");
    let gnuplot = a.take_str("--gnuplot");
    let top_p = a.f32_or("--top-p", 0.0);
    let temperature = a.f32_or("--temperature", 0.0);
    let kronos_tok = a.take_str("--kronos-tokenizer");
    let kronos_dec = a.take_str("--kronos-decoder");
    a.finish();

    // The two sampling knobs that set how WIDE the predicted band is. Left
    // unset, the model keeps the reference defaults; the coverage line below is
    // what says whether those defaults are calibrated on this series.
    if top_p > 0.0 {
        std::env::set_var("BRAIN_KRONOS_TOP_P", top_p.to_string());
    }
    if temperature > 0.0 {
        std::env::set_var("BRAIN_KRONOS_TEMPERATURE", temperature.to_string());
    }

    if csv_path.is_empty() {
        eprintln!("usage: brain forecast predict --csv <ohlcv.csv> [--horizon 48] [--context N] [--samples N]");
        eprintln!("         [--origins 1] [--seed 7] [--season 24] [--gnuplot chart.png]");
        eprintln!("         [--top-p 0.9] [--temperature 1.0]");
        eprintln!("         [--kronos-tokenizer D --kronos-decoder D]");
        eprintln!();
        eprintln!("The CSV is timestamp,open,high,low,close,volume. The last --horizon rows are held out");
        eprintln!("as ground truth; everything before them is the model's context. --origins N repeats");
        eprintln!("that at N disjoint windows and averages, so the numbers are a measurement rather than");
        eprintln!("one draw; the chart always shows the most recent origin.");
        std::process::exit(2);
    }

    // Fail on a missing renderer BEFORE loading 400 MB of weights: a run that
    // cannot produce the artifact the caller asked for should say so in
    // milliseconds, not after a minute of rollout.
    if gnuplot.is_some() && !forecast::chart::gnuplot_available() {
        eprintln!("brain forecast predict: --gnuplot needs the gnuplot CLI, which is not on PATH");
        eprintln!("  {}", forecast::chart::INSTALL_HINT);
        std::process::exit(1);
    }

    let text = match std::fs::read_to_string(&csv_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain forecast predict: read {csv_path}: {e}");
            std::process::exit(1);
        }
    };
    let series = match forecast::csv::parse_ohlcv(&text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("brain forecast predict: {csv_path}: {e}");
            std::process::exit(1);
        }
    };

    // Weights: explicit flags win, else the env pair, else auto-fetch sets the
    // env pair from `NeoQuasar/Kronos-base` + `NeoQuasar/Kronos-Tokenizer-base`.
    if kronos_tok.is_none() || kronos_dec.is_none() {
        crate::supply::ensure_env_weights("kronos");
    }
    let env = |v: &str| std::env::var(v).ok().filter(|s| !s.is_empty());
    let (Some(tok), Some(dec)) = (kronos_tok.or_else(|| env("BRAIN_KRONOS_TOKENIZER")), kronos_dec.or_else(|| env("BRAIN_KRONOS_DECODER"))) else {
        eprintln!("brain forecast predict: no kronos checkpoint - auto-fetch did not resolve one, and neither");
        eprintln!("  --kronos-tokenizer/--kronos-decoder nor BRAIN_KRONOS_TOKENIZER/BRAIN_KRONOS_DECODER are set");
        std::process::exit(1);
    };
    let model = match kronos::KronosForecaster::load(&tok, &dec) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("brain forecast predict: load kronos from {tok} + {dec}: {e}");
            std::process::exit(1);
        }
    };
    // Default context: as much history as the checkpoint's window allows while
    // leaving room for the horizon INSIDE it. Past `max_context - horizon` the
    // rollout's attention window slides, and the KV-cached rollout stops being
    // equal to the reference's re-run-the-whole-window one. That is not a
    // rounding difference: on this checkpoint the sampled cloud drifts
    // measurably away from the reference's once the window moves, while inside
    // this regime the two agree sample for sample. A default that is silently
    // an approximation is the wrong default; ask for the longer window
    // explicitly with `--context` if you want it.
    let max_context = forecast::ForecastModel::capabilities(&model).max_context;
    let exact_context = max_context.saturating_sub(horizon).max(1);
    let context = if context == 0 { exact_context } else { context.min(max_context) };
    if context > exact_context {
        eprintln!("brain forecast predict: --context {context} + --horizon {horizon} exceeds the model's {max_context}-bar window,");
        eprintln!("  so the cached rollout slides it and is an approximation of the reference (use --context {exact_context} or less for exactness)");
    }

    // `samples == 1` means the deterministic modal rollout: one stable path,
    // reproducible run to run, and N times cheaper than drawing a cloud. More
    // than one draws real trajectories, the point path becomes the median, and
    // the chart gains a 10-90% band.
    let spec = forecast::ForecastSpec {
        horizon,
        representations: vec![forecast::Representation::Quantiles, forecast::Representation::Point],
        quantile_levels: vec![0.1, 0.5, 0.9],
        num_samples: samples.max(1),
        seed,
    };
    if samples <= 1 {
        std::env::set_var("BRAIN_KRONOS_ARGMAX", "1");
    }

    // Origin 0 is the end of the file; each further origin steps `horizon`
    // bars earlier, so the held-out windows are disjoint and no bar is scored
    // twice. Metrics are averaged over all of them - one origin is a draw, not
    // a measurement.
    let mut acc: Vec<(String, Vec<Score>)> = Vec::new(); // (model name, per-origin scores)
    let mut calib: Vec<(f32, f32)> = Vec::new(); // per-origin (band coverage, direction hit rate)
    let mut first: Option<(forecast::csv::Split, Vec<f32>, forecast::TargetForecast)> = None;
    let t0 = std::time::Instant::now();
    for o in 0..origins.max(1) {
        let split = match series.split_at_origin(context, horizon, o * horizon) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("brain forecast predict: {csv_path}: {e}");
                std::process::exit(1);
            }
        };
        let panel = forecast::csv::panel(&split, &item, &freq);
        let out = match forecast::ForecastModel::forecast(&model, &panel, &spec) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("brain forecast predict: forecast failed: {} ({})", e.message, e.code);
                std::process::exit(1);
            }
        };
        let Some(tf) = out.targets.into_iter().find(|t| t.name == "close") else {
            eprintln!("brain forecast predict: model returned no `close` target");
            std::process::exit(1);
        };
        let Some(pred) = point_path(&tf, horizon) else {
            eprintln!("brain forecast predict: model returned no usable point path");
            std::process::exit(1);
        };

        // The baselines. Persistence is the random-walk answer; drift
        // extrapolates the context's own mean log return; seasonal naive is the
        // same bar one period ago. All three are read from the CONTEXT only, so
        // they stay leak-free.
        let ctx_close: Vec<f32> = split.context.iter().map(|b| b.ohlcv[forecast::csv::CLOSE]).collect();
        let actual: Vec<f32> = split.actual.iter().map(|b| b.ohlcv[forecast::csv::CLOSE]).collect();
        let last = *ctx_close.last().expect("split guarantees a non-empty context");
        let mut row: Vec<(String, Score)> = vec![
            ("kronos".into(), Score::probabilistic(&pred, &tf, &actual)),
            ("persistence (last close)".into(), Score::point(&vec![last; horizon], &actual)),
            ("drift (context mean return)".into(), Score::point(&drift_path(&ctx_close, horizon), &actual)),
        ];
        if let Some(s) = seasonal_naive(&ctx_close, horizon, season) {
            row.push((format!("seasonal naive ({season} bars)"), Score::point(&s, &actual)));
        }
        for (i, (name, s)) in row.into_iter().enumerate() {
            if acc.len() <= i {
                acc.push((name, Vec::new()));
            }
            acc[i].1.push(s);
        }
        calib.push((band_coverage(&tf, &actual).unwrap_or(f32::NAN), forecast::metrics::directional_accuracy(&pred, &actual, last)));
        // Only the most recent origin is charted: it is the one whose held-out
        // window a reader can line up against the end of the input file.
        if o == 0 {
            first = Some((split, pred, tf));
        }
    }
    let elapsed = t0.elapsed();
    let n = origins.max(1);

    println!(
        "kronos forecast: {context} bars of context -> {horizon} held-out bars x {n} rolling origin{}  ({:.1}s, {} sample{})",
        if n == 1 { "" } else { "s" },
        elapsed.as_secs_f64(),
        spec.num_samples,
        if spec.num_samples == 1 { "" } else { "s" }
    );
    println!("  {:<28} {:>10} {:>10} {:>10}", "close, vs held-out truth", "mean MAE", "CRPS", "pinball");
    for (name, scores) in &acc {
        println!(
            "  {name:<28} {:>10.4} {:>10.4} {:>10.4}",
            mean_of(scores, |s| s.mae),
            mean_of(scores, |s| s.crps),
            mean_of(scores, |s| s.pinball)
        );
    }
    // Calibration is the claim a band forecast is actually making, so it is
    // printed as a number rather than left to the eye on the chart.
    let cov = calib.iter().map(|c| c.0).sum::<f32>() / calib.len() as f32;
    let dir = calib.iter().map(|c| c.1).sum::<f32>() / calib.len() as f32;
    // A one-sample run is the deterministic modal rollout: there is no band, so
    // a coverage line would read 0% and mean nothing.
    if cov.is_finite() && spec.num_samples > 1 {
        println!("  10-90% band covers {:.0}% of held-out bars (nominal 80%); direction hit rate {:.0}%", cov * 100.0, dir * 100.0);
    }
    if let (Some(k), Some(p)) = (acc.first(), acc.get(1)) {
        // CRPS, not MAE, is the headline: it is the proper score for the
        // distribution the model emits, and it reduces to MAE for the point
        // baselines - so the two rows are comparable on the same axis.
        let wins = k.1.iter().zip(&p.1).filter(|(a, b)| a.crps < b.crps).count();
        let (kc, pc) = (mean_of(&k.1, |s| s.crps), mean_of(&p.1, |s| s.crps));
        println!("  vs persistence: {:+.1}% CRPS reduction, better at {wins}/{n} origins", (1.0 - kc / pc) * 100.0);
    }

    if let (Some(path), Some((split, pred, tf))) = (gnuplot, &first) {
        match render_chart(&path, split, pred, tf, &item, horizon) {
            Ok(p) => println!("  chart: {}", p.display()),
            Err(e) => {
                eprintln!("brain forecast predict: {e}");
                std::process::exit(1);
            }
        }
    }
}

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
struct Score {
    mae: f32,
    crps: f32,
    pinball: f32,
}

impl Score {
    /// Score a distributional forecast: CRPS from the sample ensemble (the
    /// model's native representation) and pinball from its quantile grid.
    fn probabilistic(pred: &[f32], tf: &forecast::TargetForecast, actual: &[f32]) -> Score {
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
            // No ensemble: the point path IS the distribution, and its CRPS is
            // its MAE.
            _ => forecast::metrics::mae(pred, actual),
        };
        let pinball = match &tf.quantiles {
            Some(q) if !tf.levels.is_empty() && q.data.len() >= h * tf.levels.len() => {
                forecast::metrics::mean_pinball(&q.data[..h * tf.levels.len()], &tf.levels, actual)
            }
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
fn band_coverage(tf: &forecast::TargetForecast, actual: &[f32]) -> Option<f32> {
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
fn drift_path(ctx: &[f32], horizon: usize) -> Vec<f32> {
    let last = *ctx.last().unwrap_or(&0.0);
    if ctx.len() < 2 || last <= 0.0 || ctx.iter().any(|v| *v <= 0.0) {
        return vec![last; horizon];
    }
    let mu = (last.ln() - ctx[0].ln()) / (ctx.len() - 1) as f32;
    (1..=horizon).map(|h| last * (mu * h as f32).exp()).collect()
}

/// The forecast's point path over the horizon: the median quantile when the
/// model produced quantiles (robust for a sampled model), else the mean.
fn point_path(tf: &forecast::TargetForecast, horizon: usize) -> Option<Vec<f32>> {
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
fn seasonal_naive(ctx: &[f32], horizon: usize, season: usize) -> Option<Vec<f32>> {
    if season == 0 || ctx.len() < season {
        return None;
    }
    Some((0..horizon).map(|h| ctx[ctx.len() - season + (h % season)]).collect())
}

/// Assemble and render the evidence chart: the tail of the context, the
/// forecast, the held-out actual, and (when the model produced quantiles) the
/// 10-90% band under the forecast.
fn render_chart(
    path: &str,
    split: &forecast::csv::Split,
    pred: &[f32],
    tf: &forecast::TargetForecast,
    item: &str,
    horizon: usize,
) -> Result<std::path::PathBuf, String> {
    // Four horizons of history, and never less than two days of hourly bars:
    // enough to read the volatility regime the forecast is continuing (a
    // couple of dozen bars of a random walk read as a smooth trend, which is
    // exactly the wrong impression), without squeezing the part under
    // judgement into the right-hand margin.
    let show = (horizon * 4).max(48).min(split.context.len());
    let first = split.context.len() - show;
    let mut chart = forecast::chart::ForecastChart::new(format!("{item}: kronos {horizon}-bar forecast vs held-out actual"));
    chart.y_label = "close".to_string();
    chart.history = split.context[first..].iter().enumerate().map(|(i, b)| (i as f64, b.ohlcv[forecast::csv::CLOSE] as f64)).collect();
    let origin = show as f64 - 1.0;
    // The forecast and the actual both start at the context's last bar, so the
    // three lines meet at the origin rather than floating apart by one step.
    let last = split.context[split.context.len() - 1].ohlcv[forecast::csv::CLOSE] as f64;
    chart.forecast = std::iter::once((origin, last)).chain(pred.iter().enumerate().map(|(h, v)| (origin + 1.0 + h as f64, *v as f64))).collect();
    chart.actual = std::iter::once((origin, last)).chain(split.actual.iter().enumerate().map(|(h, b)| (origin + 1.0 + h as f64, b.ohlcv[forecast::csv::CLOSE] as f64))).collect();
    if let Some(q) = &tf.quantiles {
        let n = tf.levels.len();
        let (lo, hi) = (0usize, n.saturating_sub(1));
        if n >= 3 && q.data.len() >= horizon * n {
            chart.band = (0..horizon).map(|h| (origin + 1.0 + h as f64, q.data[h * n + lo] as f64, q.data[h * n + hi] as f64)).collect();
        }
    }
    forecast::chart::render_png(&chart, std::path::Path::new(path))
}

/// Where the foundation models' weights live, so the registry can load them.
#[derive(Clone, Default)]
struct FmPaths {
    chronos2: Option<String>,
    kronos_tokenizer: Option<String>,
    kronos_decoder: Option<String>,
    fincast: Option<String>,
}

/// Build a registry with the statistical baselines registered by name, plus any
/// foundation models whose weights `fm` supplies.
fn build_registry(fm: &FmPaths) -> runtime::Registry {
    let mut reg = runtime::Registry::new();
    reg.register_forecast(Arc::new(fcbench::RandomWalk));
    reg.register_forecast(Arc::new(fcbench::Drift));
    reg.register_forecast(Arc::new(fcbench::Arima { p: 2, d: 1 }));
    reg.register_forecast(Arc::new(fcbench::Garch11));
    if let Some(path) = &fm.chronos2 {
        match chronos2::Chronos2Forecaster::load(path) {
            Ok(m) => {
                reg.register_forecast(Arc::new(m));
                eprintln!("brain forecast: loaded chronos2 from {path}");
            }
            Err(e) => eprintln!("brain forecast: failed to load chronos2 from {path}: {e}"),
        }
    }
    if let (Some(tok), Some(dec)) = (&fm.kronos_tokenizer, &fm.kronos_decoder) {
        match kronos::KronosForecaster::load(tok, dec) {
            Ok(m) => {
                reg.register_forecast(Arc::new(m));
                eprintln!("brain forecast: loaded kronos from {tok} + {dec}");
            }
            Err(e) => eprintln!("brain forecast: failed to load kronos: {e}"),
        }
    }
    if let Some(path) = &fm.fincast {
        match fincast::FincastForecaster::load(path) {
            Ok(m) => {
                reg.register_forecast(Arc::new(m));
                eprintln!("brain forecast: loaded fincast from {path}");
            }
            Err(e) => eprintln!("brain forecast: failed to load fincast from {path}: {e}"),
        }
    }
    reg
}

/// `brain forecast import --hf <dir> --out chronos2.safetensors` (Chronos-2), or
/// `brain forecast import --fincast <v1.safetensors> --out fincast.safetensors`.
fn import(args: &[String]) {
    let mut a = Args::new(args);
    let hf = a.str_or("--hf", "");
    let fincast_ckpt = a.take_str("--fincast");
    let out = a.str_or("--out", "chronos2.safetensors");
    a.finish();
    if let Some(ckpt) = fincast_ckpt {
        let out = if out == "chronos2.safetensors" { "fincast.safetensors".to_string() } else { out };
        match fincast::import::import(&ckpt, &out) {
            Ok(()) => println!("ok: wrote {out}"),
            Err(e) => eprintln!("import failed: {e}"),
        }
        return;
    }
    if hf.is_empty() {
        eprintln!("usage: brain forecast import --hf <amazon/chronos-2 dir> --out chronos2.safetensors");
        eprintln!("   or: brain forecast import --fincast <FinCast safetensors> --out fincast.safetensors");
        return;
    }
    match chronos2::import::import(&hf, &out) {
        Ok(()) => println!("ok: wrote {out}"),
        Err(e) => eprintln!("import failed: {e}"),
    }
}

fn compare(args: &[String]) {
    let mut a = Args::new(args);
    let windows = a.usize_or("--windows", 24);
    let seed = a.u64_or("--seed", 1337);
    let html_path = a.take_str("--html");
    let chronos2_weights = a.take_str("--chronos2");
    let kronos_tok = a.take_str("--kronos-tokenizer");
    let kronos_dec = a.take_str("--kronos-decoder");
    let fincast_weights = a.take_str("--fincast");
    a.finish();

    let mut models = fcbench::baselines::default_set();
    // optionally add the parity-verified Chronos-2 foundation model.
    if let Some(path) = &chronos2_weights {
        match chronos2::Chronos2Forecaster::load(path) {
            Ok(m) => {
                eprintln!("comparison: loaded chronos2 from {path} (slow on CPU — keep --windows small)");
                models.push(Box::new(m));
            }
            Err(e) => eprintln!("comparison: failed to load chronos2 from {path}: {e}"),
        }
    }
    // optionally add the finance-native Kronos model.
    if let (Some(tok), Some(dec)) = (&kronos_tok, &kronos_dec) {
        match kronos::KronosForecaster::load(tok, dec) {
            Ok(m) => {
                eprintln!("comparison: loaded kronos from {tok} + {dec}");
                models.push(Box::new(m));
            }
            Err(e) => eprintln!("comparison: failed to load kronos: {e}"),
        }
    }
    // optionally add the finance-native FinCast MoE model.
    if let Some(path) = &fincast_weights {
        match fincast::FincastForecaster::load(path) {
            Ok(m) => {
                eprintln!("comparison: loaded fincast from {path} (slow on CPU — keep --windows small)");
                models.push(Box::new(m));
            }
            Err(e) => eprintln!("comparison: failed to load fincast from {path}: {e}"),
        }
    }
    let scenarios = fcbench::scenarios::default_battery();
    let cmp = fcbench::harness::run(&models, &scenarios, windows, seed);

    // markdown to stdout
    print!("{}", fcbench::report::markdown(&cmp));

    // the negative control is a hard gate: fail the process if a model falsely
    // beats naive on the random walk.
    let violations = cmp.negative_control_violations("random_walk", 0.10);
    if !violations.is_empty() {
        eprintln!("negative-control FAILED: {} falsely beat naive", violations.join(", "));
        std::process::exit(1);
    }

    if let Some(path) = html_path {
        match std::fs::write(&path, fcbench::report::html(&cmp)) {
            Ok(()) => eprintln!("wrote {path}"),
            Err(e) => eprintln!("failed to write {path}: {e}"),
        }
    }
}

fn serve(args: &[String]) {
    let mut a = Args::new(args);
    let socket = a.take_str("--socket");
    let listen = a.take_str("--listen");
    let chronos2 = a.take_str("--chronos2");
    let kronos_tok = a.take_str("--kronos-tokenizer");
    let kronos_dec = a.take_str("--kronos-decoder");
    let fincast = a.take_str("--fincast");
    let max_conn = a.usize_or("--max-connections", 64);
    a.finish();

    let fm = FmPaths {
        chronos2: chronos2.clone(),
        kronos_tokenizer: kronos_tok,
        kronos_decoder: kronos_dec,
        fincast,
    };

    let opts = server::ServeOpts { max_connections: max_conn };
    // A fresh controller (with its own model instances) per connection, so
    // per-instance state never crosses threads. Foundation models are (re)loaded
    // per connection for now; sharing one replica across workers is the deferred
    // throughput optimization.
    let fm_conn = fm.clone();
    let make: server::transport::SessionFactory = Arc::new(move || {
        let reg = build_registry(&fm_conn);
        Box::new(server::ControllerSession::new(runtime::Controller::new(reg)))
    });

    if let Some(path) = socket {
        eprintln!("brain forecast: serving on unix socket {path}");
        if let Err(e) = server::serve_unix(&path, make, opts) {
            eprintln!("serve_unix failed: {e}");
            std::process::exit(1);
        }
    } else if let Some(addr) = listen {
        eprintln!("brain forecast: serving on tcp {addr}");
        if let Err(e) = server::serve_tcp(&addr, make, opts) {
            eprintln!("serve_tcp failed: {e}");
            std::process::exit(1);
        }
    } else {
        // stdio: a single session on stdin/stdout.
        eprintln!("brain forecast: serving on stdio");
        let reg = build_registry(&fm);
        let mut session = server::ControllerSession::new(runtime::Controller::new(reg));
        if let Err(e) = server::serve_stdio(&mut session) {
            eprintln!("serve_stdio failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Load a directory of `<TICKER>.csv` (Date,open,high,low,close,volume) into
/// leak-safe-windowing `Series` (drops the index ETF `QQQ` if present).
fn load_series(dir: &str) -> Vec<forecast::train_data::Series> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    for ent in rd.flatten() {
        let name = ent.file_name().to_string_lossy().to_string();
        let Some(ticker) = name.strip_suffix(".csv") else { continue };
        if ticker.eq_ignore_ascii_case("QQQ") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(ent.path()) else { continue };
        let mut dates = Vec::new();
        let mut ohlcv = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if i == 0 {
                continue;
            }
            let c: Vec<&str> = line.split(',').collect();
            if c.len() < 6 {
                continue;
            }
            let d: Vec<i64> = c[0][..10.min(c[0].len())].split('-').filter_map(|x| x.parse().ok()).collect();
            let v: Option<Vec<f32>> = c[1..6].iter().map(|x| x.parse().ok()).collect();
            if d.len() == 3 {
                if let Some(v) = v {
                    dates.push((d[0] as i32, d[1] as u32, d[2] as u32));
                    ohlcv.push([v[0], v[1], v[2], v[3], v[4]]);
                }
            }
        }
        if !ohlcv.is_empty() {
            out.push(forecast::train_data::Series { ticker: ticker.into(), dates, ohlcv });
        }
    }
    out
}

/// `brain forecast finetune` — weekly gated fine-tune of the Kronos decoder over a
/// universe of OHLCV CSVs. Fine-tunes on the past, and writes a promoted checkpoint
/// ONLY if it beats the base on a held-out (embargoed) split.
fn finetune(args: &[String]) {
    let mut a = Args::new(args);
    let tok = a.str_or("--kronos-tokenizer", "");
    let dec = a.str_or("--kronos-decoder", "");
    let data = a.str_or("--data", "");
    let out = a.take_str("--out");
    let holdout = a.take_str("--holdout-data");
    let context = a.usize_or("--context", 180);
    let horizon = a.usize_or("--horizon", 5);
    let epochs = a.usize_or("--epochs", 8) as u32;
    let lr = a.f32_or("--lr", 4e-5);
    let lora_rank = a.usize_or("--lora", 0);
    let embargo = a.usize_or("--embargo", horizon);
    let batch = a.usize_or("--batch", 1).max(1) as u32;
    a.finish();
    if tok.is_empty() || dec.is_empty() || data.is_empty() {
        eprintln!("usage: brain forecast finetune --kronos-tokenizer <dir> --kronos-decoder <dir> --data <csv-dir> \\");
        eprintln!("         [--out <ckpt>] [--context 180] [--horizon 5] [--epochs 8] [--lr 4e-5] [--lora RANK] [--embargo N] [--batch B]");
        return;
    }
    let (cfg, base) = match kronos::import::load_decoder(&dec) {
        Ok(x) => x,
        Err(e) => return eprintln!("load decoder: {e}"),
    };
    let model = match kronos::import::load_model(&tok, &dec) {
        Ok(m) => m,
        Err(e) => return eprintln!("load model: {e}"),
    };
    let series = load_series(&data);
    if series.len() < 2 {
        return eprintln!("need >= 2 series with data in {data}");
    }
    eprintln!("finetune: {} names · context {context} · horizon {horizon} · epochs {epochs} · lr {lr}{}",
        series.len(), if lora_rank > 0 { format!(" · LoRA r{lora_rank}") } else { " · full".into() });
    let split = forecast::train_data::SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo };
    let lora = (lora_rank > 0).then(|| kronos::train::LoraCfg::attn(lora_rank, (lora_rank * 2) as f32));
    let opts = kronos::train::FinetuneOpts { epochs, lr, wd: 0.1, clip: 3.0, lora, batch, progress: true };
    let (rep, weights) = kronos::finetune::finetune_universe(&model, &base, &series, context, horizon, split, &opts);
    println!(
        "\ngate (INCLUDED names, held-out future): base_val {:.4} → ft_val {:.4}  ({} steps)  ⇒  {}",
        rep.base_val, rep.ft_val, rep.steps,
        if rep.promoted { "PROMOTE (fine-tune beats base out-of-sample)" } else { "KEEP BASE (no held-out improvement)" }
    );
    let w = weights.expect("weights always returned");
    // Save FIRST (so a slow generalization eval can't cost us the checkpoint).
    if rep.promoted {
        let path = out.unwrap_or_else(|| "kronos-decoder-ft.safetensors".into());
        kronos::finetune::save_decoder_weights(&cfg, &w, &path);
        println!("promoted checkpoint → {path}");
    } else {
        println!("not promoted → no checkpoint written (base kept)");
    }
    // Generalization: does the fine-tune also improve names it NEVER trained on?
    if let Some(hd) = holdout {
        let hs = load_series(&hd);
        if !hs.is_empty() {
            let base_h = kronos::finetune::eval_universe_loss(&model, &base, &hs, context, horizon);
            let ft_h = kronos::finetune::eval_universe_loss(&model, &w, &hs, context, horizon);
            println!(
                "held-out NAMES ({} names, never fine-tuned): base {base_h:.4} → ft {ft_h:.4}  ⇒  {}",
                hs.len(),
                if ft_h < base_h { "GENERALIZES (more accurate on unseen instruments)" } else { "no gain on unseen names" }
            );
        }
    }
}
