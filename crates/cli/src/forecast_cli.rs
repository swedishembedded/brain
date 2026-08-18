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

/// The index the universe is drawn FROM, rather than one of its constituents:
/// a file for it is skipped by name (and said so), never trained on.
const INDEX_ETF: &str = "QQQ";

/// One CSV the loader refused, and why. `reason` carries the message
/// [`forecast::csv::parse_ohlcv`] produced, so it already names the offending
/// 1-based file line - a universe is hundreds of vendor files, and "something
/// in your data is wrong" is not something a caller can act on.
#[derive(Debug)]
struct Rejected {
    file: String,
    reason: String,
}

/// What a universe directory turned out to hold. Every count the caller needs
/// to state what it trained on is here; nothing is dropped on the floor.
#[derive(Debug, Default)]
struct Universe {
    /// The files that passed validation, in file-name order.
    series: Vec<forecast::train_data::Series>,
    /// The files that did not, in file-name order.
    rejected: Vec<Rejected>,
    /// Files deliberately not treated as instruments (see [`INDEX_ETF`]).
    excluded: Vec<String>,
}

/// Load a directory of `<TICKER>.csv` (`Date,open,high,low,close,volume`) into
/// leak-safe-windowing [`forecast::train_data::Series`].
///
/// Every file goes through [`forecast::csv::parse_ohlcv`] - the same structural
/// and semantic validation `forecast predict` uses - so a ragged row, an
/// unparseable number, a repeated or backwards date, a non-finite value or an
/// impossible bar is REPORTED with its file and line instead of skipped. A
/// fine-tune that quietly trains on a fraction of the data it was pointed at
/// still prints a promotion verdict, and that verdict is then about an
/// experiment nobody chose.
///
/// `min_rows` is the request's own length floor (`context + horizon`): a file
/// too short to yield a single training window contributes nothing to the run,
/// so it is a rejection rather than a silent no-op.
///
/// `Err` is a directory-level failure (unreadable, or holding no CSV at all).
/// Per-file failures come back in [`Universe::rejected`] for [`accept_universe`]
/// to apply the run's policy to.
fn load_universe(dir: &str, min_rows: usize) -> Result<Universe, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("read {dir}: {e}"))?;
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| format!("read {dir}: {e}"))?;
        let path = ent.path();
        if path.extension().is_some_and(|x| x.eq_ignore_ascii_case("csv")) && path.is_file() {
            files.push(path);
        }
    }
    if files.is_empty() {
        return Err(format!("{dir}: no `<TICKER>.csv` files (expected one OHLCV file per instrument)"));
    }
    // `read_dir` yields whatever order the filesystem feels like, which would
    // make the universe - and therefore the window order the fine-tune walks -
    // differ run to run on the same directory. Sort, so a rerun is a rerun.
    files.sort();

    let mut u = Universe::default();
    for path in files {
        let file = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy().to_string();
        let ticker = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        if ticker.eq_ignore_ascii_case(INDEX_ETF) {
            u.excluded.push(file);
            continue;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                u.rejected.push(Rejected { file, reason: format!("read failed: {e}") });
                continue;
            }
        };
        let parsed = match forecast::csv::parse_ohlcv(&text) {
            Ok(s) => s,
            Err(e) => {
                u.rejected.push(Rejected { file, reason: e });
                continue;
            }
        };
        if parsed.len() < min_rows {
            u.rejected.push(Rejected {
                file,
                reason: format!("csv: {} data rows is too few for one training window ({min_rows} needed) - shorten --context/--horizon or supply a longer file", parsed.len()),
            });
            continue;
        }
        u.series.push(forecast::train_data::Series {
            ticker,
            dates: parsed.bars.iter().map(|b| (b.stamp.year, b.stamp.month, b.stamp.day)).collect(),
            ohlcv: parsed.bars.iter().map(|b| b.ohlcv).collect(),
        });
    }
    Ok(u)
}

/// Apply the run's strictness policy to a loaded [`Universe`], after printing
/// what it contains.
///
/// **Strict is the default, and a rejected file fails the whole run.** A
/// fine-tune is an hours-long job whose entire output is one promote/keep
/// verdict about one universe; a universe that quietly lost a third of its
/// names yields a verdict about a different experiment, and nothing in the
/// output distinguishes the two afterwards. Validation costs milliseconds and
/// happens before any weights load, so the refusal is cheap and immediate,
/// while the wasted run it prevents is not.
///
/// `--skip-invalid` is the escape hatch for the case where refusing everything
/// is the wrong answer - a several-hundred-name vendor dump with one bad row in
/// one file. It still lists every rejection with its file and line, and both
/// paths print the loaded/rejected split, so the run is never quieter about its
/// data than the data deserves.
fn accept_universe(dir: &str, label: &str, u: Universe, skip_invalid: bool) -> Result<Vec<forecast::train_data::Series>, String> {
    let total = u.series.len() + u.rejected.len();
    for r in &u.rejected {
        eprintln!("{label}: {}: {}", r.file, r.reason);
    }
    if !u.excluded.is_empty() {
        eprintln!("{label}: excluded {} ({INDEX_ETF} is the benchmark index, not one of its constituents)", u.excluded.join(", "));
    }
    eprintln!("{label}: {dir}: {} of {total} series loaded, {} rejected", u.series.len(), u.rejected.len());
    if !u.rejected.is_empty() && !skip_invalid {
        return Err(format!(
            "{label}: refusing to run: {} of {total} files in {dir} failed validation (listed above) - fix them, or pass --skip-invalid to train on the {} that parsed",
            u.rejected.len(),
            u.series.len()
        ));
    }
    if u.series.is_empty() {
        return Err(format!("{label}: {dir}: no usable series"));
    }
    Ok(u.series)
}

/// [`load_universe`] + [`accept_universe`], exiting with a non-zero status
/// rather than returning on any refusal.
fn load_universe_or_exit(dir: &str, label: &str, min_rows: usize, skip_invalid: bool) -> Vec<forecast::train_data::Series> {
    // Both messages already open with `label`, which names WHICH directory
    // (`--data` or `--holdout-data`) is at fault, so no second prefix is added.
    match load_universe(dir, min_rows).map_err(|e| format!("{label}: {e}")).and_then(|u| accept_universe(dir, label, u, skip_invalid)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
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
    let skip_invalid = a.take_flag("--skip-invalid");
    a.finish();
    if tok.is_empty() || dec.is_empty() || data.is_empty() {
        eprintln!("usage: brain forecast finetune --kronos-tokenizer <dir> --kronos-decoder <dir> --data <csv-dir> \\");
        eprintln!("         [--out <ckpt>] [--context 180] [--horizon 5] [--epochs 8] [--lr 4e-5] [--lora RANK] [--embargo N] [--batch B]");
        eprintln!("         [--holdout-data <csv-dir>] [--skip-invalid]");
        eprintln!();
        eprintln!("Each <TICKER>.csv is Date,open,high,low,close,volume. Every file is validated before");
        eprintln!("anything is loaded; a file with a bad row fails the run naming the file and the line,");
        eprintln!("and --skip-invalid trains on the rest instead, reporting how many were rejected.");
        std::process::exit(2);
    }
    // A window shorter than two bars has no next token to predict, so every
    // window would be dropped by the tokenizer and the run would train on
    // nothing; a zero horizon leaves the gate scoring an empty future.
    if context < 2 || horizon == 0 {
        eprintln!("brain forecast finetune: --context {context} --horizon {horizon}: need --context >= 2 and --horizon >= 1");
        std::process::exit(2);
    }
    // The universes are read and validated FIRST: a typo in --data or one bad
    // row deserves an answer in milliseconds, not after several hundred MB of
    // checkpoint has loaded (and, for --holdout-data, not after the whole
    // fine-tune has already run).
    let min_rows = context + horizon;
    let series = load_universe_or_exit(&data, "finetune --data", min_rows, skip_invalid);
    if series.len() < 2 {
        eprintln!("brain forecast finetune: {data}: {} usable series, need >= 2 to fine-tune a universe", series.len());
        std::process::exit(1);
    }
    let holdout_series = holdout.as_ref().map(|hd| load_universe_or_exit(hd, "finetune --holdout-data", min_rows, skip_invalid));

    let (cfg, base) = match kronos::import::load_decoder(&dec) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("brain forecast finetune: load decoder: {e}");
            std::process::exit(1);
        }
    };
    let model = match kronos::import::load_model(&tok, &dec) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("brain forecast finetune: load model: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("finetune: {} names · context {context} · horizon {horizon} · epochs {epochs} · lr {lr}{}",
        series.len(), if lora_rank > 0 { format!(" · LoRA r{lora_rank}") } else { " · full".into() });
    let split = forecast::train_data::SplitConfig { train_frac: 0.7, val_frac: 0.15, embargo };
    let lora = (lora_rank > 0).then(|| kronos::train::LoraCfg::attn(lora_rank, (lora_rank * 2) as f32));
    let opts = kronos::train::FinetuneOpts { epochs, lr, wd: 0.1, clip: 3.0, lora, batch, progress: true };
    let (rep, weights) = kronos::finetune::finetune_universe(&model, &base, &series, context, horizon, split, &opts);
    // Zero steps is not a verdict. It means the embargoed split left no training
    // window at all, and printing "KEEP BASE" for it would read as "the
    // fine-tune was tried and lost" when nothing was ever trained.
    if rep.steps == 0 {
        eprintln!("brain forecast finetune: 0 training steps - the {} series in {data} yielded no TRAIN windows", series.len());
        eprintln!("  after the embargoed temporal split (train_frac 0.7, embargo {embargo} bars). Supply longer series,");
        eprintln!("  or shorten --context/--horizon/--embargo.");
        std::process::exit(1);
    }
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
    if let Some(hs) = holdout_series {
        let base_h = kronos::finetune::eval_universe_loss(&model, &base, &hs, context, horizon);
        let ft_h = kronos::finetune::eval_universe_loss(&model, &w, &hs, context, horizon);
        println!(
            "held-out NAMES ({} names, never fine-tuned): base {base_h:.4} → ft {ft_h:.4}  ⇒  {}",
            hs.len(),
            if ft_h < base_h { "GENERALIZES (more accurate on unseen instruments)" } else { "no gain on unseen names" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` valid daily bars from 2024-01-01: strictly increasing dates (28 days
    /// a month keeps every one a real calendar date) and OHLC invariants that
    /// hold. Every negative case below is this with exactly one thing broken -
    /// the same shape `forecast::csv`'s own tests use.
    fn good_csv(n: usize) -> String {
        let mut s = String::from("Date,open,high,low,close,volume\n");
        for i in 0..n {
            let (month, day) = (1 + i / 28, 1 + i % 28);
            let o = 100.0 + i as f32;
            s.push_str(&format!("2024-{month:02}-{day:02},{o:.2},{:.2},{:.2},{:.2},1000\n", o + 2.0, o - 2.0, o + 1.0));
        }
        s
    }

    /// A scratch directory holding `files`, removed when the test drops it.
    struct Dir(std::path::PathBuf);

    impl Dir {
        fn new(tag: &str, files: &[(&str, String)]) -> Dir {
            let d = std::env::temp_dir().join(format!("brain-cli-universe-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).expect("create scratch dir");
            for (name, text) in files {
                std::fs::write(d.join(name), text).expect("write csv");
            }
            Dir(d)
        }
        fn path(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Load a one-file universe and return the rejection message, asserting the
    /// file was refused rather than silently dropped.
    fn reason(tag: &str, file: &str, text: String, min_rows: usize) -> String {
        let d = Dir::new(tag, &[(file, text)]);
        let u = load_universe(&d.path(), min_rows).expect("the directory itself is readable");
        assert!(u.series.is_empty(), "{file} should not have loaded as a series");
        assert_eq!(u.rejected.len(), 1, "{file} should be rejected exactly once");
        assert_eq!(u.rejected[0].file, file, "the rejection must name the file");
        u.rejected[0].reason.clone()
    }

    #[test]
    fn a_well_formed_universe_loads_every_bar() {
        let d = Dir::new("good", &[("AAPL.csv", good_csv(30)), ("MSFT.csv", good_csv(40))]);
        let u = load_universe(&d.path(), 10).unwrap();
        assert!(u.rejected.is_empty(), "{:?}", u.rejected.iter().map(|r| &r.reason).collect::<Vec<_>>());
        // Sorted by file name, so a rerun walks the universe in the same order.
        assert_eq!(u.series.iter().map(|s| s.ticker.as_str()).collect::<Vec<_>>(), ["AAPL", "MSFT"]);
        assert_eq!(u.series[0].len(), 30);
        assert_eq!(u.series[1].len(), 40);
        assert_eq!(u.series[0].dates[0], (2024, 1, 1));
        assert_eq!(u.series[0].ohlcv[0], [100.0, 102.0, 98.0, 101.0, 1000.0]);
    }

    /// One test per class of defect that used to be `continue`d past. The
    /// assertion is on the message: a rejection that does not say WHICH file and
    /// WHICH line is barely better than the silent skip it replaces.
    #[test]
    fn a_ragged_row_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        text.push_str("2024-01-21,120.00,122.00,118.00\n"); // 4 fields, not 6
        let r = reason("ragged", "AAPL.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("4 fields"), "{r}");
    }

    #[test]
    fn an_unparseable_number_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        text.push_str("2024-01-21,120.00,122.00,118.00,n/a,1000\n");
        let r = reason("unparseable", "MSFT.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("is not a number"), "{r}");
    }

    #[test]
    fn a_non_monotonic_date_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        text.push_str("2024-01-19,120.00,122.00,118.00,121.00,1000\n"); // backwards
        let r = reason("backwards", "NVDA.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("not after the previous"), "{r}");
    }

    #[test]
    fn an_impossible_bar_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        // high below the close: a shape-only parser waves this straight through.
        text.push_str("2024-01-21,120.00,122.00,118.00,130.00,1000\n");
        let r = reason("impossible", "TSLA.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("is below open"), "{r}");
    }

    #[test]
    fn a_file_too_short_for_one_window_is_refused_with_its_count() {
        // 12 rows cannot serve a 10-bar context plus a 5-bar horizon.
        let r = reason("short", "AMD.csv", good_csv(12), 15);
        assert!(r.contains("12 data rows") && r.contains("15 needed"), "{r}");
    }

    #[test]
    fn a_header_only_file_is_refused_rather_than_counted_as_a_series() {
        let r = reason("empty", "INTC.csv", "Date,open,high,low,close,volume\n".into(), 1);
        assert!(r.contains("no data rows"), "{r}");
    }

    #[test]
    fn an_unreadable_or_empty_directory_is_an_error_not_an_empty_universe() {
        let missing = std::env::temp_dir().join(format!("brain-cli-universe-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&missing);
        let e = load_universe(&missing.to_string_lossy(), 1).unwrap_err();
        assert!(e.contains("read") && e.contains("absent"), "{e}");
        // A directory that exists but holds no CSV is the same mistake with a
        // different spelling, and used to return an empty Vec just as quietly.
        let d = Dir::new("nocsv", &[("README.md", "not data".into())]);
        let e = load_universe(&d.path(), 1).unwrap_err();
        assert!(e.contains("no `<TICKER>.csv` files"), "{e}");
    }

    /// The directory-case decision, gated: a mixed directory FAILS the whole run
    /// by default and only trains on the survivors when the caller explicitly
    /// asks. Either way the counts are reported, and the count is the point.
    #[test]
    fn a_mixed_directory_fails_by_default_and_skips_only_when_asked() {
        let mut bad = good_csv(20);
        bad.push_str("2024-01-21,120.00,122.00,118.00\n");
        let files = [("AAPL.csv", good_csv(30)), ("BAD.csv", bad), ("MSFT.csv", good_csv(30)), ("QQQ.csv", good_csv(30))];
        let d = Dir::new("mixed", &files);

        let u = load_universe(&d.path(), 10).unwrap();
        assert_eq!(u.series.len(), 2, "two good names");
        assert_eq!(u.rejected.len(), 1, "one bad name");
        assert_eq!(u.excluded, ["QQQ.csv"], "the index is excluded by name, and said so");

        // Default: the run is refused, and the message says how many failed, how
        // many survived, and how to proceed anyway.
        let e = accept_universe(&d.path(), "finetune", u, false).unwrap_err();
        assert!(e.contains("1 of 3 files") && e.contains("--skip-invalid"), "{e}");

        // Opt-in: the survivors train, and only the survivors.
        let u = load_universe(&d.path(), 10).unwrap();
        let s = accept_universe(&d.path(), "finetune", u, true).unwrap();
        assert_eq!(s.iter().map(|s| s.ticker.as_str()).collect::<Vec<_>>(), ["AAPL", "MSFT"]);

        // ... but --skip-invalid never degrades into "trained on nothing".
        let d = Dir::new("allbad", &[("BAD.csv", "Date,open,high,low,close,volume\nnonsense\n".into())]);
        let u = load_universe(&d.path(), 10).unwrap();
        let e = accept_universe(&d.path(), "finetune", u, true).unwrap_err();
        assert!(e.contains("no usable series"), "{e}");
    }
}
