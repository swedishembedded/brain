// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-agnostic out-of-sample cross-sectional skill eval + per-forecast latency
//! benchmark. Generalizes `crates/kronos/tests/rankic_eval.rs` to drive **any**
//! [`forecast::ForecastModel`] (Chronos-2, Kronos, FinCast) through the object-safe
//! seam — the panel shape is chosen from each model's own
//! [`capabilities`](forecast::ForecastModel::capabilities) (OHLCV bars for a model
//! that `requires_variates`, univariate close otherwise), so one loop evaluates all
//! three with no per-model code.
//!
//! At each weekly origin (last-observed date `d`), for every liquid name in the
//! universe we forecast the `H`-bar-ahead close from `CTX` bars of real history,
//! record the predicted vs realized `H`-bar return, accumulate point MASE (vs the
//! naive last-value baseline), and time the forecast. A separate Python step
//! (`tools/oos_skill_report.py`) ranks each origin's cross-section and computes
//! RankIC (Spearman) ± stderr + t, directional accuracy, a cost-aware long/short
//! basket vs `^gspc`, and the shuffled negative control + naive baseline.
//!
//! In-process (weights loaded once, warm across the sweep). Env-gated; skips
//! without checkpoints + data. Leak-safe: origins are chosen strictly after the
//! model's training cutoff by the caller (2026 dates); normalization is past-only
//! inside each model.
//!
//! Env:
//!   CHRONOS2_WEIGHTS                          — chronos2 `.weights` (optional)
//!   KRONOS_TOKENIZER_DIR, KRONOS_DECODER_DIR  — kronos checkpoints (optional)
//!   FINCAST_WEIGHTS                           — fincast `.weights` (optional)
//!   OOS_DATA      — dir of `<TICKER>.csv` (Date,open,high,low,close,volume) (required)
//!   OOS_OUT       — output JSON path (required)
//!   OOS_CTX=200  OOS_HORIZON=5  OOS_STEP=5  OOS_START=2026-01-01
//!   OOS_NSAMPLES=16 (kronos sample count)  OOS_MAXORIG=0(=all)  OOS_WARMUP=2
//!   OOS_LATENCY_ONLY=0  (1 = warmup + a few timed forecasts per model, no eval)
//!   BRAIN_DEVICE  — cpu|gpu|vulkan (recorded in meta; selects the backend)

use forecast::{
    Forecast, ForecastModel, ForecastSpec, Item, Kind, Panel, Representation, Role, Variate,
};
use std::fmt::Write as _;
use std::time::Instant;

struct Series {
    ticker: String,
    dates: Vec<(i32, u32, u32)>,
    ohlcv: Vec<[f32; 5]>,
}

fn load_csv(path: &str, ticker: &str) -> Option<Series> {
    let text = std::fs::read_to_string(path).ok()?;
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
        let vals: Option<Vec<f32>> = c[1..6].iter().map(|x| x.parse().ok()).collect();
        if d.len() == 3 {
            if let Some(v) = vals {
                dates.push((d[0] as i32, d[1] as u32, d[2] as u32));
                ohlcv.push([v[0], v[1], v[2], v[3], v[4]]);
            }
        }
    }
    if ohlcv.is_empty() {
        return None;
    }
    Some(Series { ticker: ticker.into(), dates, ohlcv })
}

fn envu(k: &str, def: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}
fn envs(k: &str, def: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| def.to_string())
}

/// Load whichever foundation models the env points at, as `(name, model)`.
fn build_models() -> Vec<(String, Box<dyn ForecastModel>)> {
    let mut models: Vec<(String, Box<dyn ForecastModel>)> = Vec::new();
    if let Ok(w) = std::env::var("CHRONOS2_WEIGHTS") {
        match chronos2::Chronos2Forecaster::load(&w) {
            Ok(m) => {
                models.push(("chronos2".into(), Box::new(m)));
                eprintln!("loaded chronos2 from {w}");
            }
            Err(e) => eprintln!("chronos2 load failed: {e}"),
        }
    }
    if let (Ok(t), Ok(d)) = (std::env::var("KRONOS_TOKENIZER_DIR"), std::env::var("KRONOS_DECODER_DIR")) {
        match kronos::KronosForecaster::load(&t, &d) {
            Ok(m) => {
                models.push(("kronos".into(), Box::new(m)));
                eprintln!("loaded kronos from {t} + {d}");
            }
            Err(e) => eprintln!("kronos load failed: {e}"),
        }
    }
    if let Ok(w) = std::env::var("FINCAST_WEIGHTS") {
        match fincast::FincastForecaster::load(&w) {
            Ok(m) => {
                models.push(("fincast".into(), Box::new(m)));
                eprintln!("loaded fincast from {w}");
            }
            Err(e) => eprintln!("fincast load failed: {e}"),
        }
    }
    models
}

/// Build the capability-appropriate panel for one context window. A model that
/// requires OHLCV variates (Kronos) gets a six-column bar item; a univariate model
/// (Chronos-2, FinCast) gets a single `close` target — chosen from the model's own
/// advertised capabilities, so the driver stays model-agnostic.
fn build_panel(model: &dyn ForecastModel, ctx: &[[f32; 5]]) -> Panel {
    let caps = model.capabilities();
    let n = ctx.len();
    if caps.requires_variates.is_empty() {
        // univariate close target.
        let close: Vec<f32> = ctx.iter().map(|b| b[3]).collect();
        Panel::single("1d", "X", vec![Variate::target("close", close)])
    } else {
        // OHLCV bar item: close = target, the rest past-covariates (read by name).
        let names = ["open", "high", "low", "close", "volume"];
        let roles = [Role::PastCovariate, Role::PastCovariate, Role::PastCovariate, Role::Target, Role::PastCovariate];
        let variates: Vec<Variate> = (0..5)
            .map(|c| Variate {
                name: names[c].into(),
                role: roles[c],
                kind: Kind::Continuous,
                data: ctx.iter().map(|b| b[c]).collect(),
                future: None,
                observed: None,
                cardinality: None,
            })
            .collect();
        Panel { freq: "1d".into(), start: None, items: vec![Item::new("X", variates)] }
    }
}

/// Extract the point (median/mean) close path from a forecast's first target.
fn point_path(fc: &Forecast) -> Option<Vec<f32>> {
    let tf = fc.targets.first()?;
    if let Some(m) = &tf.mean {
        return Some(m.data.clone());
    }
    None
}

struct Rec {
    o: usize,
    date: String,
    ticker: String,
    pred: f32,
    real: f32,
}

struct ModelOut {
    recs: Vec<Rec>,
    mase_sum: f64,
    mase_n: usize,
    lat_ms: Vec<f32>, // per-forecast latency, warmup excluded
}

#[test]
fn oos_skill_eval() {
    let Ok(data) = std::env::var("OOS_DATA") else {
        eprintln!("OOS_DATA unset; skipping OOS skill eval");
        return;
    };
    let Ok(out) = std::env::var("OOS_OUT") else {
        eprintln!("OOS_OUT unset; skipping OOS skill eval");
        return;
    };
    let ctx_len = envu("OOS_CTX", 200);
    let horizon = envu("OOS_HORIZON", 5);
    let step = envu("OOS_STEP", 5);
    let nsamples = envu("OOS_NSAMPLES", 16);
    let max_orig = envu("OOS_MAXORIG", 0);
    let warmup = envu("OOS_WARMUP", 2);
    let latency_only = envu("OOS_LATENCY_ONLY", 0) != 0;
    let start = envs("OOS_START", "2026-01-01");
    let device = envs("BRAIN_DEVICE", "gpu");
    let start_key: (i32, u32, u32) = {
        let p: Vec<i64> = start.split('-').filter_map(|x| x.parse().ok()).collect();
        (p[0] as i32, p[1] as u32, p[2] as u32)
    };

    let models = build_models();
    if models.is_empty() {
        eprintln!("no models loaded (set CHRONOS2_WEIGHTS / KRONOS_*_DIR / FINCAST_WEIGHTS); skipping");
        return;
    }

    // universe: all *.csv in OOS_DATA (drop any ^index).
    let mut tickers: Vec<String> = std::fs::read_dir(&data)
        .expect("read OOS_DATA")
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_string_lossy().strip_suffix(".csv").map(|s| s.to_string()))
        .filter(|t| !t.starts_with('^') && !t.eq_ignore_ascii_case("gspc"))
        .collect();
    tickers.sort();
    let series: Vec<Series> = tickers.iter().filter_map(|t| load_csv(&format!("{data}/{t}.csv"), t)).collect();
    assert!(series.len() >= 3, "need >=3 tickers, got {}", series.len());
    eprintln!("universe: {} names; device={device}; ctx={ctx_len} horizon={horizon} step={step} nsamp={nsamples}", series.len());

    // origin dates: distinct last-observed dates >= start, weekly (every `step`
    // trading bars of the calendar), leaving room for the horizon. Built from the
    // union of all names' dates so the cross-section is as wide as possible.
    let mut all_dates: Vec<(i32, u32, u32)> = Vec::new();
    for s in &series {
        for d in &s.dates {
            all_dates.push(*d);
        }
    }
    all_dates.sort();
    all_dates.dedup();
    let oos_dates: Vec<(i32, u32, u32)> = all_dates.into_iter().filter(|d| *d >= start_key).collect();
    // weekly stride
    let mut origin_dates: Vec<(i32, u32, u32)> = oos_dates.iter().step_by(step).copied().collect();
    if max_orig > 0 {
        origin_dates.truncate(max_orig);
    }
    eprintln!("origin weeks in OOS window: {}", origin_dates.len());

    // fast date -> index per series
    let idx_of: Vec<std::collections::HashMap<(i32, u32, u32), usize>> = series
        .iter()
        .map(|s| s.dates.iter().enumerate().map(|(i, d)| (*d, i)).collect())
        .collect();

    let mut outs: Vec<ModelOut> =
        models.iter().map(|_| ModelOut { recs: Vec::new(), mase_sum: 0.0, mase_n: 0, lat_ms: Vec::new() }).collect();

    let spec = ForecastSpec {
        horizon,
        representations: vec![Representation::Point, Representation::Quantiles],
        quantile_levels: vec![0.1, 0.5, 0.9],
        num_samples: nsamples,
        seed: 12345,
    };

    let t0 = Instant::now();

    // latency-only fast path: warmup + a handful of timed forecasts per model.
    let n_origins = if latency_only { origin_dates.len().min(warmup + 12) } else { origin_dates.len() };

    // per-model warmup counter (exclude first `warmup` forecasts from latency stats)
    let mut done_count = vec![0usize; models.len()];

    let write_json = |outs: &[ModelOut], n_done: usize| {
        let mut js = String::from("{\n");
        let _ = write!(
            js,
            "  \"meta\": {{\"ctx\": {ctx_len}, \"horizon\": {horizon}, \"step\": {step}, \"nsamples\": {nsamples}, \"n_names\": {}, \"n_origins\": {n_done}, \"device\": \"{device}\", \"start\": \"{start}\"}},\n",
            series.len()
        );
        js.push_str("  \"models\": {\n");
        for (mi, (name, _)) in models.iter().enumerate() {
            let o = &outs[mi];
            let lat = &o.lat_ms;
            let (mean_ms, med_ms, min_ms, p90_ms) = lat_stats(lat);
            let mase = if o.mase_n > 0 { o.mase_sum / o.mase_n as f64 } else { f64::NAN };
            let comma = if mi + 1 < models.len() { "," } else { "" };
            let _ = write!(
                js,
                "    \"{name}\": {{\n      \"latency_ms\": {{\"mean\": {mean_ms:.3}, \"median\": {med_ms:.3}, \"min\": {min_ms:.3}, \"p90\": {p90_ms:.3}, \"n\": {}}},\n      \"mase_mean\": {},\n      \"records\": [\n",
                lat.len(),
                if mase.is_nan() { "null".to_string() } else { format!("{mase:.6}") },
            );
            for (i, r) in o.recs.iter().enumerate() {
                let c = if i + 1 < o.recs.len() { "," } else { "" };
                let _ = write!(
                    js,
                    "        {{\"o\": {}, \"date\": \"{}\", \"ticker\": \"{}\", \"pred\": {:.6}, \"real\": {:.6}}}{c}\n",
                    r.o, r.date, r.ticker, r.pred, r.real
                );
            }
            js.push_str(&format!("      ]\n    }}{comma}\n"));
        }
        js.push_str("  }\n}\n");
        let tmp = format!("{out}.tmp");
        std::fs::write(&tmp, js).expect("write json");
        std::fs::rename(&tmp, &out).expect("rename json");
    };

    for (oi, od) in origin_dates.iter().take(n_origins).enumerate() {
        let date = format!("{:04}-{:02}-{:02}", od.0, od.1, od.2);
        for (si, s) in series.iter().enumerate() {
            let Some(&idx) = idx_of[si].get(od) else { continue };
            // need ctx bars ending at idx, and horizon bars after idx.
            if idx + 1 < ctx_len || idx + horizon >= s.ohlcv.len() {
                continue;
            }
            let ctx = &s.ohlcv[idx + 1 - ctx_len..=idx];
            let last_close = ctx[ctx_len - 1][3];
            let real_close = s.ohlcv[idx + horizon][3];
            let real_ret = real_close / last_close - 1.0;
            let ins_close: Vec<f32> = ctx.iter().map(|b| b[3]).collect();
            let real_path: Vec<f32> = (1..=horizon).map(|h| s.ohlcv[idx + h][3]).collect();

            for (mi, (_name, model)) in models.iter().enumerate() {
                let panel = build_panel(model.as_ref(), ctx);
                let t = Instant::now();
                let fc = match model.forecast(&panel, &spec) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("[{}] forecast err {si}/{}: {}", _name, s.ticker, e.message);
                        continue;
                    }
                };
                let ms = t.elapsed().as_secs_f32() * 1000.0;
                done_count[mi] += 1;
                if done_count[mi] > warmup {
                    outs[mi].lat_ms.push(ms);
                }
                let Some(path) = point_path(&fc) else { continue };
                if path.len() < horizon {
                    continue;
                }
                let pred_close = path[horizon - 1];
                let pred_ret = pred_close / last_close - 1.0;
                if !latency_only {
                    // point MASE over the horizon vs naive last-value scale.
                    let m = forecast::metrics::mase(&path[..horizon], &real_path, &ins_close, 1);
                    if m.is_finite() {
                        outs[mi].mase_sum += m as f64;
                        outs[mi].mase_n += 1;
                    }
                    outs[mi].recs.push(Rec { o: oi, date: date.clone(), ticker: s.ticker.clone(), pred: pred_ret, real: real_ret });
                }
            }
        }
        if !latency_only {
            write_json(&outs, oi + 1);
        }
        eprintln!("[{}/{}] week {date} done ({:.1}s elapsed)", oi + 1, n_origins, t0.elapsed().as_secs_f32());
    }
    write_json(&outs, n_origins);
    for (mi, (name, _)) in models.iter().enumerate() {
        let (mean_ms, med_ms, _, _) = lat_stats(&outs[mi].lat_ms);
        eprintln!(
            "{name}: {} recs, mase {:.4}, latency mean {mean_ms:.1}ms median {med_ms:.1}ms (n={})",
            outs[mi].recs.len(),
            if outs[mi].mase_n > 0 { outs[mi].mase_sum / outs[mi].mase_n as f64 } else { f64::NAN },
            outs[mi].lat_ms.len()
        );
    }
    eprintln!("wrote {out} ({:.1}s total)", t0.elapsed().as_secs_f32());
}

fn lat_stats(lat: &[f32]) -> (f32, f32, f32, f32) {
    if lat.is_empty() {
        return (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }
    let mean = lat.iter().sum::<f32>() / lat.len() as f32;
    let mut s = lat.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = s[s.len() / 2];
    let min = s[0];
    let p90 = s[((s.len() as f32 * 0.9) as usize).min(s.len() - 1)];
    (mean, med, min, p90)
}
