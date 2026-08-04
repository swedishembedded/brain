// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cross-sectional RankIC backtest for Kronos — the evaluation the paper actually
//! reports (a ranking/alpha metric), not single-name absolute-price accuracy.
//!
//! At each rolling origin, for every stock in the universe we forecast the K-bar
//! -ahead close from `CTX` bars of real OHLCV(+real calendar), form the predicted
//! K-bar return, and record it beside the realized return. A separate step
//! (`tools`/python) ranks the cross-section per origin and computes RankIC (Spearman),
//! IC (Pearson), a cost-aware long/short portfolio, and a shuffled negative control.
//!
//! In-process (loads weights once, warm across the whole sweep) so it is immune to
//! the host's client/server flakiness. Env-gated; skips without checkpoints + data.
//!
//! Env:
//!   KRONOS_TOKENIZER_DIR, KRONOS_DECODER_DIR  — checkpoints (required)
//!   RANKIC_DATA   — dir of `<TICKER>.csv` (Date,open,high,low,close,volume)
//!   RANKIC_OUT    — output JSON path
//!   RANKIC_TICKERS — comma list, or "all" (default: all *.csv in RANKIC_DATA)
//!   RANKIC_CTX=200  RANKIC_HORIZON=5  RANKIC_STEP=10  RANKIC_START=<auto>
//!   RANKIC_ARGMAX=1 (modal path; else sampled)  RANKIC_NSAMPLES=1  RANKIC_MAXORIG=0(=all)

use kronos::{GenOpts, KronosModel};
use std::fmt::Write as _;

/// Howard Hinnant days-from-civil; weekday Monday=0 (matches pandas .weekday()).
fn weekday(y: i64, m: i64, d: i64) -> u32 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    (((days % 7) + 3 + 7) % 7) as u32
}

struct Series {
    ticker: String,
    dates: Vec<(i64, i64, i64)>,
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
                dates.push((d[0], d[1], d[2]));
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

/// Forecast the K-bar-ahead close return for one context window. `ctx` is the
/// OHLCV slice `[o-CTX, o)`, `fut_dates` the K future bar dates (for the calendar).
/// Returns predicted (close[o+K-1]/close[o-1] - 1).
fn predict_return(
    model: &KronosModel,
    ctx: &[[f32; 5]],
    cdates: &[(i64, i64, i64)],
    fdates: &[(i64, i64, i64)],
    horizon: usize,
    argmax: bool,
    nsamples: usize,
    seed0: u64,
) -> f32 {
    let feat = model.feat();
    let n = ctx.len();
    // 6-col bars: open,high,low,close,volume,amount(=vol*mean(OHLC))
    let mut bars = vec![0.0f32; n * feat];
    for r in 0..n {
        for c in 0..5 {
            bars[r * feat + c] = ctx[r][c];
        }
        if feat == 6 {
            let mean_ohlc = (ctx[r][0] + ctx[r][1] + ctx[r][2] + ctx[r][3]) * 0.25;
            bars[r * feat + 5] = ctx[r][4] * mean_ohlc;
        }
    }
    // real calendar stamps [minute,hour,weekday,day,month]
    let stamp_of = |d: &(i64, i64, i64)| [0u32, 0, weekday(d.0, d.1, d.2), d.2 as u32, d.1 as u32];
    let mut ctx_stamp = vec![0u32; n * 5];
    for (r, d) in cdates.iter().enumerate() {
        ctx_stamp[r * 5..r * 5 + 5].copy_from_slice(&stamp_of(d));
    }
    let mut fut_stamp = vec![0u32; horizon * 5];
    for (h, d) in fdates.iter().take(horizon).enumerate() {
        fut_stamp[h * 5..h * 5 + 5].copy_from_slice(&stamp_of(d));
    }
    let last_close = ctx[n - 1][3];
    let mut acc = 0.0f32;
    let draws = if argmax { 1 } else { nsamples.max(1) };
    for k in 0..draws {
        let opts = GenOpts {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            argmax,
            seed: seed0.wrapping_add(k as u64),
        };
        let path = model.forecast_cached(&bars, &ctx_stamp, &fut_stamp, horizon, &opts);
        acc += path[(horizon - 1) * feat + 3]; // predicted close at h=K-1
    }
    let pred_close = acc / draws as f32;
    pred_close / last_close - 1.0
}

#[test]
fn rankic_backtest() {
    let (Ok(tok), Ok(dec)) = (std::env::var("KRONOS_TOKENIZER_DIR"), std::env::var("KRONOS_DECODER_DIR")) else {
        eprintln!("KRONOS_{{TOKENIZER,DECODER}}_DIR unset; skipping RankIC backtest");
        return;
    };
    let Ok(data) = std::env::var("RANKIC_DATA") else {
        eprintln!("RANKIC_DATA unset; skipping RankIC backtest");
        return;
    };
    let out = std::env::var("RANKIC_OUT")
        .unwrap_or_else(|_| std::env::temp_dir().join("rankic.json").to_string_lossy().into_owned());
    let ctx_len = envu("RANKIC_CTX", 200);
    let horizon = envu("RANKIC_HORIZON", 5);
    let step = envu("RANKIC_STEP", 10);
    let argmax = envu("RANKIC_ARGMAX", 1) != 0;
    let nsamples = envu("RANKIC_NSAMPLES", 1);
    let max_orig = envu("RANKIC_MAXORIG", 0);

    // universe
    let tickers: Vec<String> = match std::env::var("RANKIC_TICKERS") {
        Ok(s) if s != "all" => s.split(',').map(|x| x.trim().to_string()).collect(),
        _ => {
            let mut v: Vec<String> = std::fs::read_dir(&data)
                .expect("read data dir")
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    n.strip_suffix(".csv").map(|s| s.to_string())
                })
                .filter(|t| t != "QQQ") // drop the index ETF from the cross-section
                .collect();
            v.sort();
            v
        }
    };
    let series: Vec<Series> = tickers
        .iter()
        .filter_map(|t| load_csv(&format!("{data}/{t}.csv"), t))
        .collect();
    assert!(series.len() >= 3, "need >=3 tickers, got {}", series.len());
    let n_bars = series.iter().map(|s| s.ohlcv.len()).min().unwrap();
    eprintln!("universe: {} tickers, {} bars each", series.len(), n_bars);

    let start = envu("RANKIC_START", n_bars.saturating_sub(260).max(ctx_len));
    let last_origin = n_bars - horizon; // need close[o+H-1] to exist => o <= n-H
    let mut origins: Vec<usize> = (start..=last_origin).step_by(step).collect();
    if max_orig > 0 {
        origins.truncate(max_orig);
    }
    eprintln!(
        "origins: {} (start={start} step={step} horizon={horizon} ctx={ctx_len} argmax={argmax} nsamp={nsamples})",
        origins.len()
    );

    let model = kronos::import::load_model(&tok, &dec).expect("load kronos");
    let t_start = std::time::Instant::now();

    // Serialize the accumulated records to `out`. Called after EVERY origin (not
    // just at the end) so a run killed by a wall-clock timeout — common on a
    // contended box — still leaves a valid, usable JSON of the origins completed
    // so far. `n_origins` reflects origins actually written, so downstream tooling
    // can intersect a partial fine-tuned run against a complete base run.
    let write_json = |recs: &[(usize, String, String, f32, f32)], n_orig_done: usize| {
        let mut js = String::from("{\n");
        let _ = writeln!(
            js,
            "  \"meta\": {{\"context\": {ctx_len}, \"horizon\": {horizon}, \"step\": {step}, \"argmax\": {argmax}, \"nsamples\": {nsamples}, \"n_tickers\": {}, \"n_origins\": {n_orig_done}}},",
            series.len(),
        );
        js.push_str("  \"records\": [\n");
        for (i, (o, date, tk, pred, real)) in recs.iter().enumerate() {
            let comma = if i + 1 < recs.len() { "," } else { "" };
            let _ = writeln!(
                js,
                "    {{\"o\": {o}, \"date\": \"{date}\", \"ticker\": \"{tk}\", \"pred\": {pred:.6}, \"real\": {real:.6}}}{comma}"
            );
        }
        js.push_str("  ]\n}\n");
        // atomic-ish: write to a temp then rename so a reader never sees a half-file.
        let tmp = format!("{out}.tmp");
        std::fs::write(&tmp, js).expect("write json");
        std::fs::rename(&tmp, &out).expect("rename json");
    };

    // records: (origin, date, ticker, pred_ret, real_ret)
    let mut recs: Vec<(usize, String, String, f32, f32)> = Vec::new();
    for (oi, &o) in origins.iter().enumerate() {
        let od = &series[0].dates[o - 1];
        let date = format!("{:04}-{:02}-{:02}", od.0, od.1, od.2);
        for (ti, s) in series.iter().enumerate() {
            if o < ctx_len || o + horizon > s.ohlcv.len() {
                continue;
            }
            let ctx = &s.ohlcv[o - ctx_len..o];
            let cdates = &s.dates[o - ctx_len..o];
            let fdates = &s.dates[o..o + horizon];
            let last_close = ctx[ctx_len - 1][3];
            let real = s.ohlcv[o + horizon - 1][3] / last_close - 1.0;
            let seed = (o as u64) << 20 ^ (ti as u64);
            let pred = predict_return(&model, ctx, cdates, fdates, horizon, argmax, nsamples, seed);
            recs.push((o, date.clone(), s.ticker.clone(), pred, real));
        }
        write_json(&recs, oi + 1); // checkpoint after each origin
        eprintln!("[{}/{}] origin {o} ({date}) done ({:.1}s elapsed)", oi + 1, origins.len(), t_start.elapsed().as_secs_f32());
    }
    eprintln!("wrote {} records for {} origins to {out} ({:.1}s total)", recs.len(), origins.len(), t_start.elapsed().as_secs_f32());
}
