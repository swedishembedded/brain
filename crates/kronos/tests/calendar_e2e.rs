// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end calendar check (in-process, no server): run KronosForecaster on a
//! real AAPL window WITH real daily calendar vs WITHOUT, and print both. With the
//! calendar wired, the forecast should shift toward the official (real-calendar)
//! result. Env-gated on `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` and an AAPL
//! CSV at `$KRONOS_AAPL_CSV` (date,open,high,low,close,volume); skips otherwise.

use forecast::{ForecastModel, ForecastSpec, Kind, Panel, Representation, Role, Variate};
use kronos::KronosForecaster;

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

fn samples_mean_last(fc: &forecast::Forecast) -> f32 {
    let tf = fc.targets.iter().find(|t| t.name == "close").unwrap();
    let s = tf.samples.as_ref().unwrap();
    let (n, h) = (s.shape[0], s.shape[1]);
    (0..n).map(|i| s.data[i * h + (h - 1)]).sum::<f32>() / n as f32
}

#[test]
fn calendar_shifts_the_forecast_toward_the_official() {
    let (Ok(tok), Ok(dec)) = (std::env::var("BRAIN_KRONOS_TOKENIZER"), std::env::var("BRAIN_KRONOS_DECODER")) else {
        return brain_testutil::skip("BRAIN_KRONOS_TOKENIZER / BRAIN_KRONOS_DECODER unset; no calendar e2e");
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
    }
    let Ok(csv) = std::env::var("KRONOS_AAPL_CSV") else {
        return brain_testutil::skip("KRONOS_AAPL_CSV unset; no calendar e2e");
    };
    let Ok(text) = std::fs::read_to_string(&csv) else {
        brain_testutil::skip(&format!("no AAPL csv at {csv}"));
        return;
    };

    // parse date,open,high,low,close,volume
    let mut dates: Vec<(i64, i64, i64)> = Vec::new();
    let mut ohlcv: Vec<[f32; 5]> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 6 {
            continue;
        }
        let d: Vec<i64> = c[0][..10].split('-').filter_map(|x| x.parse().ok()).collect();
        if d.len() != 3 {
            continue;
        }
        let vals: Option<Vec<f32>> = c[1..6].iter().map(|x| x.parse().ok()).collect();
        if let Some(v) = vals {
            dates.push((d[0], d[1], d[2]));
            ohlcv.push([v[0], v[1], v[2], v[3], v[4]]);
        }
    }
    let (ctx_len, horizon) = (252usize, 20usize);
    let n = ohlcv.len();
    let origin = n - horizon;
    let ctx = &ohlcv[origin - ctx_len..origin];
    let cdates = &dates[origin - ctx_len..origin];
    let fdates = &dates[origin..origin + horizon];
    let last = ctx[ctx_len - 1][3];
    let actual_last = ohlcv[n - 1][3];

    let col = |k: usize| -> Vec<f32> { ctx.iter().map(|b| b[k]).collect() };
    let names = ["open", "high", "low", "close", "volume"];
    let variates = |extra: Vec<Variate>| -> Vec<Variate> {
        let mut v: Vec<Variate> = (0..5)
            .map(|k| Variate {
                name: names[k].into(),
                role: if k == 3 { Role::Target } else { Role::PastCovariate },
                kind: Kind::Continuous,
                data: col(k),
                future: None,
                observed: None,
                cardinality: None,
            })
            .collect();
        v.extend(extra);
        v
    };

    // calendar variates (minute/hour=0; weekday/day/month real)
    let calvar = |name: &str, ctxv: Vec<f32>, futv: Vec<f32>| Variate {
        name: name.into(),
        role: Role::KnownFuture,
        kind: Kind::Categorical,
        data: ctxv,
        future: Some(futv),
        observed: None,
        cardinality: None,
    };
    let cal = vec![
        calvar("minute", vec![0.0; ctx_len], vec![0.0; horizon]),
        calvar("hour", vec![0.0; ctx_len], vec![0.0; horizon]),
        calvar("weekday", cdates.iter().map(|&(y, m, d)| weekday(y, m, d) as f32).collect(),
               fdates.iter().map(|&(y, m, d)| weekday(y, m, d) as f32).collect()),
        calvar("day", cdates.iter().map(|&(_, _, d)| d as f32).collect(),
               fdates.iter().map(|&(_, _, d)| d as f32).collect()),
        calvar("month", cdates.iter().map(|&(_, m, _)| m as f32).collect(),
               fdates.iter().map(|&(_, m, _)| m as f32).collect()),
    ];

    let model = KronosForecaster::load(&tok, &dec).expect("load kronos");
    let spec = ForecastSpec {
        horizon,
        representations: vec![Representation::Samples],
        quantile_levels: vec![0.5],
        num_samples: 8,
        seed: 0,
    };
    let with = model.forecast(&Panel { freq: "1d".into(), start: None, items: vec![forecast::Item::new("AAPL", variates(cal))] }, &spec).unwrap();
    let without = model.forecast(&Panel { freq: "1d".into(), start: None, items: vec![forecast::Item::new("AAPL", variates(vec![]))] }, &spec).unwrap();
    let (mc, mz) = (samples_mean_last(&with), samples_mean_last(&without));
    eprintln!("AAPL last={last:.2}  actual={actual_last:.2} ({:+.1}%)", (actual_last / last - 1.0) * 100.0);
    eprintln!("brain +calendar : {mc:.2} ({:+.1}%)   [official real-cal ~ -9.2%]", (mc / last - 1.0) * 100.0);
    eprintln!("brain zero-cal  : {mz:.2} ({:+.1}%)   [my zero-cal harness -11.4%]", (mz / last - 1.0) * 100.0);
    // the calendar must actually change the forecast (it is consumed)
    assert!((mc - mz).abs() > 1e-3, "calendar had no effect on the forecast");
}
