// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The user-facing path, end to end: an OHLCV CSV on disk -> a validated
//! `Panel` -> real Kronos weights -> a scored forecast, plus the chart.
//!
//! Everything else in this directory tests an internal invariant (tokenizer
//! parity, KV-cache bit-identity, gradient checks). Those can all hold while
//! `brain forecast predict` is broken, because none of them read a file, build
//! a panel, or compare the result to anything a user would recognise as
//! "right". This test is what makes `make forecast/parity` a gate on the thing
//! people actually run.
//!
//! Two halves, deliberately:
//!
//! - the CSV boundary, which needs **no weights** and therefore always runs -
//!   a defect here is caught on any machine;
//! - the forecast itself, which needs the checkpoints and uses
//!   [`brain_testutil::skip`], so a run that declares `BRAIN_REQUIRE_FIXTURES=1`
//!   gets a RED suite when the weights are missing rather than a green one
//!   that certified nothing.
//!
//! The committed example series is the input on purpose: it is a GARCH random
//! walk with the statistical character of a real tape (see
//! `tools/forecast/make_synthetic_ohlcv.py`), which is the distribution this
//! checkpoint was trained on - so a failure here is a defect in brain rather
//! than the model being shown data it has never met.
//!
//! **What this file can and cannot assert.** On a near-random-walk the
//! conditional mean of the next 12 bars IS very nearly the last close, so
//! nothing beats persistence on point error by a margin that would survive a
//! change of seed - and a gate that demanded one would be a gate on luck. What
//! a working probabilistic forecaster must do instead is emit a spread that is
//! informative, grows with lead time, and brackets reality at a measurable
//! rate. Those are the invariants below.

use forecast::ForecastModel;

/// The committed example series, resolved repo-relative (an in-repo artifact,
/// never an absolute machine path).
const CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/forecast/synthetic_hourly.csv");

/// The scored configuration. Eight disjoint held-out windows rather than four:
/// coverage is strongly correlated WITHIN a window (a forecast either brackets
/// its window or does not), so the effective sample size is the origin count,
/// not the bar count - four origins put the measured coverage anywhere in a
/// range too wide to gate on, run to run. Eight costs about
/// four minutes of CPU and is the smallest count that holds still.
/// `CONTEXT + HORIZON` is exactly the checkpoint's 512-bar attention window, so
/// the KV-cached rollout under test is the regime that is EXACT against the
/// upstream reference (and is what `brain forecast predict` now defaults to).
/// A longer context would slide the window and silently gate an approximation.
const CONTEXT: usize = 500;
const HORIZON: usize = 12;
const ORIGINS: usize = 8;
const SAMPLES: usize = 16;

fn series() -> forecast::OhlcvSeries {
    let text = std::fs::read_to_string(CSV).unwrap_or_else(|e| panic!("read {CSV}: {e}"));
    forecast::parse_ohlcv(&text).unwrap_or_else(|e| panic!("{CSV}: {e}"))
}

/// The committed CSV must stay parseable and long enough for the scored
/// configuration below. No weights, so this runs everywhere - if someone
/// regenerates the example series with fewer bars, this is what says so.
#[test]
fn the_committed_example_csv_parses_and_covers_the_scored_configuration() {
    let s = series();
    assert!(s.len() >= CONTEXT + HORIZON * ORIGINS, "{} bars is too few for {ORIGINS} origins of {CONTEXT}+{HORIZON}", s.len());
    // Hourly, 24/7: consecutive bars differ by exactly one hour, which is what
    // makes the `hour`/`weekday` calendar features meaningful rather than
    // decorative.
    let b = &s.bars;
    assert_eq!((b[1].stamp.hour + 24 - b[0].stamp.hour) % 24, 1);
    let panel = forecast::csv::panel(&s.split(CONTEXT, HORIZON).expect("split"), "SYN", "1h");
    let item = &panel.items[0];
    for name in ["open", "high", "low", "close", "volume", "minute", "hour", "weekday", "day", "month"] {
        let v = item.variate(name).unwrap_or_else(|| panic!("panel has no {name} variate"));
        assert_eq!(v.data.len(), CONTEXT, "{name}");
    }
    // The calendar is a KNOWN-FUTURE covariate: the held-out bars' own stamps
    // are supplied, which is the only reason Kronos can be told what hour it
    // is forecasting.
    assert_eq!(item.variate("hour").unwrap().future.as_ref().map(Vec::len), Some(HORIZON));
}

/// The whole point of validating at the boundary: each of these is a file a
/// user really produces, and each must be rejected HERE with a line number
/// rather than deep inside the tokenizer's normalization. Weight-free.
#[test]
fn a_malformed_csv_is_rejected_at_the_boundary_not_inside_the_model() {
    let good = "timestamp,open,high,low,close,volume\n\
                2026-01-05T00:00:00,100,101,99,100,10\n\
                2026-01-05T01:00:00,100,102,99,101,10\n";
    assert!(forecast::parse_ohlcv(good).is_ok());
    let cases = [
        // a column dropped by a spreadsheet export
        ("timestamp,open,high,low,close\n2026-01-05,100,101,99,100\n", "6"),
        // one ragged row in the middle of a good file
        ("timestamp,open,high,low,close,volume\n2026-01-05T00:00:00,100,101,99,100\n", "line 2"),
        // a NaN from a gap-filled feed
        ("timestamp,open,high,low,close,volume\n2026-01-05T00:00:00,100,NaN,99,100,10\n", "not finite"),
        // rows sorted newest-first, which every naive reader accepts
        ("timestamp,open,high,low,close,volume\n2026-01-05T01:00:00,100,101,99,100,10\n2026-01-05T00:00:00,100,101,99,100,10\n", "not after"),
        // a bar whose high is below its close
        ("timestamp,open,high,low,close,volume\n2026-01-05T00:00:00,100,101,99,105,10\n", "is below open"),
    ];
    for (text, want) in cases {
        let e = forecast::parse_ohlcv(text).unwrap_err();
        assert!(e.contains(want), "error {e:?} does not mention {want:?}");
    }
    // Too short for the request is a SEMANTIC rejection at split time, with
    // the number of rows needed spelled out.
    let short = forecast::parse_ohlcv(good).unwrap();
    assert!(short.split(CONTEXT, HORIZON).unwrap_err().contains("too few"));
}

/// The end-to-end claim: real weights, the real CSV, and a **probabilistic**
/// forecast whose spread is informative, widens with lead time, and brackets
/// the held-out truth at a rate that a collapsed or unanchored rollout cannot
/// reach.
///
/// Four assertions, each of which a real regression breaks and none of which
/// flatters the model:
///
/// 1. **anchored** - the median's first step sits within a few typical bars of
///    the last close. A rollout that has come loose from its context (the
///    detokenization-window class of defect) fails here first.
/// 2. **the ensemble is not degenerate, and it accumulates** - the p10-p90 band
///    has non-zero width at every step and is wider late in the horizon than
///    early. A shared prefill that handed every sample the same RNG stream, or
///    a sampler stuck on the mode, collapses the band to nothing.
/// 3. **the spread is informative** - mean CRPS beats the MAE of the model's
///    OWN median path. CRPS collapses to MAE for a point forecast, so this says
///    the distribution is worth more than the single number drawn from it.
/// 4. **the band brackets reality at a measurable rate** - empirical coverage
///    of the p10-p90 band over every held-out bar sits in a wide but
///    non-trivial range. It is deliberately NOT asserted at that band's
///    nominal coverage level: this
///    checkpoint measurably under-covers as the horizon grows (see the
///    per-origin log this prints), and pinning the gate to the nominal number
///    would be asserting a claim the model does not support.
///
/// What is NOT asserted: that Kronos beats persistence. It does not - not on
/// point error at any horizon, and not on CRPS at this one (its CRPS edge on
/// this series exists only out to about 6 bars). The comparison is printed so a
/// human reading the gate log sees where the model stands; gating on it would
/// be gating on a number the model has no claim to.
#[test]
fn forecasting_the_example_csv_yields_a_calibrated_widening_band() {
    let (Some(tok), Some(dec)) = (env("BRAIN_KRONOS_TOKENIZER"), env("BRAIN_KRONOS_DECODER")) else {
        return brain_testutil::skip("BRAIN_KRONOS_TOKENIZER / BRAIN_KRONOS_DECODER unset; no CSV-to-forecast e2e");
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
    }
    let model = kronos::KronosForecaster::load(&tok, &dec).expect("load kronos");
    let s = series();
    let spec = forecast::ForecastSpec {
        horizon: HORIZON,
        representations: vec![forecast::Representation::Quantiles, forecast::Representation::Point],
        quantile_levels: vec![0.1, 0.5, 0.9],
        num_samples: SAMPLES,
        seed: 7,
    };

    let (mut maes, mut crpss, mut naive_crps) = (Vec::new(), Vec::new(), Vec::new());
    let (mut covered, mut scored) = (0usize, 0usize);
    for o in 0..ORIGINS {
        let split = s.split_at_origin(CONTEXT, HORIZON, o * HORIZON).expect("split");
        let panel = forecast::csv::panel(&split, "SYN", "1h");
        let out = model.forecast(&panel, &spec).expect("forecast");
        let tf = out.targets.iter().find(|t| t.name == "close").expect("a close target");

        let q = tf.quantiles.as_ref().expect("quantiles derived from the samples");
        assert_eq!(q.shape, vec![HORIZON, 3], "origin {o}");
        assert!(q.data.iter().all(|v| v.is_finite()), "origin {o}: a non-finite forecast value");
        let lo: Vec<f32> = (0..HORIZON).map(|h| q.data[h * 3]).collect();
        let pred: Vec<f32> = (0..HORIZON).map(|h| q.data[h * 3 + 1]).collect(); // the median level
        let hi: Vec<f32> = (0..HORIZON).map(|h| q.data[h * 3 + 2]).collect();

        let actual: Vec<f32> = split.actual.iter().map(|b| b.ohlcv[forecast::csv::CLOSE]).collect();
        let ctx: Vec<f32> = split.context.iter().map(|b| b.ohlcv[forecast::csv::CLOSE]).collect();
        let last = *ctx.last().unwrap();
        // The typical size of one bar's move on THIS window - the natural unit
        // for "is the forecast still attached to its input".
        let bar = ctx.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (ctx.len() - 1) as f32;

        // (1) anchored. A rollout that lost its context window does not land
        // near the last close, and every error metric below would still be "a
        // number". 8 typical bars is many times the model's own first-step
        // spread and still far tighter than the old price-scale check it
        // replaced, which was a fraction of the price itself.
        assert!((pred[0] - last).abs() < 8.0 * bar, "origin {o}: first step {} is {:.1} typical bars from the last close {last} -- the rollout is not anchored to its context", pred[0], (pred[0] - last).abs() / bar);
        // The quantile levels must be ordered at every step, or "the median"
        // is not the median.
        for h in 0..HORIZON {
            assert!(lo[h] <= pred[h] && pred[h] <= hi[h], "origin {o} step {h}: quantiles out of order");
        }
        // (2) the ensemble is not degenerate, and uncertainty accumulates.
        let width: Vec<f32> = (0..HORIZON).map(|h| hi[h] - lo[h]).collect();
        assert!(width.iter().all(|w| *w > 0.0), "origin {o}: the 10-90% band has zero width somewhere -- the samples collapsed: {width:?}");
        let quarter = (HORIZON / 4).max(1);
        let early = width[..quarter].iter().sum::<f32>() / quarter as f32;
        let late = width[HORIZON - quarter..].iter().sum::<f32>() / quarter as f32;
        assert!(late > early, "origin {o}: the band does not widen with lead time ({early:.4} -> {late:.4}) -- a rollout that does not accumulate uncertainty is not sampling");

        covered += (0..HORIZON).filter(|&h| actual[h] >= lo[h] && actual[h] <= hi[h]).count();
        scored += HORIZON;
        let samples = tf.samples.as_ref().expect("kronos emits sample trajectories natively");
        let crps = mean_crps(samples, HORIZON, &actual);
        let mae = forecast::metrics::mae(&pred, &actual);
        let naive = forecast::metrics::mae(&[last; HORIZON], &actual);
        eprintln!("origin {o}: kronos CRPS {crps:.4}  own-median MAE {mae:.4}  persistence MAE/CRPS {naive:.4}  band {early:.3}->{late:.3}");
        maes.push(mae);
        crpss.push(crps);
        naive_crps.push(naive);
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (crps, mae, naive) = (mean(&crpss), mean(&maes), mean(&naive_crps));
    let coverage = covered as f32 / scored as f32;
    // Persistence is REPORTED, not gated: on a random walk the honest edge is
    // small and seed-dependent, and a gate on it would be a gate on luck.
    eprintln!("mean over {ORIGINS} origins: CRPS {crps:.4} (persistence {naive:.4}, {:+.1}%)  own-median MAE {mae:.4}  p10-p90 coverage {:.0}% of {scored} bars", (1.0 - crps / naive) * 100.0, coverage * 100.0);

    // (3) the distribution is worth more than the point path drawn from it.
    assert!(crps < mae, "mean CRPS {crps:.4} is not better than the MAE {mae:.4} of the model's own median -- the sampled spread carries no information");
    // (4) the band brackets reality at a measurable rate. The window is wide on
    // purpose: the p10-p90 band's NOMINAL coverage is not what this checkpoint
    // reaches (the band narrows relative to truth as the horizon grows), so the gate
    // pins the regime that was actually measured. Below the floor means a
    // collapsed or misplaced band; at the ceiling the band is so wide it has
    // stopped being a forecast.
    assert!((0.20..0.98).contains(&coverage), "p10-p90 band coverage {coverage:.2} is outside the measured regime [0.20, 0.98)");
}

/// Mean CRPS over the horizon of a `[n_samples, horizon]` sample block.
fn mean_crps(samples: &forecast::Block, horizon: usize, actual: &[f32]) -> f32 {
    let (n, h) = (samples.shape[0], samples.shape[1]);
    let mut acc = 0.0;
    for (t, &y) in actual.iter().enumerate().take(horizon) {
        let col: Vec<f32> = (0..n).map(|k| samples.data[k * h + t]).collect();
        acc += forecast::metrics::crps_ensemble(&col, y);
    }
    acc / horizon as f32
}

/// The chart the Quick start commits, rendered for real from a real forecast.
/// A separate test because it is gated on a MACHINE capability (gnuplot
/// installed), which `BRAIN_REQUIRE_FIXTURES` must not be able to turn into a
/// failure - unlike a missing checkpoint, which it must.
#[test]
fn the_chart_renders_from_a_real_forecast() {
    let (Some(tok), Some(dec)) = (env("BRAIN_KRONOS_TOKENIZER"), env("BRAIN_KRONOS_DECODER")) else {
        return brain_testutil::skip("BRAIN_KRONOS_TOKENIZER / BRAIN_KRONOS_DECODER unset; no chart e2e");
    };
    if !forecast::chart::gnuplot_available() {
        return brain_testutil::skip_unavailable("gnuplot is not installed on this machine");
    }
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
    }
    let model = kronos::KronosForecaster::load(&tok, &dec).expect("load kronos");
    let s = series();
    let split = s.split(CONTEXT, HORIZON).expect("split");
    let spec = forecast::ForecastSpec { horizon: HORIZON, representations: vec![forecast::Representation::Point], num_samples: 4, seed: 7, ..Default::default() };
    let out = model.forecast(&forecast::csv::panel(&split, "SYN", "1h"), &spec).expect("forecast");
    let tf = out.targets.iter().find(|t| t.name == "close").expect("a close target");
    let pred = tf.mean.as_ref().expect("a point path").data.clone();

    let mut chart = forecast::chart::ForecastChart::new("kronos e2e");
    chart.history = split.context.iter().rev().take(HORIZON * 3).rev().enumerate().map(|(i, b)| (i as f64, b.ohlcv[forecast::csv::CLOSE] as f64)).collect();
    let origin = chart.history.len() as f64 - 1.0;
    chart.forecast = pred.iter().enumerate().map(|(h, v)| (origin + 1.0 + h as f64, *v as f64)).collect();
    chart.actual = split.actual.iter().enumerate().map(|(h, b)| (origin + 1.0 + h as f64, b.ohlcv[forecast::csv::CLOSE] as f64)).collect();

    let out_path = std::env::temp_dir().join(format!("brain-kronos-e2e-{}.png", std::process::id()));
    let p = forecast::chart::render_png(&chart, &out_path).expect("render");
    let bytes = std::fs::read(&p).expect("read the png");
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    // The Quick start commits this image, so its size is a repo concern, not
    // just an aesthetic one.
    assert!(bytes.len() > 2000 && bytes.len() < 200_000, "{} bytes", bytes.len());
    std::fs::remove_file(&p).ok();
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}
