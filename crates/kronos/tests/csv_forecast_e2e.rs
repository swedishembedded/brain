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
//! The committed example series is the input on purpose: it has a known
//! predictable structure and a known irreducible noise floor (see
//! `tools/forecast/make_synthetic_ohlcv.py`), so "did the forecast work" has an
//! answer that is not a matter of taste.

use forecast::ForecastModel;

/// The committed example series, resolved repo-relative (an in-repo artifact,
/// never an absolute machine path).
const CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/forecast/synthetic_hourly.csv");

/// The scored configuration. Small enough that the gate stays under a minute
/// on CPU, large enough that the claim below is a measurement: 4 disjoint
/// held-out windows, not one draw.
const CONTEXT: usize = 512;
const HORIZON: usize = 12;
const ORIGINS: usize = 4;
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

/// The end-to-end claim: real weights, the real CSV, and a forecast that beats
/// the random-walk baseline over several disjoint held-out windows.
///
/// Scored on MEAN MAE across `ORIGINS` origins rather than one, because a
/// single origin is a draw - Kronos wins some and loses some, and a gate built
/// on one window would flap. The threshold is deliberately loose (strictly
/// better on the mean, and better at a majority of origins): this asserts that
/// the model is connected to its input and is extracting real structure, not
/// that it hits a particular number.
#[test]
fn forecasting_the_example_csv_beats_persistence_over_rolling_origins() {
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

    let (mut kronos_mae, mut naive_mae, mut wins) = (Vec::new(), Vec::new(), 0usize);
    for o in 0..ORIGINS {
        let split = s.split_at_origin(CONTEXT, HORIZON, o * HORIZON).expect("split");
        let panel = forecast::csv::panel(&split, "SYN", "1h");
        let out = model.forecast(&panel, &spec).expect("forecast");
        let tf = out.targets.iter().find(|t| t.name == "close").expect("a close target");

        let q = tf.quantiles.as_ref().expect("quantiles derived from the samples");
        assert_eq!(q.shape, vec![HORIZON, 3], "origin {o}");
        assert!(q.data.iter().all(|v| v.is_finite()), "origin {o}: a non-finite forecast value");
        let pred: Vec<f32> = (0..HORIZON).map(|h| q.data[h * 3 + 1]).collect(); // the median level

        let actual: Vec<f32> = split.actual.iter().map(|b| b.ohlcv[forecast::csv::CLOSE]).collect();
        let last = split.context.last().unwrap().ohlcv[forecast::csv::CLOSE];
        // Sanity before skill: a rollout that has come loose from its input
        // wanders off the price scale entirely, and every error metric below
        // would still be "a number".
        assert!(pred.iter().all(|p| *p > last * 0.5 && *p < last * 2.0), "origin {o}: forecast left the price scale: {pred:?}");
        // The quantile levels must be ordered at every step, or "the median"
        // is not the median.
        for h in 0..HORIZON {
            assert!(q.data[h * 3] <= q.data[h * 3 + 1] && q.data[h * 3 + 1] <= q.data[h * 3 + 2], "origin {o} step {h}: quantiles out of order");
        }

        let k = forecast::metrics::mae(&pred, &actual);
        let n = forecast::metrics::mae(&[last; HORIZON], &actual);
        if k < n {
            wins += 1;
        }
        eprintln!("origin {o}: kronos MAE {k:.4}  persistence MAE {n:.4}");
        kronos_mae.push(k);
        naive_mae.push(n);
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (k, n) = (mean(&kronos_mae), mean(&naive_mae));
    eprintln!("mean over {ORIGINS} origins: kronos {k:.4} vs persistence {n:.4} ({:+.1}%), better at {wins}/{ORIGINS}", (1.0 - k / n) * 100.0);
    assert!(k < n, "kronos mean MAE {k:.4} is not better than persistence {n:.4} -- the model is not extracting the structure this series was built to carry");
    assert!(wins * 2 > ORIGINS, "kronos beat persistence at only {wins}/{ORIGINS} origins -- a mean win carried by one lucky window is not skill");
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
