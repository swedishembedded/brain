// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Will an industrial cooling loop trip its over-temperature threshold in the
//! next 5 days - and when? A worked, non-toy forecasting problem showing why
//! this needs a model that can consume covariates natively, not just a state
//! observer.
//!
//! The physics (see `tools/forecast/make_cooling_loop.py`, the one place that
//! generates this data): a heat exchanger's conductance quietly fouls between
//! cleanings, while an unmeasured, shift-schedule-driven heat load pushes the
//! return coolant temperature up. Three forecasts of the SAME held-out window,
//! one chart:
//!
//! - **A physics observer** - the conventional answer. It inverts the SAME
//!   energy balance the generator used to estimate the exchanger's current
//!   conductance from data, then forecasts by holding tomorrow's load at
//!   today's level. It tracks the PRESENT well (that is what an observer is
//!   for) and is systematically wrong about the FUTURE, because persisting
//!   the load ignores the shift schedule that is about to change it.
//! - **Seasonal-naive** - repeat the last day's cycle. Cheap, and a real
//!   baseline to beat.
//! - **TimesFM-3**, natively multivariate: the return temperature (target),
//!   pump power (a past-only covariate - measured, not known in advance) and
//!   BOTH the ambient temperature and the shift schedule (known-future
//!   covariates - a site already has a short-range ambient forecast and its
//!   own production plan) all attend to each other through one decode() call.
//!
//! Usage:
//!   python3 tools/forecast/make_cooling_loop.py --out cooling_loop.csv
//!   cargo run --release -p brain-timesfm3 --example cooling_loop -- \
//!     cooling_loop.csv <timesfm3.safetensors> [chart.png]

use forecast::{ForecastModel, ForecastSpec, Item, Panel, Role, Variate};
use timesfm3::Timesfm3Forecaster;

// Multiples of the checkpoint's 32-step patch length - context/horizon
// lengths that are not are a ledgered gap (build_input has no left-padding
// path yet).
const CONTEXT: usize = 576;
const HORIZON: usize = 128;
const TRIP: f32 = 45.0;

/// The columns `make_cooling_loop.py` writes, minus the timestamp and the
/// (unmeasured-in-reality) `q_load` debug column, which the physics observer
/// below re-derives instead of reading.
struct Series {
    t_return: Vec<f32>,
    t_amb: Vec<f32>,
    pump_power: Vec<f32>,
    shift_on: Vec<f32>,
}

fn read_csv(path: &str) -> Series {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut s = Series { t_return: vec![], t_amb: vec![], pump_power: vec![], shift_on: vec![] };
    for line in text.lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() < 6 {
            continue;
        }
        s.t_return.push(c[1].parse().unwrap());
        s.t_amb.push(c[3].parse().unwrap());
        s.pump_power.push(c[4].parse().unwrap());
        s.shift_on.push(c[5].parse().unwrap());
    }
    s
}

/// The conventional answer: invert the SAME energy balance the generator
/// integrated forward (`C*dT/dt = Q - UA*(T-T_amb)`) to estimate the
/// exchanger's current conductance from the last `WINDOW` hours of data, then
/// forecast by holding both the load and the ambient temperature at their
/// last observed values - an observer has a model of the PLANT, not of what
/// operators are about to do to it or what tomorrow's weather is.
fn physics_observer_forecast(t_return: &[f32], t_amb: &[f32], pump_power: &[f32], horizon: usize) -> Vec<f32> {
    const C: f32 = 40.0;
    const PUMP_TO_LOAD: f32 = 0.35; // the generator's own Q -> pump_power gain, known from the plant's commissioning data
    const WINDOW: usize = 24;
    let n = t_return.len();

    let mut ua_estimates = Vec::with_capacity(WINDOW);
    for t in (n - WINDOW)..n {
        let dtdt = t_return[t] - t_return[t - 1]; // dt = 1h
        let q_hat = pump_power[t] / PUMP_TO_LOAD;
        let denom = t_return[t] - t_amb[t];
        if denom.abs() > 1.0 {
            ua_estimates.push(((q_hat - C * dtdt) / denom).clamp(0.5, 30.0));
        }
    }
    let ua_hat = ua_estimates.iter().sum::<f32>() / ua_estimates.len().max(1) as f32;
    let q_hat = pump_power[n - 1] / PUMP_TO_LOAD;
    let t_amb_hat = t_amb[n - 1];

    let mut out = Vec::with_capacity(horizon);
    let mut t = t_return[n - 1];
    for _ in 0..horizon {
        t += (q_hat - ua_hat * (t - t_amb_hat)) / C;
        out.push(t);
    }
    out
}

/// Repeat the last full day's cycle - the cheap baseline any real forecaster
/// has to beat.
fn seasonal_naive_forecast(t_return: &[f32], horizon: usize) -> Vec<f32> {
    let season = 24;
    let n = t_return.len();
    (0..horizon).map(|h| t_return[n - season + (h % season)]).collect()
}

/// First index (hours from the origin) at which `path` crosses [`TRIP`], or
/// `None` if it never does within the given horizon.
fn trip_hour(path: &[f32]) -> Option<usize> {
    path.iter().position(|&v| v > TRIP)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: cooling_loop <cooling_loop.csv> <timesfm3.safetensors> [chart.png]");
        std::process::exit(2);
    }
    let (csv_path, weights_path) = (&args[1], &args[2]);
    let chart_path = args.get(3);

    let s = read_csv(csv_path);
    assert!(s.t_return.len() >= CONTEXT + HORIZON, "{csv_path}: need at least {} hourly bars, got {}", CONTEXT + HORIZON, s.t_return.len());
    let (ctx, actual) = (0..CONTEXT, CONTEXT..CONTEXT + HORIZON);

    // ---- TimesFM-3: one native multivariate decode over all four series ----
    let forecaster = Timesfm3Forecaster::load(weights_path).unwrap_or_else(|e| panic!("load {weights_path}: {e}"));
    let item = Item::new(
        "cooling-loop",
        vec![
            Variate::target("t_return", s.t_return[ctx.clone()].to_vec()),
            {
                let mut v = Variate::target("pump_power", s.pump_power[ctx.clone()].to_vec());
                v.role = Role::PastCovariate;
                v
            },
            {
                let mut v = Variate::target("t_amb", s.t_amb[ctx.clone()].to_vec());
                v.role = Role::KnownFuture;
                v.future = Some(s.t_amb[actual.clone()].to_vec());
                v
            },
            {
                let mut v = Variate::target("shift_on", s.shift_on[ctx.clone()].to_vec());
                v.role = Role::KnownFuture;
                v.future = Some(s.shift_on[actual.clone()].to_vec());
                v
            },
        ],
    );
    let panel = Panel::single("1h", "cooling-loop", item.variates);
    let spec = ForecastSpec { horizon: HORIZON, quantile_levels: vec![0.1, 0.5, 0.9], ..ForecastSpec::default() };
    let out = forecaster.forecast(&panel, &spec).unwrap_or_else(|e| panic!("forecast: {}", e.message));
    let tf = out.targets.into_iter().find(|t| t.name == "t_return").expect("t_return in the forecast");
    let q = tf.quantiles.as_ref().expect("native quantiles");
    let n_levels = tf.levels.len();
    let median_col = tf.levels.iter().position(|&l| (l - 0.5).abs() < 1e-6).unwrap();
    let timesfm3_path: Vec<f32> = (0..HORIZON).map(|h| q.data[h * n_levels + median_col]).collect();

    // ---- the two baselines, over the SAME held-out window ----
    let observer_path = physics_observer_forecast(&s.t_return[..CONTEXT], &s.t_amb[..CONTEXT], &s.pump_power[..CONTEXT], HORIZON);
    let seasonal_path = seasonal_naive_forecast(&s.t_return[..CONTEXT], HORIZON);
    let actual_path = &s.t_return[actual.clone()];

    // ---- the numbers that matter: when does the trip actually happen, and did each forecast see it coming? ----
    let mae = |p: &[f32]| p.iter().zip(actual_path).map(|(a, b)| (a - b).abs()).sum::<f32>() / p.len() as f32;
    let report = |name: &str, path: &[f32]| {
        let hour = trip_hour(path).map(|h| format!("hour {h}")).unwrap_or_else(|| "never".to_string());
        println!("  {name:<24} MAE {:>6.2}   predicted trip: {hour}", mae(path));
    };
    println!("cooling loop: {CONTEXT}h context -> {HORIZON}h held-out forecast, trip threshold {TRIP} C");
    println!("  {:<24}                predicted trip: {}", "actual", trip_hour(actual_path).map(|h| format!("hour {h}")).unwrap_or_else(|| "never".to_string()));
    report("physics observer", &observer_path);
    report("seasonal naive", &seasonal_path);
    report("timesfm3", &timesfm3_path);

    // ---- one chart, three forecasts against the truth ----
    if let Some(path) = chart_path {
        let show = 4 * HORIZON;
        let first = CONTEXT.saturating_sub(show);
        let mut chart = forecast::chart::ForecastChart::new(format!("cooling loop: {HORIZON}h return-temperature forecast vs held-out actual"));
        chart.y_label = "return temp (C)".to_string();
        chart.forecast_label = "timesfm3 forecast".to_string();
        chart.history = (first..CONTEXT).map(|i| ((i - first) as f64, s.t_return[i] as f64)).collect();
        let origin = (CONTEXT - first) as f64 - 1.0;
        let last = s.t_return[CONTEXT - 1] as f64;
        chart.forecast = std::iter::once((origin, last)).chain(timesfm3_path.iter().enumerate().map(|(h, v)| (origin + 1.0 + h as f64, *v as f64))).collect();
        chart.actual = std::iter::once((origin, last)).chain(actual_path.iter().enumerate().map(|(h, v)| (origin + 1.0 + h as f64, *v as f64))).collect();
        chart.band = (0..HORIZON)
            .map(|h| (origin + 1.0 + h as f64, q.data[h * n_levels] as f64, q.data[h * n_levels + n_levels - 1] as f64))
            .collect();
        chart.extra_lines.push((
            "physics observer".to_string(),
            std::iter::once((origin, last)).chain(observer_path.iter().enumerate().map(|(h, v)| (origin + 1.0 + h as f64, *v as f64))).collect(),
        ));
        chart.extra_lines.push((
            "seasonal naive".to_string(),
            std::iter::once((origin, last)).chain(seasonal_path.iter().enumerate().map(|(h, v)| (origin + 1.0 + h as f64, *v as f64))).collect(),
        ));
        match forecast::chart::render_png(&chart, std::path::Path::new(path)) {
            Ok(p) => println!("  chart: {}", p.display()),
            Err(e) if !forecast::chart::gnuplot_available() => println!("  chart skipped: {} ({})", forecast::chart::INSTALL_HINT, e),
            Err(e) => panic!("{e}"),
        }
    }
}
