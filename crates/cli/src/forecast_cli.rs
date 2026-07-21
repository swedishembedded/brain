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
//! - `import --hf <dir> --out chronos2.weights` converts an `amazon/chronos-2`
//!   checkpoint into a brain `.weights` container.

use crate::args::Args;
use std::sync::Arc;

pub fn run_forecast(argv: &[String]) {
    match argv.first().map(|s| s.as_str()) {
        Some("compare") => compare(&argv[1..]),
        Some("serve") => serve(&argv[1..]),
        Some("import") => import(&argv[1..]),
        other => {
            eprintln!("usage: brain forecast <compare|serve|import> ...  (got {other:?})");
        }
    }
}

/// Where the foundation models' weights live, so the registry can load them.
#[derive(Clone, Default)]
struct FmPaths {
    chronos2: Option<String>,
    kronos_tokenizer: Option<String>,
    kronos_decoder: Option<String>,
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
    reg
}

/// `brain forecast import --hf <dir> --out chronos2.weights`.
fn import(args: &[String]) {
    let mut a = Args::new(args);
    let hf = a.str_or("--hf", "");
    let out = a.str_or("--out", "chronos2.weights");
    a.finish();
    if hf.is_empty() {
        eprintln!("usage: brain forecast import --hf <amazon/chronos-2 dir> --out chronos2.weights");
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
    let max_conn = a.usize_or("--max-connections", 64);
    a.finish();

    let fm = FmPaths {
        chronos2: chronos2.clone(),
        kronos_tokenizer: kronos_tok,
        kronos_decoder: kronos_dec,
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
