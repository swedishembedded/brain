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
        Some("finetune") => finetune(&argv[1..]),
        other => {
            eprintln!("usage: brain forecast <compare|serve|import|finetune> ...  (got {other:?})");
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
    a.finish();
    if tok.is_empty() || dec.is_empty() || data.is_empty() {
        eprintln!("usage: brain forecast finetune --kronos-tokenizer <dir> --kronos-decoder <dir> --data <csv-dir> \\");
        eprintln!("         [--out <ckpt>] [--context 180] [--horizon 5] [--epochs 8] [--lr 4e-5] [--lora RANK] [--embargo N]");
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
    let opts = kronos::train::FinetuneOpts { epochs, lr, wd: 0.1, clip: 3.0, lora, progress: true };
    let (rep, weights) = kronos::finetune::finetune_universe(&model, &base, &series, context, horizon, split, &opts);
    println!(
        "\ngate (INCLUDED names, held-out future): base_val {:.4} → ft_val {:.4}  ({} steps)  ⇒  {}",
        rep.base_val, rep.ft_val, rep.steps,
        if rep.promoted { "PROMOTE (fine-tune beats base out-of-sample)" } else { "KEEP BASE (no held-out improvement)" }
    );
    let w = weights.expect("weights always returned");
    // Save FIRST (so a slow generalization eval can't cost us the checkpoint).
    if rep.promoted {
        let path = out.unwrap_or_else(|| "kronos-decoder-ft.weights".into());
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
