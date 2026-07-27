// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain run` / `brain serve` — the event-driven stdio controller loop.
//!
//! Reads JSONL [`events::Event`] lines from stdin (a blocking read is the idle
//! wait), feeds each to a [`runtime::Controller`], and writes every emitted event
//! back as a JSONL line to stdout (flushed per line). Diagnostics go to stderr.
//!
//! Flags:
//!   * `--gpt <path>` (or env `BRAIN_GPT`) — load a GPT checkpoint as the text
//!     model. With none, a fake echo model is used so the loop is testable
//!     without a trained model.
//!   * `--yolo <path>` (or env `BRAIN_YOLO`) — load a YOLO checkpoint as the
//!     object detector. With none, a `FakeDetectModel` returns a fixed box so the
//!     loop runs without a trained detector.
//!   * `--conf <f32>` (or env `BRAIN_CONF`) — detection confidence threshold for
//!     the YOLO detector (default 0.25). Lower it so a lightly-trained tiny model's
//!     low-confidence boxes still surface. No effect on the fake detector.
//!   * `--max-new N`, `--temp X`, `--top-k K`, `--seed S` — generation config.

use std::io::{BufRead, Write};

use events::Envelope;
use runtime::{
    Controller, DetectModel, Emit, FakeDetectModel, FakeInferModel, GenConfig, GptInfer, Registry,
    YoloDetect,
};

/// A live [`Emit`] sink over a stdout writer: encodes each envelope to a JSONL
/// line and flushes it immediately, so `brain run` streams token-by-token as the
/// controller produces them (not one batch at the end of the turn). `ok` latches
/// false once the pipe closes so the loop can stop.
struct StdoutSink<'a, W: Write> {
    w: &'a mut W,
    ok: bool,
}

impl<W: Write> Emit for StdoutSink<'_, W> {
    fn emit(&mut self, env: Envelope) {
        if !self.ok {
            return;
        }
        if writeln!(self.w, "{}", events::encode_envelope(&env)).is_err() {
            self.ok = false;
            return;
        }
        let _ = self.w.flush();
    }
}

pub fn run_serve(args: &[String]) {
    let mut gpt_path = std::env::var("BRAIN_GPT").ok();
    let mut yolo_path = std::env::var("BRAIN_YOLO").ok();
    let mut cfg = GenConfig { max_new: 256, temperature: 0.0, top_k: 0, eos: None, seed: 0 };
    // Optional detection confidence threshold for the YOLO detector. A tiny model
    // trained for only a few hundred steps emits low-confidence boxes that the
    // default 0.25 filter would drop, so the demo can lower it (also `BRAIN_CONF`).
    let mut conf: Option<f32> =
        std::env::var("BRAIN_CONF").ok().and_then(|s| s.parse().ok());
    // D-Bus control surface (`--dbus [--dbus-system] [--dbus-name NAME]`).
    let (mut dbus, mut dbus_system, mut dbus_name) = (false, false, None::<String>);
    let mut dbus_reserve_gb: u64 = 2; // GB kept free per GPU (headroom for activations)

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gpt" => {
                i += 1;
                gpt_path = args.get(i).cloned();
            }
            "--yolo" => {
                i += 1;
                yolo_path = args.get(i).cloned();
            }
            "--max-new" => {
                i += 1;
                cfg.max_new = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.max_new);
            }
            "--temp" | "--temperature" => {
                i += 1;
                cfg.temperature =
                    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.temperature);
            }
            "--top-k" => {
                i += 1;
                cfg.top_k = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.top_k);
            }
            "--seed" => {
                i += 1;
                cfg.seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.seed);
            }
            "--conf" => {
                i += 1;
                conf = args.get(i).and_then(|s| s.parse().ok()).or(conf);
            }
            "--dbus" => dbus = true,
            "--dbus-system" => {
                dbus = true;
                dbus_system = true;
            }
            "--dbus-name" => {
                i += 1;
                dbus_name = args.get(i).cloned();
            }
            "--reserve-gb" => {
                i += 1;
                dbus_reserve_gb = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(dbus_reserve_gb);
            }
            other => eprintln!("brain run: ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    // The D-Bus control surface replaces the stdio loop when requested: it serves
    // every registered model over `com.swedishembedded.Brain1` until Ctrl-C.
    if dbus {
        return run_dbus(dbus_system, dbus_name, dbus_reserve_gb);
    }

    // Build the registry: a real GPT if a checkpoint was given, else a fake echo
    // model so the loop runs end-to-end without a trained model.
    let infer: Box<dyn runtime::InferModel> = match &gpt_path {
        Some(path) => {
            eprintln!("brain run: loading GPT checkpoint {path}");
            // Char models embed itos; the pump uses it for the EOS-less stop at
            // max_new. We leave eos as configured (None unless the user sets one).
            Box::new(GptInfer::load(path))
        }
        None => {
            eprintln!("brain run: no --gpt checkpoint; using fake echo model");
            // The fake echoes a fixed greeting and terminates at its EOS sentinel.
            cfg.eos = Some(256);
            Box::new(FakeInferModel::echoing("hello from brain"))
        }
    };
    // A real YOLO if a checkpoint was given, else the fixed-box fake detector.
    let detect: Box<dyn DetectModel> = match &yolo_path {
        Some(path) => {
            eprintln!("brain run: loading YOLO checkpoint {path}");
            let mut det = YoloDetect::load(path);
            if let Some(c) = conf {
                eprintln!("brain run: detection confidence threshold {c}");
                // Keep the default IoU (0.45); only override the confidence gate.
                det = det.with_thresholds(c, 0.45);
            }
            Box::new(det)
        }
        None => {
            eprintln!("brain run: no --yolo checkpoint; using fake detector");
            Box::new(FakeDetectModel::default())
        }
    };

    let mut ctrl = Controller::with_config(Registry::with_models(infer, detect), cfg);

    // Expose the generic capability providers over the event API (manifest_request
    // / action_request) — the same actions `brain do` runs, now network-reachable.
    ctrl.register_provider(std::sync::Arc::new(zimage::caps::ZImageProvider::load().expect("z-image provider")));

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Announce readiness.
    let _ = writeln!(out, "{}", events::encode_line(&events::Event::Ready));
    let _ = out.flush();

    // Blocking line read = idle wait. EOF (None) ends the loop.
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("brain run: stdin error: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        // Stream each emitted envelope to stdout as it is produced (flushed per
        // line), so a long chat response appears token-by-token rather than all at
        // once when the turn completes. The req_id (if any) is echoed on every line
        // for client-side demuxing. No control source on stdin's blocking read: a
        // `cancel` is honored as the next line (recoverable), between turns.
        let mut sink = StdoutSink { w: &mut out, ok: true };
        ctrl.feed_line_streaming(&line, &mut sink, &mut ());
        if !sink.ok {
            return; // stdout closed
        }
    }
}

/// Serve the D-Bus control surface (`brain serve --dbus`). Registers every model
/// and hands the registry to `brain_dbus::serve`, which owns it for the service's
/// lifetime. Compiled into the default build; only reached when `--dbus` is passed.
fn run_dbus(system: bool, name: Option<String>, reserve_gb: u64) {
    // Discover the GPUs' capacity so the scheduler can budget/evict against real VRAM.
    let gpus = query_gpu_mem();
    if gpus.is_empty() {
        eprintln!("brain serve --dbus: no GPUs detected (nvidia-smi); serving with CPU-only budget");
    }
    let ram = query_ram_bytes();
    let reserved = reserve_gb << 30;
    eprintln!(
        "brain serve --dbus: {} GPU(s), {} GB reserved/card, {} GB RAM budget",
        gpus.len(),
        reserve_gb,
        ram >> 30
    );
    let executor = crate::resident::build_executor(&gpus, reserved, ram, residency::Policy::default());
    let served: Vec<&str> = executor.manifests().iter().map(|m| m.model.as_str()).collect();
    eprintln!("brain serve --dbus: models: {}", served.join(", "));
    let opts = brain_dbus::DbusOpts {
        bus: if system { brain_dbus::BusKind::System } else { brain_dbus::BusKind::Session },
        name: name.unwrap_or_else(|| "com.swedishembedded.Brain1".to_string()),
    };
    if let Err(e) = brain_dbus::serve(executor, opts) {
        eprintln!("brain serve --dbus: {e}");
        std::process::exit(1);
    }
}

/// Per-GPU `(index, total_bytes)` via `nvidia-smi` (empty if none/unavailable).
fn query_gpu_mem() -> Vec<(u32, u64)> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .enumerate()
            .filter_map(|(i, l)| l.trim().parse::<u64>().ok().map(|mib| (i as u32, mib << 20)))
            .collect(),
        _ => Vec::new(),
    }
}

/// Total system RAM in bytes (from `/proc/meminfo`; falls back to 16 GB).
fn query_ram_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("MemTotal:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb << 10)
        .unwrap_or(16 << 30)
}
