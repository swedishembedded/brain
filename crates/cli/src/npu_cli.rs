// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain npu ...` - ONNX graphs on the Intel NPU via OpenVINO.
//!
//!   brain npu check    --onnx M [--ov-device NPU]
//!   brain npu bench    --onnx M [--input S --ov-device NPU --iters N --warmup W --hint latency|throughput ...]
//!
//! `check` decodes a graph and, where OpenVINO and an NPU are present,
//! compiles it; `bench` times one. Each model's own export and benchmark is an
//! action of its sample (`npu-export` and `npu-bench` of `samples/models/lfm2`,
//! `npu-export` of `samples/models/deepseekocr2` and
//! `samples/models/qwen3omnimoe`, `npu-forecast` of the forecast samples, the
//! YOLO export, INT8 quantization and simulation of `samples/models/yolov8`).

use std::path::Path;

use npu::openvino::{NpuConfig, NpuDevice, NpuError, NpuSession, PerfHint};

pub fn run_npu(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("check") => check(&args[1..]),
        Some("bench") => bench(&args[1..]),
        other => eprintln!("usage: brain npu <check|bench> ...  (got {other:?})"),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

/// Shared NPU run/compile options.
struct NpuOpts {
    device: NpuDevice,
    hint: PerfHint,
    cache_dir: Option<String>,
    turbo: bool,
    allow_fallback: bool,
    profiling: bool,
}

impl Default for NpuOpts {
    fn default() -> Self {
        NpuOpts {
            device: NpuDevice::Npu,
            hint: PerfHint::Latency,
            cache_dir: None,
            turbo: false,
            allow_fallback: false,
            profiling: false,
        }
    }
}

impl NpuOpts {
    fn to_config(&self) -> NpuConfig {
        NpuConfig {
            device: self.device,
            perf_hint: self.hint,
            cache_dir: self.cache_dir.as_ref().map(std::path::PathBuf::from),
            turbo: self.turbo,
            tiles: None,
            compilation_params: Some("optimization-level=2 performance-hint-override=latency".into()),
            qdq_opt: true,
            profiling: self.profiling,
            allow_fallback: self.allow_fallback,
        }
    }

    /// Parse a recognised NPU flag at `args[*i]`; returns true if consumed.
    fn parse_flag(&mut self, args: &[String], i: &mut usize) -> bool {
        match args[*i].as_str() {
            "--ov-device" => {
                let d = val(args, i, "--ov-device");
                self.device = NpuDevice::parse(&d).unwrap_or_else(|| {
                    eprintln!("brain npu: --ov-device expects npu|cpu|gpu|auto (got {d:?})");
                    std::process::exit(2);
                });
            }
            // Deprecated alias for `--ov-device` (the OpenVINO target device),
            // kept for ONE release. `main.rs`'s `select_backend` already
            // translates `brain npu … --device X` to `--ov-device X` (with its
            // own deprecation note) before this parser ever runs, so this arm
            // is normally unreachable from the `brain` binary; it stays here
            // for direct callers of `run_npu`/`NpuOpts::parse_flag` (tests,
            // library use) that bypass that translation.
            "--device" => {
                eprintln!(
                    "brain npu: --device is deprecated for the OpenVINO target device; use --ov-device instead"
                );
                let d = val(args, i, "--device");
                self.device = NpuDevice::parse(&d).unwrap_or_else(|| {
                    eprintln!("brain npu: --device expects npu|cpu|gpu|auto (got {d:?})");
                    std::process::exit(2);
                });
            }
            "--hint" => {
                self.hint = match val(args, i, "--hint").as_str() {
                    "throughput" => PerfHint::Throughput,
                    _ => PerfHint::Latency,
                };
            }
            "--cache-dir" => self.cache_dir = Some(val(args, i, "--cache-dir")),
            "--turbo" => self.turbo = true,
            "--allow-fallback" => self.allow_fallback = true,
            "--profile" => self.profiling = true,
            _ => return false,
        }
        true
    }
}

fn die(e: NpuError) -> ! {
    eprintln!("brain npu: {e}");
    std::process::exit(1);
}

// ---- check ----
fn check(args: &[String]) {
    let mut onnx_path = String::new();
    let mut opts = NpuOpts::default();
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--onnx" => onnx_path = val(args, &mut i, "--onnx"),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if onnx_path.is_empty() {
        eprintln!("usage: brain npu check --onnx M [--ov-device NPU]");
        return;
    }
    print!("{}", npu::check::structural_report(&onnx_path, &opts.to_config()));
}

/// Build a `YoloConfig` for decode from explicit flags (the ONNX doesn't carry
/// nc/reg_max). Defaults are yolov8n (nc 80, reg_max 16); input from the model.
fn infer_cfg(session: &NpuSession, nc: Option<u32>, reg_max: Option<u32>, input: Option<u32>) -> yolov8::YoloConfig {
    let mut cfg = yolov8::YoloConfig::yolov8n();
    let s = session.input_shape();
    cfg.input = input.unwrap_or(s[2] as u32);
    if let Some(n) = nc {
        cfg.nc = n;
    }
    if let Some(r) = reg_max {
        cfg.reg_max = r;
    }
    cfg
}

// ---- bench (needs NPU) ----
fn bench(args: &[String]) {
    let mut onnx_path = String::new();
    let mut input = 640u32;
    let mut iters = 200usize;
    let mut warmup = 20usize;
    let mut image = String::new();
    let mut opts = NpuOpts::default();
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--onnx" => onnx_path = val(args, &mut i, "--onnx"),
            "--input" => input = val(args, &mut i, "--input").parse().unwrap_or(input),
            "--iters" => iters = val(args, &mut i, "--iters").parse().unwrap_or(iters),
            "--warmup" => warmup = val(args, &mut i, "--warmup").parse().unwrap_or(warmup),
            "--image" => image = val(args, &mut i, "--image"),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if onnx_path.is_empty() {
        eprintln!("usage: brain npu bench --onnx M [--input S --ov-device NPU --iters N --warmup W --hint throughput]");
        return;
    }
    let mut session = match NpuSession::load(Path::new(&onnx_path), &opts.to_config()) {
        Ok(s) => s,
        Err(e) => die(e),
    };
    let shape = session.input_shape();
    let shape = if shape[2] == 0 { [1, 3, input as usize, input as usize] } else { shape };
    // Use the image if provided, else a mid-grey constant input.
    let chw = if image.is_empty() {
        vec![0.5f32; shape.iter().product()]
    } else {
        match crate::image_io::load_image(&image) {
            Ok((hwc, w, h)) => {
                let cfg = infer_cfg(&session, None, None, Some(shape[2] as u32));
                let (chw, _) = yolov8::boxmath::letterbox_rgb(&hwc, w, h, cfg.input, 114.0 / 255.0);
                chw
            }
            Err(e) => {
                eprintln!("brain npu bench: {e}");
                std::process::exit(1);
            }
        }
    };
    match npu::openvino::bench(&mut session, &chw, shape, warmup, iters) {
        Ok(r) => {
            println!("device       {}", r.device);
            println!("iters        {}", r.iters);
            println!("latency p50  {:.3} ms", r.p50_ms);
            println!("latency p99  {:.3} ms", r.p99_ms);
            println!("latency mean {:.3} ms", r.mean_ms);
            println!("throughput   {:.1} fps", r.throughput_fps);
        }
        Err(e) => die(e),
    }
}

