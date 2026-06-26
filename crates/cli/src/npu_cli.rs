// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain npu …` — deploy the YOLO detector to the Intel NPU via OpenVINO.
//!
//!   brain npu export   --weights F --out model.onnx [--input S --opset N]
//!   brain npu quantize --weights F --calib <dir> --out model.int8.onnx [--input S --num-calib N]
//!   brain npu check    --onnx M [--device NPU]
//!   brain npu run      --onnx M --image <P6|dir> [--device NPU --conf X --iou X --cache-dir D ...]
//!   brain npu bench    --onnx M [--input S --device NPU --iters N --warmup W --hint latency|throughput ...]
//!   brain npu sim      --weights F --calib <dir> --data <dir>   # fp32 vs INT8 mAP, no NPU
//!
//! `export`/`quantize`/`sim` are pure Rust and run anywhere. `run`/`bench`/`check`
//! (compile) need OpenVINO + an Intel NPU at run time; on a machine without them
//! they print a clear diagnostic. Output of `run` matches `brain yolo detect`.

use std::path::Path;

use npu::openvino::{NpuConfig, NpuDevice, NpuError, NpuSession, PerfHint};

pub fn run_npu(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("export") => export(&args[1..]),
        Some("quantize") => quantize(&args[1..]),
        Some("check") => check(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("sim") => sim(&args[1..]),
        other => eprintln!(
            "usage: brain npu <export|quantize|check|run|bench|sim> ...  (got {other:?})"
        ),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

fn parse_opt_u32(args: &[String], i: &mut usize, flag: &str) -> Option<u32> {
    val(args, i, flag).parse().ok()
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
            "--device" => {
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

// ---- export (pure Rust) ----
fn export(args: &[String]) {
    let mut weights = String::new();
    let mut out = String::new();
    let mut input: Option<u32> = None;
    let mut opset = onnx::DEFAULT_OPSET;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--out" => out = val(args, &mut i, "--out"),
            "--input" => input = parse_opt_u32(args, &mut i, "--input"),
            "--opset" => opset = val(args, &mut i, "--opset").parse().unwrap_or(opset),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || out.is_empty() {
        eprintln!("usage: brain npu export --weights F --out model.onnx [--input S --opset N]");
        return;
    }
    if let Err(e) = npu::export_fp32(&weights, &out, input, opset) {
        eprintln!("brain npu export: writing {out}: {e}");
        std::process::exit(1);
    }
    eprintln!("brain npu export: wrote fp32 ONNX {out}");
}

// ---- quantize (pure Rust) ----
fn quantize(args: &[String]) {
    let mut weights = String::new();
    let mut calib = String::new();
    let mut out = String::new();
    let mut input: Option<u32> = None;
    let mut num_calib = 300usize;
    let mut opset = onnx::DEFAULT_OPSET;
    let mut scales_out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--calib" => calib = val(args, &mut i, "--calib"),
            "--out" => out = val(args, &mut i, "--out"),
            "--input" => input = parse_opt_u32(args, &mut i, "--input"),
            "--num-calib" => num_calib = val(args, &mut i, "--num-calib").parse().unwrap_or(num_calib),
            "--opset" => opset = val(args, &mut i, "--opset").parse().unwrap_or(opset),
            "--scales-out" => scales_out = Some(val(args, &mut i, "--scales-out")),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || calib.is_empty() || out.is_empty() {
        eprintln!("usage: brain npu quantize --weights F --calib <dir> --out model.int8.onnx [--input S --num-calib N]");
        return;
    }
    let cfg = npu::config_of(&weights);
    let input_size = input.unwrap_or(cfg.input);
    let images = match npu::load_calib_images(&calib, input_size, num_calib) {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            eprintln!("brain npu quantize: no calibration images found in {calib}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("brain npu quantize: loading calibration images from {calib}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("brain npu quantize: calibrating on {} images at {input_size}px…", images.len());
    let quant = npu::calibrate_from_weights(&weights, &images);
    if let Some(p) = &scales_out {
        let _ = std::fs::write(p, serde_json::to_vec_pretty(&quant.to_json()).unwrap());
    }
    if let Err(e) = npu::export_int8(&weights, &quant, &out, input, opset) {
        eprintln!("brain npu quantize: writing {out}: {e}");
        std::process::exit(1);
    }
    eprintln!("brain npu quantize: {} conv activations calibrated; wrote INT8 ONNX {out}", quant.len());
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
        eprintln!("usage: brain npu check --onnx M [--device NPU]");
        return;
    }
    // Structural check (always available): decode + op histogram.
    let bytes = match std::fs::read(&onnx_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("brain npu check: reading {onnx_path}: {e}");
            std::process::exit(1);
        }
    };
    match onnx::decode_model(&bytes) {
        Ok(m) => {
            let g = m.graph.unwrap_or_default();
            let mut ops: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
            for n in &g.node {
                *ops.entry(n.op_type.clone()).or_insert(0) += 1;
            }
            println!("onnx: {} nodes, {} initializers, {} inputs, {} outputs",
                g.node.len(), g.initializer.len(), g.input.len(), g.output.len());
            println!("ops:");
            for (op, c) in &ops {
                println!("  {op:<20} {c}");
            }
        }
        Err(e) => {
            eprintln!("brain npu check: malformed ONNX: {e}");
            std::process::exit(1);
        }
    }
    // Device compile check (needs OpenVINO): probe devices, try to compile.
    match npu::openvino::available_devices() {
        Ok(devs) if devs.is_empty() => {
            println!("openvino: no devices (runtime not installed); compile/op-coverage check skipped");
        }
        Ok(devs) => {
            println!("openvino devices: {devs:?}");
            match NpuSession::load(Path::new(&onnx_path), &opts.to_config()) {
                Ok(s) => println!("compiled OK on {} (input {:?})", s.device(), s.input_shape()),
                Err(e) => eprintln!("compile failed: {e}"),
            }
        }
        Err(e) => println!("openvino unavailable: {e}"),
    }
}

// ---- run (needs NPU) ----
fn run(args: &[String]) {
    let mut onnx_path = String::new();
    let mut image = String::new();
    let mut conf = 0.25f32;
    let mut iou = 0.45f32;
    let mut nc: Option<u32> = None;
    let mut reg_max: Option<u32> = None;
    let mut input: Option<u32> = None;
    let mut opts = NpuOpts::default();
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--onnx" => onnx_path = val(args, &mut i, "--onnx"),
            "--image" => image = val(args, &mut i, "--image"),
            "--conf" => conf = val(args, &mut i, "--conf").parse().unwrap_or(conf),
            "--iou" => iou = val(args, &mut i, "--iou").parse().unwrap_or(iou),
            "--nc" => nc = parse_opt_u32(args, &mut i, "--nc"),
            "--reg-max" => reg_max = parse_opt_u32(args, &mut i, "--reg-max"),
            "--input" => input = parse_opt_u32(args, &mut i, "--input"),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if onnx_path.is_empty() || image.is_empty() {
        eprintln!("usage: brain npu run --onnx M --image <P6|dir> [--device NPU --conf X --iou X --nc C --reg-max R --input S]");
        return;
    }
    let (hwc, w, h) = match crate::image_io::load_image(&image) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain npu run: {e}");
            std::process::exit(1);
        }
    };
    let mut session = match NpuSession::load(Path::new(&onnx_path), &opts.to_config()) {
        Ok(s) => s,
        Err(e) => die(e),
    };
    // The decode needs nc + reg_max + input. They are not in the ONNX, so the
    // caller supplies them (defaults match yolov8n) or they are inferred.
    let cfg = infer_cfg(&session, nc, reg_max, input);
    let dets = match npu::detect_image(&mut session, &hwc, w, h, &cfg, conf, iou) {
        Ok(d) => d,
        Err(e) => die(e),
    };
    for d in &dets {
        println!("[{:.2},{:.2},{:.2},{:.2},{:.4},{}]", d[0], d[1], d[2], d[3], d[4], d[5] as u32);
    }
    eprintln!("brain npu run: {} detection(s) on {w}x{h} via {}", dets.len(), session.device());
}

/// Build a `YoloConfig` for decode from explicit flags (the ONNX doesn't carry
/// nc/reg_max). Defaults are yolov8n (nc 80, reg_max 16); input from the model.
fn infer_cfg(session: &NpuSession, nc: Option<u32>, reg_max: Option<u32>, input: Option<u32>) -> yolo::YoloConfig {
    let mut cfg = yolo::YoloConfig::yolov8n();
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
        eprintln!("usage: brain npu bench --onnx M [--input S --device NPU --iters N --warmup W --hint throughput]");
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
                let (chw, _) = yolo::boxmath::letterbox_rgb(&hwc, w, h, cfg.input, 114.0 / 255.0);
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

// ---- sim (pure Rust: fp32 vs INT8 mAP, no NPU) ----
fn sim(args: &[String]) {
    let mut weights = String::new();
    let mut calib = String::new();
    let mut data = String::new();
    let mut conf = 0.25f32;
    let mut iou = 0.45f32;
    let mut num_calib = 300usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--calib" => calib = val(args, &mut i, "--calib"),
            "--data" => data = val(args, &mut i, "--data"),
            "--conf" => conf = val(args, &mut i, "--conf").parse().unwrap_or(conf),
            "--iou" => iou = val(args, &mut i, "--iou").parse().unwrap_or(iou),
            "--num-calib" => num_calib = val(args, &mut i, "--num-calib").parse().unwrap_or(num_calib),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || data.is_empty() {
        eprintln!("usage: brain npu sim --weights F --data <dir> [--calib <dir> --num-calib N --conf X --iou X]");
        return;
    }
    let cfg = npu::config_of(&weights);
    // Calibrate on --calib if given, else reuse the eval dataset's images.
    let calib_dir = if calib.is_empty() { data.clone() } else { calib };
    let images = npu::load_calib_images(&calib_dir, cfg.input, num_calib).unwrap_or_default();
    if images.is_empty() {
        eprintln!("brain npu sim: no calibration images in {calib_dir}");
        std::process::exit(1);
    }
    let quant = npu::calibrate_from_weights(&weights, &images);
    let (m_fp32, m_int8) = npu::simulate_map(&weights, &data, &quant, conf, iou);
    println!("metric            value");
    println!("mAP@0.5 fp32      {m_fp32:.4}");
    println!("mAP@0.5 int8(sim) {m_int8:.4}");
    println!("delta             {:.4}", m_fp32 - m_int8);
}
