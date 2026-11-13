// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain npu …` - ONNX graphs on the Intel NPU via OpenVINO.
//!
//!   brain npu check    --onnx M [--ov-device NPU]
//!   brain npu bench    --onnx M [--input S --ov-device NPU --iters N --warmup W --hint latency|throughput ...]
//!   brain npu omni     --hf-dir DIR [--out-dir out/omni] [--n-audio N --grid-h H --grid-w W --code-len T]
//!   brain npu deepseekocr2 --weights DIR [--seq S --out-dir out]  # resampler (2 views) + decoder ONNX
//!
//! `check` decodes a graph and, where OpenVINO and an NPU are present,
//! compiles it; `bench` times one. YOLO's export, INT8 quantization, NPU
//! detection and INT8 simulation are actions of `samples/models/yolov8`.
//!
//! `omni` exports omni's three NPU-eligible pieces (audio tower, vision tower,
//! Code2Wav vocoder - NOT the 30B-parameter MoE Thinker decoder, not an NPU
//! target) as fp32 ONNX and runs the same structural + (where available)
//! OpenVINO-compile check `check` does on each. See `qwen3omnimoe::npu_export`'s
//! module doc for the exact scope (single-merger vision path, no DeepStack).

use std::path::Path;
use std::time::Instant;

use npu::openvino::{NpuConfig, NpuDevice, NpuError, NpuSession, PerfHint};

pub fn run_npu(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("check") => check(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("lfm") => lfm(&args[1..]),
        Some("deepseekocr2") => deepseekocr2(&args[1..]),
        Some("lfm-bench") => lfm_bench(&args[1..]),
        Some("omni") => omni(&args[1..]),
        other => eprintln!(
            "usage: brain npu <check|bench|lfm|lfm-bench|omni|deepseekocr2> ...  (got {other:?})"
        ),
    }
}

/// `brain npu deepseekocr2 --weights DIR --seq S --out-dir out` - export
/// DeepSeek-OCR-2's ONNX-eligible pieces (the resampler, one graph per view
/// size, and the decoder) for a best-effort OpenVINO/NPU compile attempt.
/// `--weights` is the directory `deepseekocr2::import::Files::locate` reads
/// (both GGUFs, or their already-expanded fp32 forms). SAM is NOT exported -
/// see `npu::deepseekocr2_topology`'s module doc for the real, unresolved
/// reason (no windowed-attention/relative-position-bias ONNX precedent
/// exists in this crate yet).
fn deepseekocr2(args: &[String]) {
    let mut weights = String::new();
    let mut seq = 512usize;
    let mut out_dir = String::from("out/deepseekocr2-onnx");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--seq" => seq = val(args, &mut i, "--seq").parse().unwrap_or(seq),
            "--out-dir" => out_dir = val(args, &mut i, "--out-dir"),
            other => {
                eprintln!("brain npu deepseekocr2: unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!("usage: brain npu deepseekocr2 --weights DIR --seq S --out-dir out");
        std::process::exit(2);
    }
    if let Err(e) = npu::deepseekocr2_export::export_all(&weights, seq, &out_dir) {
        eprintln!("brain npu deepseekocr2: {e}");
        std::process::exit(1);
    }
    println!("brain npu deepseekocr2: wrote resampler_local.onnx, resampler_global.onnx, decoder.onnx -> {out_dir}");
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

/// `brain npu lfm --weights F --seq S --out model.onnx [--int8]` - export the
/// LFM2.5-Encoder at a fixed sequence-length bucket for OpenVINO compilation
/// (static shapes; one graph per bucket).
fn lfm(args: &[String]) {
    let mut weights = String::new();
    let mut seq = 1024usize;
    let mut out = String::from("out/lfm.onnx");
    let mut int8 = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--seq" => seq = val(args, &mut i, "--seq").parse().unwrap_or(seq),
            "--out" => out = val(args, &mut i, "--out"),
            "--int8" => int8 = true,
            other => {
                eprintln!("brain npu lfm: unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!("usage: brain npu lfm --weights F --seq S --out model.onnx [--int8]");
        std::process::exit(2);
    }
    if let Err(e) = npu::lfm_export::export(&weights, seq, &out, int8) {
        eprintln!("brain npu lfm: {e}");
        std::process::exit(1);
    }
}

fn cosine_f32(a: &[f32], b: &[f32]) -> (f64, f32) {
    // Cosine from the shared model::hostmath (audit F35); only the max|Δ|
    // companion stays local.
    let max_abs = a.iter().zip(b).fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (model::hostmath::cosine(a, b) as f64, max_abs)
}

/// `brain npu lfm-bench` - export the LFM2.5-Encoder at a fixed sequence bucket,
/// compile it on the accelerator (no CPU/GPU fallback), and time the encoder's
/// one-shot forward: `--warmup` runs excluded, then `--iters` timed `sess.run()`
/// calls → p50/p99/mean ms. **Compile time is reported separately** from
/// inference (never mixed). `--compare` also runs brain's own chunked forward on
/// the *same* fixed token ids and reports cosine - the NPU-fp16 vs brain-fp32
/// parity gate (≥ 0.999; the NPU executes fp16 internally so we gate on cosine,
/// not max-abs). The device is asserted to be exactly the requested one so a
/// silent fallback is never reported as an NPU number.
fn lfm_bench(args: &[String]) {
    use npu::openvino::LfmSession;
    use npu::qwen_topology::Quant;
    let mut weights = String::new();
    let mut seq = 8192usize;
    let mut out = String::from("out/lfm-bench.onnx");
    let mut iters = 20usize;
    let mut warmup = 5usize;
    // The Intel NPU is fp16-native, so an fp32 graph runs in fp16 on it: that is
    // the `f16` precision. `int8`/`int4` add weight-only quantization on top.
    let mut quant = Quant::F32;
    let mut quant_label = "f16 (NPU-native)";
    let mut compare = false;
    let mut opts = NpuOpts::default(); // device NPU, latency hint, allow_fallback = false
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--seq" => seq = val(args, &mut i, "--seq").parse().unwrap_or(seq),
            "--out" => out = val(args, &mut i, "--out"),
            "--iters" => iters = val(args, &mut i, "--iters").parse().unwrap_or(iters),
            "--warmup" => warmup = val(args, &mut i, "--warmup").parse().unwrap_or(warmup),
            "--quant" => {
                let q = val(args, &mut i, "--quant");
                (quant, quant_label) = match q.as_str() {
                    "f16" | "fp16" | "f32" => (Quant::F32, "f16 (NPU-native)"),
                    "int8" | "i8" => (Quant::Int8, "int8-weights (fp16 acts)"),
                    "int4" | "i4" => (Quant::Int4, "int4-weights (fp16 acts)"),
                    other => {
                        eprintln!("brain npu lfm-bench: --quant expects f16|int8|int4 (got {other:?})");
                        std::process::exit(2);
                    }
                };
            }
            "--int8" => (quant, quant_label) = (Quant::Int8, "int8-weights (fp16 acts)"),
            "--compare" => compare = true,
            other => {
                eprintln!("brain npu lfm-bench: unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if weights.is_empty() || iters == 0 {
        eprintln!(
            "usage: brain npu lfm-bench --weights F [--seq S --iters N --warmup W \
             --ov-device NPU --quant f16|int8|int4 --compare]"
        );
        std::process::exit(2);
    }

    // Vocab (for in-range token ids) from the checkpoint config - cheap header read.
    let cfg = lfm2::config::LfmConfig::from_json(&checkpoint::read_config(&weights));
    let vocab = cfg.vocab;

    // 1) Export the fixed-shape graph (external-data sidecar) - pure Rust, one-time.
    if let Some(p) = Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let t_exp = Instant::now();
    if let Err(e) = npu::lfm_export::export_quant(&weights, seq, &out, quant) {
        eprintln!("brain npu lfm-bench: export: {e}");
        std::process::exit(1);
    }
    let export_ms = t_exp.elapsed().as_secs_f64() * 1e3;

    // 2) Compile on the accelerator (no fallback). Compile time is measured and
    //    reported on its own line - it is a one-time cost, not per-inference.
    let ncfg = opts.to_config();
    let t_c = Instant::now();
    let mut sess = match LfmSession::load_path(&out, &ncfg) {
        Ok(s) => s,
        Err(e) => die(e),
    };
    let compile_ms = t_c.elapsed().as_secs_f64() * 1e3;

    // Hard rule: a CPU/GPU fallback must never be reported as an NPU number.
    let dev = sess.device().to_string();
    let want = match opts.device {
        NpuDevice::Npu => "NPU",
        NpuDevice::Gpu => "GPU",
        NpuDevice::Cpu => "CPU",
        NpuDevice::Auto => "",
    };
    if !want.is_empty() && dev != want {
        eprintln!(
            "brain npu lfm-bench: compiled on {dev}, not {want} (allow_fallback off) - \
             refusing to report a non-{want} number"
        );
        std::process::exit(1);
    }
    if sess.seq_len() != seq {
        eprintln!("brain npu lfm-bench: compiled S={} != requested {seq}", sess.seq_len());
        std::process::exit(1);
    }

    // 3) Fixed inputs: token ids [1,S] + a zero key-mask (no padding).
    let ids_u32: Vec<u32> = (0..seq as u32).map(|k| (k.wrapping_mul(2_654_435_761) % vocab).max(1)).collect();
    let ids_i64: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
    let kmask = vec![0.0f32; seq];

    for _ in 0..warmup {
        sess.run(&ids_i64, &kmask).unwrap_or_else(|e| die(e));
    }
    let mut ms: Vec<f64> = Vec::with_capacity(iters);
    let mut got = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        got = sess.run(&ids_i64, &kmask).unwrap_or_else(|e| die(e));
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| ms[(((ms.len() - 1) as f64) * q).round() as usize];
    let (p50, p99, mean) = (pick(0.5), pick(0.99), ms.iter().sum::<f64>() / ms.len() as f64);

    println!("model        LFM2.5-Encoder  S={seq}  {quant_label}");
    println!("device       {dev}");
    println!("export       {export_ms:.1} ms   (pure Rust, one-time)");
    println!("compile      {compile_ms:.1} ms   (one-time; cache with --cache-dir)");
    println!("iters        {iters}   (warmup {warmup} excluded)");
    println!("latency p50  {p50:.1} ms");
    println!("latency p99  {p99:.1} ms");
    println!("latency mean {mean:.1} ms");
    println!("e2e/infer    {:.3} s   (single-sequence {seq}-token forward)", p50 / 1e3);

    // 4) Parity gate: NPU output vs brain's own chunked forward on identical ids.
    if compare {
        let m = lfm2::model::Lfm::load_inference_chunked(&weights, 1, seq as u32, 512 << 20, 0);
        m.set_tokens(&ids_u32);
        m.forward();
        let reference = m.read_hidden();
        drop(m);
        if reference.len() != got.len() {
            eprintln!("parity: len {} != NPU {}", reference.len(), got.len());
            std::process::exit(1);
        }
        let (cos, max_abs) = cosine_f32(&reference, &got);
        println!("parity       cosine {cos:.6}  max_abs {max_abs:.4}  (NPU fp16 vs brain fp32)");
        if cos < 0.999 {
            eprintln!("brain npu lfm-bench: PARITY FAIL cosine {cos:.6} < 0.999");
            std::process::exit(1);
        }
    }
}

/// `brain npu omni` - export omni's audio tower, vision tower and Code2Wav
/// vocoder as fp32 ONNX graphs and run the same structural (+ where an
/// OpenVINO device is present, compile) check `check` does on each. Pure
/// Rust export, runs anywhere; the compile step needs OpenVINO. No NPU
/// device run is attempted here - none is available on this box (see
/// `qwen3omnimoe::npu_export`'s module doc).
fn omni(args: &[String]) {
    let mut hf_dir = String::new();
    let mut out_dir = "out/omni".to_string();
    let mut n_audio = 100u32;
    let mut grid_h = 16u32;
    let mut grid_w = 16u32;
    let mut code_len = 32usize;
    let mut opts = NpuOpts::default();
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--hf-dir" => hf_dir = val(args, &mut i, "--hf-dir"),
            "--out-dir" => out_dir = val(args, &mut i, "--out-dir"),
            "--n-audio" => n_audio = val(args, &mut i, "--n-audio").parse().unwrap_or(n_audio),
            "--grid-h" => grid_h = val(args, &mut i, "--grid-h").parse().unwrap_or(grid_h),
            "--grid-w" => grid_w = val(args, &mut i, "--grid-w").parse().unwrap_or(grid_w),
            "--code-len" => code_len = val(args, &mut i, "--code-len").parse().unwrap_or(code_len),
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if hf_dir.is_empty() {
        eprintln!(
            "usage: brain npu omni --hf-dir DIR [--out-dir out/omni] [--n-audio 100] \
             [--grid-h 16 --grid-w 16] [--code-len 32] [--ov-device NPU]"
        );
        return;
    }
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("brain npu omni: create {out_dir}: {e}");
        std::process::exit(1);
    }
    let reader = match checkpoint::weightio::WeightReader::open_hf_dir(Path::new(&hf_dir)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain npu omni: open {hf_dir}: {e}");
            std::process::exit(1);
        }
    };
    let config_json = match std::fs::read_to_string(Path::new(&hf_dir).join("config.json")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("brain npu omni: read config.json: {e}");
            std::process::exit(1);
        }
    };
    let root: serde_json::Value = match serde_json::from_str(&config_json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("brain npu omni: parse config.json: {e}");
            std::process::exit(1);
        }
    };
    let cfg = qwen3omnimoe::config::OmniConfig::from_json(&root);

    let audio_path = format!("{out_dir}/audio_tower.onnx");
    match qwen3omnimoe::npu_export::export_audio_onnx(&reader, &audio_path, n_audio) {
        Ok(()) => {
            println!("== audio tower -> {audio_path} (n_audio={n_audio}) ==");
            structural_check(&audio_path, &opts);
        }
        Err(e) => eprintln!("audio tower export failed: {e}"),
    }
    let vision_path = format!("{out_dir}/vision_tower.onnx");
    match qwen3omnimoe::npu_export::export_vision_onnx(&reader, &vision_path, grid_h, grid_w) {
        Ok(()) => {
            println!("== vision tower -> {vision_path} (grid {grid_h}x{grid_w}) ==");
            structural_check(&vision_path, &opts);
        }
        Err(e) => eprintln!("vision tower export failed: {e}"),
    }
    let codec_path = format!("{out_dir}/codec_decoder.onnx");
    match qwen3omnimoe::npu_export::export_codec_onnx(&reader, &cfg.code2wav, &codec_path, code_len) {
        Ok(()) => {
            println!("== codec decoder -> {codec_path} (code_len={code_len}) ==");
            structural_check(&codec_path, &opts);
        }
        Err(e) => eprintln!("codec decoder export failed: {e}"),
    }
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
    structural_check(&onnx_path, &opts);
}

/// Structural ONNX validation (always available: decode + op histogram) plus -
/// where an OpenVINO device is present - an attempted compile. Pure diagnostic
/// printing, no process exit on failure (callers that export several graphs in
/// one run, e.g. `brain npu omni`, want to see every graph's result, not abort
/// at the first).
fn structural_check(onnx_path: &str, opts: &NpuOpts) {
    let bytes = match std::fs::read(onnx_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("brain npu check: reading {onnx_path}: {e}");
            return;
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
            return;
        }
    }
    // Device compile check (needs OpenVINO): probe devices, try to compile.
    match npu::openvino::available_devices() {
        Ok(devs) if devs.is_empty() => {
            println!("openvino: no devices (runtime not installed); compile/op-coverage check skipped");
        }
        Ok(devs) => {
            println!("openvino devices: {devs:?}");
            match NpuSession::load(Path::new(onnx_path), &opts.to_config()) {
                Ok(s) => println!("compiled OK on {} (input {:?})", s.device(), s.input_shape()),
                Err(e) => eprintln!("compile failed: {e}"),
            }
        }
        Err(e) => println!("openvino unavailable: {e}"),
    }
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

