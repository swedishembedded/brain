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
use std::time::Instant;

use npu::openvino::{
    Chronos2Session, FincastSession, KronosS1Session, KronosS2Session, NpuConfig, NpuDevice, NpuError, NpuSession,
    PerfHint,
};

pub fn run_npu(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("export") => export(&args[1..]),
        Some("quantize") => quantize(&args[1..]),
        Some("check") => check(&args[1..]),
        Some("run") => run(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("sim") => sim(&args[1..]),
        Some("lfm") => lfm(&args[1..]),
        Some("lfm-bench") => lfm_bench(&args[1..]),
        Some("chronos2") => chronos2(&args[1..]),
        Some("kronos") => kronos(&args[1..]),
        Some("fincast") => fincast(&args[1..]),
        other => eprintln!(
            "usage: brain npu <export|quantize|check|run|bench|sim|lfm|lfm-bench|chronos2|kronos|fincast> ...  (got {other:?})"
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

/// `brain npu chronos2` — forecast a context series with the Chronos-2 transformer
/// core running on the NPU. The host (this model's compute backend) does the
/// scaler/patch/embed/REG assembly and the head rearrange/denorm; the exported
/// ONNX core (`emb`+`kmask` → `qhead`) runs on the accelerator via the pluggable
/// core seam. `--compare` also runs the pure device path and reports the max diff.
/// `brain npu lfm --weights F --seq S --out model.onnx [--int8]` — export the
/// LFM2.5-Encoder at a fixed sequence-length bucket for OpenVINO compilation
/// (static shapes; one graph per bucket, see docs/models/lfm/status.md).
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
    let (mut dot, mut na, mut nb, mut max_abs) = (0f64, 0f64, 0f64, 0f32);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
        max_abs = max_abs.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), max_abs)
}

/// `brain npu lfm-bench` — export the LFM2.5-Encoder at a fixed sequence bucket,
/// compile it on the accelerator (no CPU/GPU fallback), and time the encoder's
/// one-shot forward: `--warmup` runs excluded, then `--iters` timed `sess.run()`
/// calls → p50/p99/mean ms. **Compile time is reported separately** from
/// inference (never mixed). `--compare` also runs brain's own chunked forward on
/// the *same* fixed token ids and reports cosine — the NPU-fp16 vs brain-fp32
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
             --device NPU --quant f16|int8|int4 --compare]"
        );
        std::process::exit(2);
    }

    // Vocab (for in-range token ids) from the checkpoint config — cheap header read.
    let cfg = lfm::config::LfmConfig::from_json(&checkpoint::load(&weights).header["config"]);
    let vocab = cfg.vocab;

    // 1) Export the fixed-shape graph (external-data sidecar) — pure Rust, one-time.
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
    //    reported on its own line — it is a one-time cost, not per-inference.
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
            "brain npu lfm-bench: compiled on {dev}, not {want} (allow_fallback off) — \
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
        let m = lfm::model::Lfm::load_inference_chunked(&weights, 1, seq as u32, 512 << 20, 0);
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

fn chronos2(args: &[String]) {
    use std::cell::RefCell;
    let mut weights = String::new();
    let mut context_len = 128usize;
    let mut horizon = 8usize;
    let mut series_file: Option<String> = None;
    let mut compare = false;
    let mut opts = NpuOpts::default();
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--context-len" => context_len = val(args, &mut i, "--context-len").parse().unwrap_or(context_len),
            "--horizon" => horizon = val(args, &mut i, "--horizon").parse().unwrap_or(horizon),
            "--series" => series_file = Some(val(args, &mut i, "--series")),
            "--compare" => compare = true,
            other => {
                eprintln!("brain npu chronos2: unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!(
            "usage: brain npu chronos2 --weights W [--context-len N] [--horizon H] \
             [--series file] [--compare] [--device NPU]"
        );
        return;
    }

    // context: a file of newline/space/comma-separated f32, else a synthetic sinusoid.
    let context: Vec<f32> = match &series_file {
        Some(f) => match std::fs::read_to_string(f) {
            Ok(s) => s.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()).filter_map(|t| t.parse().ok()).collect(),
            Err(e) => {
                eprintln!("read {f}: {e}");
                return;
            }
        },
        None => (0..context_len).map(|i| 100.0 + (i as f32 * 0.1).sin() * 5.0).collect(),
    };
    if context.len() < 8 {
        eprintln!("chronos2: context too short ({} points)", context.len());
        return;
    }

    let model = match chronos2::model::Chronos2::load(&weights) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load chronos2 {weights}: {e}");
            return;
        }
    };
    let d = model.config().d_model;
    let q = model.config().num_quantiles;
    let cfg = opts.to_config();

    // Run the transformer core on the NPU via the pluggable-core seam. The closure
    // compiles the ONNX for the exact (S, n_out) this context needs, then infers.
    let device = RefCell::new(String::new());
    let core_ms = RefCell::new(0.0f64);
    let t0 = Instant::now();
    let out = model.forecast_quantiles_with_core(&context, horizon, |emb, mask, n_out| {
        let s = emb.len() / d;
        let bytes = npu::chronos2_export::export_onnx(&weights, s, n_out, npu::qwen_topology::Quant::F32)
            .unwrap_or_else(|e| {
                eprintln!("export chronos2 core onnx: {e}");
                std::process::exit(1);
            });
        let mut sess = Chronos2Session::load_bytes(&bytes, &cfg).unwrap_or_else(|e| {
            eprintln!("compile chronos2 core: {e}");
            std::process::exit(1);
        });
        *device.borrow_mut() = sess.device().to_string();
        let tc = Instant::now();
        let r = sess.run(emb, mask).unwrap_or_else(|e| {
            eprintln!("npu infer: {e}");
            std::process::exit(1);
        });
        *core_ms.borrow_mut() = tc.elapsed().as_secs_f64() * 1e3;
        r
    });
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!(
        "chronos2 core on {} · horizon {horizon} · {} context pts · core {:.1} ms (total {total_ms:.1} ms)",
        device.borrow(),
        context.len(),
        core_ms.borrow()
    );
    // print the median quantile path (and the 10%/90% band if present).
    let row = |qi: usize| &out[qi * horizon..qi * horizon + horizon];
    println!("  median : {:?}", row(q / 2).iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>());
    if q >= 3 {
        println!("  p10    : {:?}", row(0).iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>());
        println!("  p90    : {:?}", row(q - 1).iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>());
    }

    if compare {
        let cpu = model.forecast_quantiles(&context, horizon);
        let maxdiff = out.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        println!("  vs device core: max abs diff {maxdiff:.5} (rel {:.2e})", maxdiff / scale);
    }
}

/// `brain npu fincast` — forecast a context series with the FinCast decoder+MoE
/// transformer core running on the NPU. The host (this model's compute backend)
/// does the patch-embed/freq assembly and the head rearrange/denorm; the exported
/// ONNX core (`emb`+`amask` → `qhead`) runs on the accelerator via the pluggable
/// core seam. `--compare` also runs the pure device path and reports the max diff.
fn fincast(args: &[String]) {
    use std::cell::RefCell;
    let mut weights = String::new();
    let mut freq = 0usize;
    let mut horizon = 8usize;
    let mut series_file: Option<String> = None;
    let mut compare = false;
    let mut opts = NpuOpts::default();
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--freq" => freq = val(args, &mut i, "--freq").parse().unwrap_or(freq),
            "--horizon" => horizon = val(args, &mut i, "--horizon").parse().unwrap_or(horizon),
            "--series" => series_file = Some(val(args, &mut i, "--series")),
            "--compare" => compare = true,
            other => {
                eprintln!("brain npu fincast: unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!(
            "usage: brain npu fincast --weights W [--freq 0|1|2] [--horizon H] \
             [--series file] [--compare] [--device NPU]"
        );
        return;
    }

    let context: Vec<f32> = match &series_file {
        Some(f) => match std::fs::read_to_string(f) {
            Ok(s) => s.split(|c: char| c.is_whitespace() || c == ',').filter(|t| !t.is_empty()).filter_map(|t| t.parse().ok()).collect(),
            Err(e) => {
                eprintln!("read {f}: {e}");
                return;
            }
        },
        None => (0..512).map(|i| 100.0 + 0.05 * i as f32 + (i as f32 * 0.1).sin() * 5.0).collect(),
    };
    if context.len() < 8 {
        eprintln!("fincast: context too short ({} points)", context.len());
        return;
    }

    let model = match fincast::model::Fincast::load(&weights) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load fincast {weights}: {e}");
            return;
        }
    };
    let no = model.config().num_outputs();
    let cfg = opts.to_config();

    let device = RefCell::new(String::new());
    let core_ms = RefCell::new(0.0f64);
    let t0 = Instant::now();
    let out = model.forecast_full_with_core(&context, freq, horizon, |emb, amask| {
        let s = (amask.len() as f64).sqrt() as usize;
        let bytes = npu::fincast_export::export_onnx(&weights, s, npu::qwen_topology::Quant::F32).unwrap_or_else(|e| {
            eprintln!("export fincast core onnx: {e}");
            std::process::exit(1);
        });
        let mut sess = FincastSession::load_bytes(&bytes, &cfg).unwrap_or_else(|e| {
            eprintln!("compile fincast core: {e}");
            std::process::exit(1);
        });
        *device.borrow_mut() = sess.device().to_string();
        let tc = Instant::now();
        let r = sess.run(emb, amask).unwrap_or_else(|e| {
            eprintln!("npu infer: {e}");
            std::process::exit(1);
        });
        *core_ms.borrow_mut() = tc.elapsed().as_secs_f64() * 1e3;
        r
    });
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!(
        "fincast core on {} · horizon {horizon} · {} context pts · core {:.1} ms (total {total_ms:.1} ms)",
        device.borrow(),
        context.len(),
        core_ms.borrow()
    );
    // out is [horizon, num_outputs] step-major: col 0 = mean, cols 1.. = quantiles.
    let mean: Vec<f32> = (0..horizon).map(|t| (out[t * no] * 100.0).round() / 100.0).collect();
    println!("  mean : {mean:?}");
    if no >= 10 {
        let p10: Vec<f32> = (0..horizon).map(|t| (out[t * no + 1] * 100.0).round() / 100.0).collect();
        let p90: Vec<f32> = (0..horizon).map(|t| (out[t * no + 9] * 100.0).round() / 100.0).collect();
        println!("  p10  : {p10:?}");
        println!("  p90  : {p90:?}");
    }

    if compare {
        let cpu = model.forecast_full(&context, freq, horizon);
        let maxdiff = out.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        println!("  vs device core: max abs diff {maxdiff:.5} (rel {:.2e})", maxdiff / scale);
    }
}

fn parse_opt_u32(args: &[String], i: &mut usize, flag: &str) -> Option<u32> {
    val(args, i, flag).parse().ok()
}

fn argmax(x: &[f32]) -> u32 {
    (0..x.len()).max_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap()).unwrap_or(0) as u32
}

/// The Kronos AR rollout, parameterized by the s1/s2 **core** so the identical
/// loop runs on the device (`core_forward_s1/s2`) or the NPU (the two sessions):
/// per step, host-embed the window → s1 core → argmax → sib-embed → s2 core →
/// argmax → append → slide the `t_win` window. Returns the generated `(s1, s2)`
/// token tails. `s1_core(x) -> (ctx, s1_logits)`; `s2_core(ctx, sib) -> s2_logits`.
fn kronos_rollout<F1, F2>(
    dec: &kronos::decoder::KronosDecoder,
    s1_ctx: &[u32],
    s2_ctx: &[u32],
    horizon: usize,
    t_win: usize,
    mut s1_core: F1,
    mut s2_core: F2,
) -> (Vec<u32>, Vec<u32>)
where
    F1: FnMut(&[f32]) -> (Vec<f32>, Vec<f32>),
    F2: FnMut(&[f32], &[f32]) -> Vec<f32>,
{
    let (vs1, vs2) = (dec.config().s1_vocab(), dec.config().s2_vocab());
    let ctx_len = s1_ctx.len();
    let mut s1 = s1_ctx.to_vec();
    let mut s2 = s2_ctx.to_vec();
    for _ in 0..horizon {
        let len = s1.len();
        let w0 = len - t_win;
        let (s1w, s2w) = (&s1[w0..], &s2[w0..]);
        let x = dec.embed_tokens(s1w, s2w, &[]);
        let (ctx, s1_logits) = s1_core(&x);
        let samp_s1 = argmax(&s1_logits[(t_win - 1) * vs1..t_win * vs1]);
        let mut s1_cond = s1w.to_vec();
        *s1_cond.last_mut().unwrap() = samp_s1;
        let sib = dec.sib_embed(&s1_cond);
        let s2_logits = s2_core(&ctx, &sib);
        let samp_s2 = argmax(&s2_logits[(t_win - 1) * vs2..t_win * vs2]);
        s1.push(samp_s1);
        s2.push(samp_s2);
    }
    (s1[ctx_len..].to_vec(), s2[ctx_len..].to_vec())
}

/// `brain npu kronos` — autoregressive Kronos forecast with both decoder graphs
/// (`decode_s1` → s1, `decode_s2` → s2) running on the NPU. The host does BSQ
/// tokenization, token embedding, argmax, and the sliding window. `--compare`
/// also runs the identical rollout on the device core and reports token
/// agreement (the NPU-vs-device parity check).
fn kronos(args: &[String]) {
    let mut tok_dir = String::new();
    let mut dec_dir = String::new();
    let mut context_len = 0usize; // 0 → use the decoder's max_context
    let mut horizon = 8usize;
    let mut compare = false;
    let mut opts = NpuOpts::default();
    let mut i = 0;
    while i < args.len() {
        if opts.parse_flag(args, &mut i) {
            i += 1;
            continue;
        }
        match args[i].as_str() {
            "--kronos-tokenizer" | "--tokenizer" => tok_dir = val(args, &mut i, "--kronos-tokenizer"),
            "--kronos-decoder" | "--decoder" => dec_dir = val(args, &mut i, "--kronos-decoder"),
            "--context-len" => context_len = val(args, &mut i, "--context-len").parse().unwrap_or(0),
            "--horizon" => horizon = val(args, &mut i, "--horizon").parse().unwrap_or(horizon),
            "--compare" => compare = true,
            other => {
                eprintln!("brain npu kronos: unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if tok_dir.is_empty() || dec_dir.is_empty() {
        eprintln!(
            "usage: brain npu kronos --kronos-tokenizer T --kronos-decoder D \
             [--context-len N] [--horizon H] [--device NPU] [--compare]"
        );
        return;
    }

    let model = match kronos::import::load_model(&tok_dir, &dec_dir) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load kronos: {e}");
            return;
        }
    };
    // The graph is fixed-length; keep the window at exactly `t_win` so the device
    // and NPU rollouts slide identically. Default to the decoder's max_context.
    let t_win = if context_len > 0 { context_len } else { model.max_context() };
    let feat = model.feat();

    // synthetic OHLCV(+amount) context of `t_win` bars → BSQ tokens.
    let mut bars = Vec::with_capacity(t_win * feat);
    for i in 0..t_win {
        let p = 100.0 + (i as f32 * 0.1).sin() * 5.0;
        let (o, h, l, c, v) = (p, p + 0.5, p - 0.5, p + 0.2, 1000.0 + i as f32);
        bars.extend_from_slice(&[o, h, l, c, v]);
        if feat >= 6 {
            bars.push(v * (o + h + l + c) / 4.0);
        }
        for _ in 6..feat {
            bars.push(0.0);
        }
    }
    let (s1_ctx, s2_ctx) = model.tokenize(&bars, t_win);
    let dec = model.decoder();
    let cfg = opts.to_config();

    // export both graphs at T = t_win.
    let s1_bytes = npu::kronos_export::export_onnx(&dec_dir, t_win, npu::qwen_topology::Quant::F32)
        .unwrap_or_else(|e| {
            eprintln!("export decode_s1: {e}");
            std::process::exit(1);
        });
    let s2_bytes = npu::kronos_export::export_dep_onnx(&dec_dir, t_win, npu::qwen_topology::Quant::F32)
        .unwrap_or_else(|e| {
            eprintln!("export decode_s2: {e}");
            std::process::exit(1);
        });
    let mut s1_sess = KronosS1Session::load_bytes(&s1_bytes, &cfg).unwrap_or_else(|e| {
        eprintln!("compile decode_s1: {e}");
        std::process::exit(1);
    });
    let mut s2_sess = KronosS2Session::load_bytes(&s2_bytes, &cfg).unwrap_or_else(|e| {
        eprintln!("compile decode_s2: {e}");
        std::process::exit(1);
    });
    let device = s1_sess.device().to_string();

    let t0 = Instant::now();
    let (g1, g2) = kronos_rollout(
        dec,
        &s1_ctx,
        &s2_ctx,
        horizon,
        t_win,
        |x| s1_sess.run(x).expect("npu decode_s1"),
        |ctx, sib| s2_sess.run(ctx, sib).expect("npu decode_s2"),
    );
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let per_step = ms / horizon.max(1) as f64;

    // detokenize the generated tail → normalized bars; show the close column.
    let recon = model.tokenizer().decode(&g1, &g2); // [horizon, feat] normalized
    let close: Vec<f32> = (0..horizon).map(|k| (recon[k * feat + 3] * 1000.0).round() / 1000.0).collect();
    println!(
        "kronos AR forecast on {device} · T={t_win} · horizon {horizon} · {ms:.1} ms ({per_step:.1} ms/step)"
    );
    println!("  generated {} s1 + {} s2 tokens; recon close (normalized): {close:?}", g1.len(), g2.len());

    if compare {
        let (d1, d2) = kronos_rollout(
            dec,
            &s1_ctx,
            &s2_ctx,
            horizon,
            t_win,
            |x| {
                let (l, c) = dec.core_forward_s1(x, t_win);
                (c, l)
            },
            |ctx, sib| dec.core_forward_s2(ctx, sib, t_win),
        );
        let a1 = g1.iter().zip(&d1).filter(|(a, b)| a == b).count();
        let a2 = g2.iter().zip(&d2).filter(|(a, b)| a == b).count();
        println!(
            "  vs device core: s1 {}/{} match, s2 {}/{} match",
            a1,
            g1.len(),
            a2,
            g2.len()
        );
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
