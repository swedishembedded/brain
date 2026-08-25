// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spike: is the Qwen3 **KV-cache decode-step graph** (`qwen_topology::
//! build_talker_decode_graph` -- generic despite the name, already used by the
//! TTS Talker) worth wiring into `qwen_decode::generate` in place of its
//! current O(T²) cache-free recompute?
//!
//! `KvSession::run_step` re-uploads the ENTIRE past K/V cache to the device
//! every token (`n_layers * 2 * nkv * cap * hd * 4` bytes -- ~0.23 MB/slot at
//! Qwen3-0.6B). That marshal cost grows linearly in `cap`, same as the O(T²)
//! recompute it is meant to beat, so which one wins at a useful `cap` is a
//! measurement, not something the graph shapes settle on their own. This bin
//! is that measurement: per `cap`, it splits the KV decode step into
//! marshal/infer/readback (`KvSession::last_{marshal,infer,readback}_ms`) and
//! reports it next to the incumbent `DecoderSession` (cache-free) p50 at the
//! same `cap`.
//!
//! No OpenVINO device named "NPU" is reachable in this container -- every
//! number this prints is the OpenVINO **CPU** device unless run on a host
//! with real NPU firmware, and it refuses
//! to report silently otherwise (`--allow-fallback` is required to accept a
//! device other than the one requested).
//!
//! Exports live under `--out` and are deleted immediately after each `cap`'s
//! measurements to bound peak disk (this box runs nearly full, with ~13G free) -- INT8
//! decode/prefill graphs are ~0.6 GB each at 0.6B, the fp32 O(T²) baseline
//! ~2.4 GB; never all four `cap` values' worth on disk at once.
//!
//! Usage:
//!   qwen_kv_bench --weights qwen.brain.safetensors [--caps 256,512,1024,2048]
//!                  [--device cpu] [--iters 20] [--warmup 5] [--baseline-iters 5]
//!                  [--verify-cap 64] [--allow-fallback] [--out out/qwen-kv-bench]

use std::path::PathBuf;
use std::time::Instant;

use npu::openvino::{DecoderSession, KvSession, NpuConfig, NpuDevice, PrefillSession};
use npu::qwen_topology::Quant;
use qwen3::config::QwenConfig;

struct Opts {
    weights: String,
    caps: Vec<usize>,
    device: NpuDevice,
    iters: usize,
    warmup: usize,
    baseline_iters: usize,
    verify_cap: usize,
    allow_fallback: bool,
    /// Weight precision of the KV decode-step graph under test in [`verify`].
    /// `bench_cap`'s sweep is fixed to `Int8` regardless (see the module doc's
    /// disk-budget note) -- this only controls the correctness check, so a
    /// quant-vs-fp32 mismatch can be told apart from a real driver bug.
    verify_quant: Quant,
    out: PathBuf,
}

fn parse_quant(s: &str) -> Option<Quant> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => Some(Quant::F32),
        "int8" | "i8" => Some(Quant::Int8),
        "int4" | "i4" => Some(Quant::Int4),
        _ => None,
    }
}

fn parse_args() -> Opts {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut weights = String::new();
    let mut caps = vec![256usize, 512, 1024, 2048];
    let mut device = NpuDevice::Cpu;
    let mut iters = 20usize;
    let mut warmup = 5usize;
    let mut baseline_iters = 5usize;
    let mut verify_cap = 64usize;
    let mut allow_fallback = false;
    let mut verify_quant = Quant::Int8;
    let mut out = PathBuf::from("out/qwen-kv-bench");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                i += 1;
                weights = args[i].clone();
            }
            "--caps" => {
                i += 1;
                caps = args[i].split(',').filter_map(|s| s.trim().parse().ok()).collect();
            }
            "--device" => {
                i += 1;
                device = NpuDevice::parse(&args[i]).unwrap_or_else(|| {
                    eprintln!("unknown --device {:?}", args[i]);
                    std::process::exit(2);
                });
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().unwrap_or(iters);
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().unwrap_or(warmup);
            }
            "--baseline-iters" => {
                i += 1;
                baseline_iters = args[i].parse().unwrap_or(baseline_iters);
            }
            "--verify-cap" => {
                i += 1;
                verify_cap = args[i].parse().unwrap_or(verify_cap);
            }
            "--allow-fallback" => allow_fallback = true,
            "--verify-quant" => {
                i += 1;
                verify_quant = parse_quant(&args[i]).unwrap_or_else(|| {
                    eprintln!("unknown --verify-quant {:?} (want f32|int8|int4)", args[i]);
                    std::process::exit(2);
                });
            }
            "--out" => {
                i += 1;
                out = PathBuf::from(&args[i]);
            }
            other => {
                eprintln!("unknown flag {other:?}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if weights.is_empty() {
        eprintln!(
            "usage: qwen_kv_bench --weights F [--caps 256,512,1024,2048] [--device cpu|npu] \
             [--iters 20] [--warmup 5] [--baseline-iters 5] [--verify-cap 64] [--verify-quant f32|int8|int4] \
             [--allow-fallback] [--out out/qwen-kv-bench]"
        );
        std::process::exit(2);
    }
    Opts { weights, caps, device, iters, warmup, baseline_iters, verify_cap, allow_fallback, verify_quant, out }
}

fn npu_cfg(device: NpuDevice, allow_fallback: bool) -> NpuConfig {
    NpuConfig { device, allow_fallback, ..Default::default() }
}

/// Refuse to report a number compiled on a device other than the one asked
/// for, unless the caller opted into a fallback -- the same rule `lfm-bench`
/// (`crates/cli/src/npu_cli.rs`) enforces, so an OpenVINO CPU fallback can
/// never masquerade as an NPU number.
fn assert_device(got: &str, want: NpuDevice, allow_fallback: bool) {
    let want_s = want.ov_str();
    if got != want_s && !allow_fallback {
        eprintln!(
            "qwen_kv_bench: compiled on {got}, not {want_s} (allow_fallback off) -- refusing to \
             report a non-{want_s} number"
        );
        std::process::exit(1);
    }
    if got != want_s {
        eprintln!("qwen_kv_bench: NOTE -- compiled on {got}, not {want_s} (--allow-fallback was set)");
    }
}

fn percentiles(mut xs: Vec<f64>) -> (f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let p50 = xs[n / 2];
    let p99 = xs[((n as f64 * 0.99) as usize).min(n - 1)];
    (p50, p99, mean)
}

/// Deterministic in-vocab id (Knuth multiplicative hash), matching the
/// `lfm-bench` convention -- cost depends on shape, not token values.
fn fake_id(k: usize, vocab: usize) -> u32 {
    ((k as u64).wrapping_mul(2_654_435_761) % vocab as u64).max(1) as u32
}

fn cleanup(out: &PathBuf) {
    let data = out.with_file_name(format!("{}.data", out.file_name().unwrap().to_str().unwrap()));
    std::fs::remove_file(out).ok();
    std::fs::remove_file(data).ok();
}

/// Marshal/infer/readback split of the KV-cache decode-step graph at a fixed
/// `cap`, plus the incumbent cache-free `DecoderSession` p50 at the same
/// `cap` -- the two numbers the crossover decision is made from.
fn bench_cap(o: &Opts, cfg: &QwenConfig) {
    let (d, nkv, hd, nl) = (cfg.d_model as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize, cfg.n_layers as usize);
    let half = hd / 2;
    let vocab = cfg.vocab as usize;

    for &cap in &o.caps {
        println!("\n=== cap={cap} ===");
        std::fs::create_dir_all(&o.out).ok();

        // ---- KV-cache decode-step graph (INT8 weights) ----
        let decode_path = o.out.join(format!("qwen-decode-int8-{cap}.onnx"));
        let t_export = Instant::now();
        if let Err(e) =
            npu::qwen_export::export_talker_decode_int8(&o.weights, decode_path.to_str().unwrap(), cap)
        {
            eprintln!("cap={cap}: decode export failed: {e}");
            continue;
        }
        let decode_export_ms = t_export.elapsed().as_secs_f64() * 1e3;

        let t_compile = Instant::now();
        let sess = KvSession::load_path(&decode_path, &npu_cfg(o.device, o.allow_fallback), nl, d, nkv, hd, cap);
        let decode_compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
        let mut sess = match sess {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cap={cap}: decode compile failed: {e}");
                cleanup(&decode_path);
                continue;
            }
        };
        assert_device(sess.device(), o.device, o.allow_fallback);

        // Synthetic, deterministic, non-zero past K/V -- simulates a cache
        // that is completely full (the worst-case marshal size; every run_step
        // uploads the whole [nkv,cap,hd] buffer regardless of how much of it
        // is masked out).
        let mk_buf = |seed: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((seed * 2_654_435_761 + i).wrapping_mul(40503) % 1000) as f32 / 1000.0 - 0.5).collect()
        };
        let past_k: Vec<Vec<f32>> = (0..nl).map(|l| mk_buf(l * 2, nkv * cap * hd)).collect();
        let past_v: Vec<Vec<f32>> = (0..nl).map(|l| mk_buf(l * 2 + 1, nkv * cap * hd)).collect();
        let x = mk_buf(999, d);
        let mut cos = vec![0.0f32; hd];
        let mut sin = vec![0.0f32; hd];
        let pos = (cap.saturating_sub(1)) as f32;
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = pos * cfg.rope_theta.powf(-2.0 * m / hd as f32);
            cos[j] = ang.cos();
            sin[j] = ang.sin();
        }
        let mask = vec![0.0f32; cap]; // full cache: every slot valid

        for _ in 0..o.warmup {
            if let Err(e) = sess.run_step(&x, &cos, &sin, &mask, &past_k, &past_v) {
                eprintln!("cap={cap}: decode warmup failed: {e}");
                cleanup(&decode_path);
                continue;
            }
        }
        let (mut marshal, mut infer, mut readback) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..o.iters {
            match sess.run_step(&x, &cos, &sin, &mask, &past_k, &past_v) {
                Ok(_) => {
                    marshal.push(sess.last_marshal_ms());
                    infer.push(sess.last_infer_ms());
                    readback.push(sess.last_readback_ms());
                }
                Err(e) => eprintln!("cap={cap}: decode run failed: {e}"),
            }
        }
        drop(sess);
        cleanup(&decode_path);

        let (m50, m99, mmean) = percentiles(marshal);
        let (i50, i99, imean) = percentiles(infer);
        let (r50, r99, rmean) = percentiles(readback);
        let step50 = m50 + i50 + r50;
        println!("KV decode-step  export {decode_export_ms:.0}ms  compile {decode_compile_ms:.0}ms");
        println!(
            "  marshal  p50 {m50:.2}ms  p99 {m99:.2}ms  mean {mmean:.2}ms  ({:.0}% of step)",
            100.0 * m50 / step50.max(1e-9)
        );
        println!("  infer    p50 {i50:.2}ms  p99 {i99:.2}ms  mean {imean:.2}ms");
        println!("  readback p50 {r50:.2}ms  p99 {r99:.2}ms  mean {rmean:.2}ms");
        println!("  step total p50 {step50:.2}ms");

        // ---- Prefill graph (INT8), one big infer over the whole context ----
        let prefill_path = o.out.join(format!("qwen-prefill-int8-{cap}.onnx"));
        let t_export = Instant::now();
        if let Err(e) =
            npu::qwen_export::export_talker_prefill_int8(&o.weights, prefill_path.to_str().unwrap(), cap)
        {
            eprintln!("cap={cap}: prefill export failed: {e}");
        } else {
            let prefill_export_ms = t_export.elapsed().as_secs_f64() * 1e3;
            let t_compile = Instant::now();
            match PrefillSession::load_path(&prefill_path, &npu_cfg(o.device, o.allow_fallback), nl, d, nkv, hd, cap) {
                Ok(mut psess) => {
                    let prefill_compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
                    assert_device(psess.device(), o.device, o.allow_fallback);
                    let embeds = mk_buf(7, cap * d);
                    for _ in 0..o.warmup.min(2) {
                        let _ = psess.run(&embeds);
                    }
                    let mut times = Vec::new();
                    for _ in 0..o.baseline_iters {
                        let t = Instant::now();
                        if psess.run(&embeds).is_ok() {
                            times.push(t.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                    let (p50, p99, pmean) = percentiles(times);
                    println!("prefill (1 infer over {cap} tok)  export {prefill_export_ms:.0}ms  compile {prefill_compile_ms:.0}ms");
                    println!("  p50 {p50:.2}ms  p99 {p99:.2}ms  mean {pmean:.2}ms");
                }
                Err(e) => eprintln!("cap={cap}: prefill compile failed: {e}"),
            }
        }
        cleanup(&prefill_path);

        // ---- Incumbent: cache-free O(T^2) DecoderSession at the same cap ----
        let base_path = o.out.join(format!("qwen-decoder-fp32-{cap}.onnx"));
        let t_export = Instant::now();
        if let Err(e) = npu::qwen_export::export_qwen_fp32(&o.weights, base_path.to_str().unwrap(), cap) {
            eprintln!("cap={cap}: baseline export failed: {e}");
        } else {
            let base_export_ms = t_export.elapsed().as_secs_f64() * 1e3;
            let t_compile = Instant::now();
            match DecoderSession::load_path(&base_path, &npu_cfg(o.device, o.allow_fallback)) {
                Ok(mut dsess) => {
                    let base_compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;
                    assert_device(dsess.device(), o.device, o.allow_fallback);
                    let ids: Vec<i64> = (0..cap).map(|k| fake_id(k, vocab) as i64).collect();
                    for _ in 0..o.warmup.min(2) {
                        let _ = dsess.run_ids(&ids);
                    }
                    let mut times = Vec::new();
                    for _ in 0..o.baseline_iters {
                        let t = Instant::now();
                        if dsess.run_ids(&ids).is_ok() {
                            times.push(t.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                    let (p50, p99, bmean) = percentiles(times);
                    println!("O(T^2) DecoderSession (cache-free, whole {cap}-ctx recompute per token)  export {base_export_ms:.0}ms  compile {base_compile_ms:.0}ms");
                    println!("  p50 {p50:.2}ms  p99 {p99:.2}ms  mean {bmean:.2}ms");
                    println!(
                        "  KV decode-step p50 is {:.2}x the O(T^2) p50 at cap={cap}",
                        step50 / p50.max(1e-9)
                    );
                }
                Err(e) => eprintln!("cap={cap}: baseline compile failed: {e}"),
            }
        }
        cleanup(&base_path);
    }
}

/// Correctness: run the KV decode-step graph token-by-token and compare its
/// `hidden` output against `qwen3::Qwen::from_reader_decode(...).step(...)` -
/// brain's own fp32 GPU/CPU decode-only forward -- over a short deterministic
/// sequence. Reports BOTH cosine and max_abs (cosine alone is scale-invariant
/// and cannot see a dropped scale factor).
/// Returns whether parity held (cosine >= 0.999). The caller decides what to
/// do with a failure -- printing it loudly is not the same as blocking the
/// timing sweep on it: a quant accuracy regression and a marshal/infer/
/// readback measurement are two different questions, and burying the timing
/// numbers behind a parity `exit(1)` would silently turn off this bin's only
/// other job every time the quant path needs work (lessons.md #1 -- a gate
/// that never runs is worse than no gate, and a gate that also takes its
/// sibling measurement down with it is worse still).
fn verify(o: &Opts, cfg: &QwenConfig) -> bool {
    let cap = o.verify_cap;
    let quant_label = format!("{:?}", o.verify_quant);
    println!("\n=== correctness (cap={cap}, {quant_label} decode-step vs brain fp32 decode-only) ===");
    let (d, nkv, hd, nl) = (cfg.d_model as usize, cfg.n_kv_heads as usize, cfg.head_dim as usize, cfg.n_layers as usize);
    let half = hd / 2;
    let n_tokens = 16.min(cap);

    let reader = match checkpoint::weightio::WeightReader::open(&o.weights) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("verify: open {}: {e}", o.weights);
            return false;
        }
    };
    let tok_weight = reader.tensor("tok.weight").expect("checkpoint is missing tensor `tok.weight`");
    let vocab = cfg.vocab as usize;
    let ids: Vec<u32> = (0..n_tokens).map(|k| fake_id(k, vocab)).collect();

    // Reference: brain's own decode-only fp32 forward.
    let reference_model = qwen3::model::Qwen::from_reader_decode(&reader, (cap as u32).max(n_tokens as u32));
    reference_model.reset_cache();
    let mut reference: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    for &id in &ids {
        reference.push(reference_model.step(id));
    }

    // Candidate: the KV decode-step graph at `o.verify_quant`, driven exactly
    // like `qwen3tts::npu_gen::KvTalker::feed1` (the existing generic driver).
    let decode_path = o.out.join(format!("qwen-decode-verify-{cap}.onnx"));
    std::fs::create_dir_all(&o.out).ok();
    let export_result = match o.verify_quant {
        Quant::F32 => npu::qwen_export::export_talker_decode_fp32(&o.weights, decode_path.to_str().unwrap(), cap),
        Quant::Int8 => npu::qwen_export::export_talker_decode_int8(&o.weights, decode_path.to_str().unwrap(), cap),
        Quant::Int4 => npu::qwen_export::export_talker_decode_int4(&o.weights, decode_path.to_str().unwrap(), cap),
    };
    if let Err(e) = export_result {
        eprintln!("verify: decode export failed: {e}");
        return false;
    }
    let mut sess = match KvSession::load_path(&decode_path, &npu_cfg(o.device, o.allow_fallback), nl, d, nkv, hd, cap) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("verify: decode compile failed: {e}");
            cleanup(&decode_path);
            return false;
        }
    };
    // A silent device fallback still terminates the whole run -- an accuracy
    // number measured on the wrong device is not a softer failure than a
    // missing one.
    assert_device(sess.device(), o.device, o.allow_fallback);

    let mut past_k: Vec<Vec<f32>> = (0..nl).map(|_| vec![0.0f32; nkv * cap * hd]).collect();
    let mut past_v: Vec<Vec<f32>> = (0..nl).map(|_| vec![0.0f32; nkv * cap * hd]).collect();
    let mut got: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    for (pos, &id) in ids.iter().enumerate() {
        let embed = &tok_weight[id as usize * d..id as usize * d + d];
        let mut cos = vec![0.0f32; hd];
        let mut sin = vec![0.0f32; hd];
        for j in 0..hd {
            let m = (j % half) as f32;
            let ang = pos as f32 * cfg.rope_theta.powf(-2.0 * m / hd as f32);
            cos[j] = ang.cos();
            sin[j] = ang.sin();
        }
        let mut mask = vec![f32::NEG_INFINITY; cap];
        for slot in mask.iter_mut().take(pos) {
            *slot = 0.0;
        }
        let (hidden, nk, nv) = match sess.run_step(embed, &cos, &sin, &mask, &past_k, &past_v) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("verify: decode step {pos} failed: {e}");
                cleanup(&decode_path);
                return false;
            }
        };
        for l in 0..nl {
            for h in 0..nkv {
                let dst = h * cap * hd + pos * hd;
                let src = h * hd;
                past_k[l][dst..dst + hd].copy_from_slice(&nk[l][src..src + hd]);
                past_v[l][dst..dst + hd].copy_from_slice(&nv[l][src..src + hd]);
            }
        }
        got.push(hidden);
    }
    drop(sess);
    cleanup(&decode_path);

    let ref_flat: Vec<f32> = reference.into_iter().flatten().collect();
    let got_flat: Vec<f32> = got.into_iter().flatten().collect();
    let cos_sim = model::hostmath::cosine(&ref_flat, &got_flat);
    let max_abs = ref_flat.iter().zip(&got_flat).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("  cosine {cos_sim:.6}  max_abs {max_abs:.4}  ({n_tokens} tokens, {quant_label} weights vs brain fp32)");
    if cos_sim < 0.999 {
        eprintln!("  PARITY FAIL: cosine {cos_sim:.6} < 0.999");
        return false;
    }
    true
}

fn main() {
    let o = parse_args();
    let reader = match checkpoint::weightio::WeightReader::open(&o.weights) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("open {}: {e}", o.weights);
            std::process::exit(1);
        }
    };
    let cfg = QwenConfig::from_json(&reader.config());
    println!(
        "qwen_kv_bench: {} -- {} layers, d_model={}, n_kv_heads={}, head_dim={}, vocab={}",
        o.weights, cfg.n_layers, cfg.d_model, cfg.n_kv_heads, cfg.head_dim, cfg.vocab
    );
    println!(
        "device requested: {} (allow_fallback={})  -- NOTE: if this prints CPU, it is an OpenVINO CPU number, not NPU",
        o.device.ov_str(),
        o.allow_fallback
    );
    drop(reader);

    let parity_ok = verify(&o, &cfg);

    let reader = checkpoint::weightio::WeightReader::open(&o.weights).unwrap();
    let cfg = QwenConfig::from_json(&reader.config());
    bench_cap(&o, &cfg);

    // The timing sweep above is the point of this bin and must always run and
    // print; the exit code reflects parity only after it has, so a CI/script
    // caller still sees a non-zero status on a real accuracy regression
    // without the timing table going missing to get there.
    if !parity_ok {
        eprintln!("\nqwen_kv_bench: correctness check FAILED -- see PARITY FAIL above; timing numbers above are still valid measurements, only the accuracy claim is not");
        std::process::exit(1);
    }
}
