// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Intel NPU route as actions beside the served ones: `npu-export` (the
//! encoder as a fixed-shape ONNX graph for one sequence-length bucket) and
//! `npu-bench` (export, compile on the accelerator, time the one-shot forward
//! and, with `compare`, gate it against brain's own forward by cosine).
//!
//! `npu-export` is pure Rust and runs anywhere; `npu-bench` needs OpenVINO and
//! an Intel NPU at run time and says so when it has neither. Compile time is
//! reported apart from inference, and a compile that landed on another device
//! than the one asked for is refused, so a CPU or GPU fallback is never
//! reported as an NPU number.
//!
//! Swedish Embedded AB implements text-encoder deployment to NPUs and other
//! edge accelerators for its clients. If your team needs expertise in
//! exporting and benchmarking long-context encoders for OpenVINO then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::sync::Arc;
use std::time::Instant;

use capability::{Action, ActionResult, ActionSpec, Assembly, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use npu::openvino::{LfmSession, NpuDevice};
use npu::qwen_topology::Quant;
use serde_json::json;

/// Cosine below which the NPU's fp16 forward is not the same encoder as brain's fp32 one.
const PARITY_COSINE: f64 = 0.999;
/// Most timed or warm-up runs one bench may ask for.
const MAX_RUNS: f64 = 1_000_000.0;
/// Rotating prime that spreads the fixed benchmark token ids over the vocabulary.
const TOKEN_STRIDE: u32 = 2_654_435_761;
/// Bytes of the reference forward's chunk buffer for `compare`.
const COMPARE_CHUNK_BYTES: u64 = 512 << 20;

const QUANTS: [&str; 3] = ["f16", "int8", "int4"];

fn seq_param(default: i64) -> ParamSpec {
    ParamSpec::new("seq", ParamType::Int, "sequence-length bucket the graph is exported for (static shapes: one graph per bucket)").min(1.0).max(capability::bounds::TOKENS).default(json!(default))
}

fn out_param(default: &str) -> ParamSpec {
    ParamSpec::new("out", ParamType::Str, "where to write the ONNX graph (its weights go to a .data sidecar), on this machine").default(json!(default)).host_resolved()
}

fn quant_param() -> ParamSpec {
    ParamSpec::new("quant", ParamType::Enum(QUANTS.map(String::from).to_vec()), "weight precision: f16 (dense weights, which the NPU runs in its native fp16), or int8 / int4 weight-only quantization with fp16 activations").default(json!("f16"))
}

pub fn export_spec() -> ActionSpec {
    ActionSpec::new("npu-export", "export the encoder as a fixed-shape ONNX graph for one sequence-length bucket, for OpenVINO")
        .param(seq_param(1024))
        .param(out_param("out/lfm.onnx"))
        .param(quant_param())
}

pub fn bench_spec() -> ActionSpec {
    let mut spec = ActionSpec::new("npu-bench", "export the encoder at one bucket, compile it on the NPU and time its one-shot forward (compile time reported apart)")
        .param(seq_param(8192))
        .param(out_param("out/lfm-bench.onnx"))
        .param(quant_param())
        .param(ParamSpec::new("iters", ParamType::Int, "timed runs").min(1.0).max(MAX_RUNS).default(json!(20)))
        .param(ParamSpec::new("warmup", ParamType::Int, "runs before timing starts, excluded from it").min(0.0).max(MAX_RUNS).default(json!(5)))
        .param(ParamSpec::new("compare", ParamType::Bool, "also run brain's own forward on the same token ids and require cosine >= 0.999 against the NPU output").default(json!(false)))
        .output(BlobSpec::new("report", Media::Text, "the device, export and compile times, latency p50/p99/mean and, with compare, the parity"));
    spec.params.extend(npu::run_options::params());
    spec
}

/// The model's served manifest plus the NPU actions.
pub fn manifest() -> Manifest {
    let mut m = lfm2::caps::manifest();
    m.actions.extend([export_spec(), bench_spec()]);
    m
}

/// The served provider, extended with the NPU actions over the same checkpoint.
pub fn provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    let served = lfm2::caps::provider(assembly)?;
    Ok(Arc::new(WithNpu { served, weights: assembly.role_path("weights").ok() }))
}

struct WithNpu {
    served: Arc<dyn Provider>,
    weights: Option<String>,
}

impl Provider for WithNpu {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let bench = match name {
            "npu-export" => false,
            "npu-bench" => true,
            _ => return self.served.action(name),
        };
        Some(Arc::new(NpuAction { bench, weights: self.weights.clone() }))
    }
}

struct NpuAction {
    bench: bool,
    weights: Option<String>,
}

fn quant_of(inv: &Invocation) -> (Quant, &'static str) {
    match inv.get_str("quant").as_deref() {
        Some("int8") => (Quant::Int8, "int8-weights (fp16 acts)"),
        Some("int4") => (Quant::Int4, "int4-weights (fp16 acts)"),
        _ => (Quant::F32, "f16 (NPU-native)"),
    }
}

fn seq_of(inv: &Invocation, default: i64) -> usize {
    inv.get_i64("seq").unwrap_or(default).max(1) as usize
}

fn out_of(inv: &Invocation, default: &str) -> String {
    inv.get_str("out").filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string())
}

fn create_parent(out: &str, verb: &str) -> Result<(), String> {
    match std::path::Path::new(out).parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(p) => std::fs::create_dir_all(p).map_err(|e| format!("lfm2 {verb}: create {}: {e}", p.display())),
        None => Ok(()),
    }
}

impl Action for NpuAction {
    fn spec(&self) -> ActionSpec {
        if self.bench { bench_spec() } else { export_spec() }
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let verb = if self.bench { "npu-bench" } else { "npu-export" };
        let weights = self.weights.as_deref().ok_or_else(|| format!("lfm2 {verb}: no LFM2.5 checkpoint (pull one, or name it with --weights)"))?;
        if self.bench { bench(weights, inv, progress) } else { export(weights, inv) }
    }
}

fn export(weights: &str, inv: &Invocation) -> ActionResult {
    let (seq, out, (quant, label)) = (seq_of(inv, 1024), out_of(inv, "out/lfm.onnx"), quant_of(inv));
    create_parent(&out, "npu-export")?;
    npu::lfm_export::export_quant(weights, seq, &out, quant).map_err(|e| format!("lfm2 npu-export: {e}"))?;
    Ok(Outcome::new().set("onnx", json!(out)).set("seq", json!(seq)).set("quant", json!(label)))
}

fn bench(weights: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let (seq, out, (quant, label)) = (seq_of(inv, 8192), out_of(inv, "out/lfm-bench.onnx"), quant_of(inv));
    let iters = inv.get_i64("iters").unwrap_or(20).max(1) as usize;
    let warmup = inv.get_i64("warmup").unwrap_or(5).max(0) as usize;
    let ncfg = npu::run_options::config(inv)?;
    let npu_err = |what: &str, e: npu::openvino::NpuError| format!("lfm2 npu-bench: {what}: {e}");

    // Vocabulary, for in-range token ids: a cheap header read.
    let vocab = lfm2::config::LfmConfig::from_json(&checkpoint::read_config(weights)).vocab;

    progress(Progress::step(0, 4, format!("exporting S={seq}")));
    create_parent(&out, "npu-bench")?;
    let t_export = Instant::now();
    npu::lfm_export::export_quant(weights, seq, &out, quant).map_err(|e| format!("lfm2 npu-bench: export: {e}"))?;
    let export_ms = t_export.elapsed().as_secs_f64() * 1e3;

    progress(Progress::step(1, 4, "compiling"));
    let t_compile = Instant::now();
    let mut sess = LfmSession::load_path(&out, &ncfg).map_err(|e| npu_err("compile", e))?;
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;

    let device = sess.device().to_string();
    let want = match ncfg.device {
        NpuDevice::Npu => "NPU",
        NpuDevice::Gpu => "GPU",
        NpuDevice::Cpu => "CPU",
        NpuDevice::Auto => "",
    };
    if !want.is_empty() && device != want {
        return Err(format!("lfm2 npu-bench: compiled on {device}, not {want} (allow_fallback off): refusing to report a non-{want} number"));
    }
    if sess.seq_len() != seq {
        return Err(format!("lfm2 npu-bench: compiled S={} != requested {seq}", sess.seq_len()));
    }

    // Fixed inputs: token ids [1, S] and a zero key mask (no padding).
    let ids_u32: Vec<u32> = (0..seq as u32).map(|k| (k.wrapping_mul(TOKEN_STRIDE) % vocab).max(1)).collect();
    let ids: Vec<i64> = ids_u32.iter().map(|&x| i64::from(x)).collect();
    let key_mask = vec![0.0f32; seq];

    progress(Progress::step(2, 4, format!("timing {iters} runs after {warmup} warm-up")));
    for _ in 0..warmup {
        sess.run(&ids, &key_mask).map_err(|e| npu_err("run", e))?;
    }
    let mut ms = Vec::with_capacity(iters);
    let mut last = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        last = sess.run(&ids, &key_mask).map_err(|e| npu_err("run", e))?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    let pick = |q: f64| ms[(((ms.len() - 1) as f64) * q).round() as usize];
    let (p50, p99, mean) = (pick(0.5), pick(0.99), ms.iter().sum::<f64>() / ms.len() as f64);

    let mut report = format!(
        "model        LFM2.5-Encoder  S={seq}  {label}\ndevice       {device}\nexport       {export_ms:.1} ms   (pure Rust, one-time)\ncompile      {compile_ms:.1} ms   (one-time; cache with cache_dir)\n\
         iters        {iters}   (warmup {warmup} excluded)\nlatency p50  {p50:.1} ms\nlatency p99  {p99:.1} ms\nlatency mean {mean:.1} ms\ne2e/infer    {:.3} s   (single-sequence {seq}-token forward)\n",
        p50 / 1e3
    );
    let mut outcome = Outcome::new().set("device", json!(device)).set("export_ms", json!(export_ms)).set("compile_ms", json!(compile_ms)).set("p50_ms", json!(p50)).set("p99_ms", json!(p99)).set("mean_ms", json!(mean));

    if inv.get_bool("compare").unwrap_or(false) {
        progress(Progress::step(3, 4, "running brain's own forward for parity"));
        let model = lfm2::model::Lfm::load_inference_chunked(weights, 1, seq as u32, COMPARE_CHUNK_BYTES, 0);
        model.set_tokens(&ids_u32);
        model.forward();
        let reference = model.read_hidden();
        drop(model);
        if reference.len() != last.len() {
            return Err(format!("lfm2 npu-bench: parity: reference length {} != NPU {}", reference.len(), last.len()));
        }
        let max_abs = reference.iter().zip(&last).fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
        let cosine = f64::from(model::hostmath::cosine(&reference, &last));
        report.push_str(&format!("parity       cosine {cosine:.6}  max_abs {max_abs:.4}  (NPU fp16 vs brain fp32)\n"));
        if cosine < PARITY_COSINE {
            return Err(format!("{report}lfm2 npu-bench: PARITY FAIL cosine {cosine:.6} < {PARITY_COSINE}"));
        }
        outcome = outcome.set("cosine", json!(cosine)).set("max_abs_diff", json!(max_abs));
    }
    Ok(outcome.blob("report", Blob::new(Media::Text, report.into_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(spec: ActionSpec) -> Vec<String> {
        spec.params.into_iter().map(|p| p.name).collect()
    }

    #[test]
    fn the_actions_offer_every_knob_the_cli_verbs_had() {
        for want in ["seq", "out", "quant"] {
            assert!(names(export_spec()).iter().any(|n| n == want), "export {want}");
        }
        for want in ["seq", "out", "quant", "iters", "warmup", "compare", "ov_device", "hint", "cache_dir", "turbo", "allow_fallback", "profile"] {
            assert!(names(bench_spec()).iter().any(|n| n == want), "bench {want}");
        }
    }

    #[test]
    fn the_program_offers_the_served_actions_and_the_npu_ones() {
        let actions: Vec<String> = manifest().actions.into_iter().map(|a| a.name).collect();
        for want in ["fill_mask", "embed", "npu-export", "npu-bench"] {
            assert!(actions.iter().any(|a| a == want), "{want} in {actions:?}");
        }
    }

    #[test]
    fn an_npu_action_without_a_checkpoint_says_how_to_name_one() {
        for bench in [false, true] {
            let e = NpuAction { bench, weights: None }.run(&Invocation::new(), &mut |_| {}).unwrap_err();
            assert!(e.contains("--weights"), "{e}");
        }
    }

    #[test]
    fn quant_defaults_to_the_npu_native_precision() {
        assert!(matches!(quant_of(&Invocation::new()).0, Quant::F32));
        assert!(matches!(quant_of(&Invocation::new().set("quant", json!("int4"))).0, Quant::Int4));
    }
}
