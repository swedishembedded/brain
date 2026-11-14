// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The NPU route as actions beside the served ones: `export-onnx` (the decoder
//! as an ONNX graph), `npu-generate` (greedy decoding through OpenVINO, the
//! Intel NPU by default) and `npu-precompile` (export and compile once into a
//! cache so later `npu-generate` runs skip the wait).
//!
//! They are actions, not workflows, so their flags are typed and bounded from
//! one spec like every served action's, and `--help` and `--json` come for
//! free. The program's manifest is the model's served one plus these; [`provider`] answers both.
//!
//! An explicit NPU request is refused when OpenVINO compiled the graph for
//! another device, unless `allow_fallback` says a CPU or GPU result is wanted:
//! a fallback that is not surfaced is how "it ran on the NPU" claims stop
//! being true. All three read a brain checkpoint (`import` converts a Hugging
//! Face release into one).
//!
//! Swedish Embedded AB implements decoder deployment to NPUs and other edge
//! accelerators for its clients. If your team needs expertise in exporting
//! language models for OpenVINO then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::tokenizer::Tokenizer;
use npu::openvino::NpuDevice;
use serde_json::json;

/// Context length the exported graph is compiled for when the caller names none.
const DEFAULT_SEQ: i64 = 32;

fn seq_param(help: &str) -> ParamSpec {
    ParamSpec::new("seq", ParamType::Int, help).min(1.0).max(capability::bounds::CONTEXT_TOKENS)
}

fn weights_param() -> ParamSpec {
    ParamSpec::new("weights", ParamType::Str, "the brain checkpoint (.safetensors) the graph is built from; overrides the model store's own pick when set").host_resolved()
}

fn fallback_param() -> ParamSpec {
    ParamSpec::new("allow_fallback", ParamType::Bool, "accept a graph OpenVINO compiled for the CPU or GPU instead of refusing it").default(json!(false))
}

fn cache_param(help: &str) -> ParamSpec {
    ParamSpec::new("npu_cache", ParamType::Str, help).host_resolved()
}

pub fn export_spec() -> ActionSpec {
    ActionSpec::new("export-onnx", "export the checkpoint as an ONNX decoder graph for OpenVINO and `brain npu check`")
        .param(weights_param())
        .param(ParamSpec::new("out", ParamType::Str, "where to write the ONNX file; large weights are written beside it as external data").host_resolved().required())
        .param(seq_param("context length the graph is compiled for").default(json!(DEFAULT_SEQ)))
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("npu-generate", "greedy text generation through OpenVINO (the Intel NPU by default): the checkpoint exported, compiled and decoded over the padded context")
        .param(weights_param())
        .param(ParamSpec::new("tokenizer", ParamType::Str, "the tokenizer.json to encode and decode with; omit to use the model store's own pick").host_resolved())
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue").required())
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").min(1.0).max(capability::bounds::TOKENS).default(json!(32)))
        .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt").default(json!(false)))
        .param(seq_param("context length the graph is compiled for, so one cache serves every prompt that fits; omit for the prompt plus max_new"))
        .param(cache_param("where the exported ONNX and the compiled blob are cached for later runs; omit to cache nothing"))
        .param(ParamSpec::new("ov_device", ParamType::Enum(["npu", "cpu", "gpu", "auto"].map(String::from).to_vec()), "OpenVINO device the graph compiles for").default(json!("npu")))
        .param(fallback_param())
        .output(BlobSpec::new("text", Media::Text, "the prompt and what was generated"))
}

pub fn precompile_spec() -> ActionSpec {
    ActionSpec::new("npu-precompile", "export and compile the NPU decoder once into the cache, so later `npu-generate` runs with the same cache and seq skip the wait")
        .param(weights_param())
        .param(seq_param("context length the graph is compiled for").required())
        .param(cache_param("where the exported ONNX and the compiled blob are written").required())
        .param(ParamSpec::new("ov_device", ParamType::Enum(["npu", "cpu", "gpu", "auto"].map(String::from).to_vec()), "OpenVINO device the graph compiles for").default(json!("npu")))
        .param(fallback_param())
}

/// The model's served manifest plus the NPU actions and `toy`.
pub fn manifest() -> Manifest {
    let mut m = qwen3::caps::manifest();
    m.actions.extend([export_spec(), generate_spec(), precompile_spec(), crate::toy_action::spec()]);
    m
}

/// The served provider, extended with the NPU actions over the same checkpoint
/// and tokenizer.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    let weights = assembly.role_path("weights").ok();
    let tokenizer = assembly.role_path("tokenizer").ok();
    let served = Arc::new(qwen3::caps::QwenProvider::new().with_defaults(weights.clone(), tokenizer.clone()));
    Ok(Arc::new(WithNpu { served, weights, tokenizer }))
}

struct WithNpu {
    served: Arc<dyn Provider>,
    weights: Option<String>,
    tokenizer: Option<String>,
}

impl Provider for WithNpu {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let kind = match name {
            "export-onnx" => Kind::Export,
            "npu-generate" => Kind::Generate,
            "npu-precompile" => Kind::Precompile,
            "toy" => return Some(Arc::new(crate::toy_action::Toy)),
            _ => return self.served.action(name),
        };
        Some(Arc::new(NpuAction { kind, weights: self.weights.clone(), tokenizer: self.tokenizer.clone() }))
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Export,
    Generate,
    Precompile,
}

struct NpuAction {
    kind: Kind,
    weights: Option<String>,
    tokenizer: Option<String>,
}

fn device(inv: &Invocation, action: &str) -> Result<NpuDevice, String> {
    let name = inv.get_str("ov_device").unwrap_or_else(|| "npu".into());
    NpuDevice::parse(&name).ok_or_else(|| format!("qwen3 {action}: ov_device expects npu|cpu|gpu|auto (got {name:?})"))
}

/// An explicit NPU request is refused when the graph compiled elsewhere,
/// unless the caller accepted that.
fn require_npu(compiled_on: &str, inv: &Invocation, action: &str) -> Result<(), String> {
    let wanted_npu = inv.get_str("ov_device").is_none_or(|d| d == "npu");
    if wanted_npu && compiled_on != "NPU" && !inv.get_bool("allow_fallback").unwrap_or(false) {
        return Err(format!("qwen3 {action}: compiled on {compiled_on}, not NPU (pass allow_fallback to accept this on purpose)"));
    }
    Ok(())
}

impl NpuAction {
    /// The brain checkpoint the graph is built from: the graph builder reads a
    /// checkpoint file, not a Hugging Face directory.
    fn checkpoint(&self, inv: &Invocation, action: &str) -> Result<String, String> {
        let weights = inv
            .get_str("weights")
            .filter(|w| !w.is_empty())
            .or_else(|| self.weights.clone())
            .ok_or_else(|| format!("qwen3 {action}: no Qwen3 checkpoint (import one, train one, or name it with --weights)"))?;
        if Path::new(&weights).is_dir() {
            return Err(format!("qwen3 {action}: {weights} is a directory; `import` a Hugging Face release into a brain checkpoint first"));
        }
        Ok(weights)
    }

    fn export(&self, inv: &Invocation) -> ActionResult {
        let weights = self.checkpoint(inv, "export-onnx")?;
        let seq = inv.get_i64("seq").unwrap_or(DEFAULT_SEQ).max(1) as usize;
        let out = PathBuf::from(inv.get_str("out").filter(|o| !o.is_empty()).ok_or("qwen3 export-onnx: missing required param 'out'")?);
        let reader = checkpoint::weightio::WeightReader::open(&weights).map_err(|e| format!("qwen3 export-onnx: {e}"))?;
        npu::qwen_export::export_qwen_fp32(&reader, &out.to_string_lossy(), seq).map_err(|e| format!("qwen3 export-onnx: {e}"))?;
        Ok(Outcome::new().set("saved", json!(out.display().to_string())).set("seq", json!(seq)))
    }

    fn generate(&self, inv: &Invocation) -> ActionResult {
        let weights = self.checkpoint(inv, "npu-generate")?;
        let tokenizer = inv.get_str("tokenizer").filter(|t| !t.is_empty()).or_else(|| self.tokenizer.clone()).ok_or("qwen3 npu-generate: no tokenizer (name one with --tokenizer)")?;
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&tokenizer).map_err(|e| format!("qwen3 npu-generate: tokenizer {tokenizer}: {e}"))?;
        let prompt = inv.get_str("prompt").ok_or("qwen3 npu-generate: missing required param 'prompt'")?;
        let text = if inv.get_bool("chat").unwrap_or(false) { tok.apply_chat_template(&[("user", &prompt)], true) } else { prompt.clone() };
        let ids = tok.encode(&text);
        if ids.is_empty() {
            return Err("qwen3 npu-generate: empty prompt".into());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(32).max(1) as usize;
        let seq = inv.get_i64("seq").filter(|&n| n > 0).map(|n| n as usize);
        let cache = inv.get_str("npu_cache").filter(|c| !c.is_empty()).map(PathBuf::from);
        let allow_fallback = inv.get_bool("allow_fallback").unwrap_or(false);
        let run = npu::qwen_decode::generate(&weights, &ids, max_new, device(inv, "npu-generate")?, allow_fallback, cache.as_deref(), seq).map_err(|e| format!("qwen3 npu-generate: {e}"))?;
        require_npu(&run.device, inv, "npu-generate")?;
        let out = format!("{prompt}{}", tok.decode(&run.tokens));
        Ok(Outcome::new()
            .set("device", json!(run.device))
            .set("onnx_cached", json!(run.onnx_cached))
            .set("load_ms", json!(run.load_ms))
            .set("gen_ms", json!(run.gen_ms))
            .set("tokens", json!(run.tokens.len()))
            .blob("text", Blob::new(Media::Text, out.into_bytes())))
    }

    fn precompile(&self, inv: &Invocation) -> ActionResult {
        let weights = self.checkpoint(inv, "npu-precompile")?;
        let seq = inv.get_i64("seq").ok_or("qwen3 npu-precompile: missing required param 'seq'")?.max(1) as usize;
        let cache = PathBuf::from(inv.get_str("npu_cache").ok_or("qwen3 npu-precompile: missing required param 'npu_cache'")?);
        let allow_fallback = inv.get_bool("allow_fallback").unwrap_or(false);
        let (compiled_on, ms) = npu::qwen_decode::precompile(&weights, seq, device(inv, "npu-precompile")?, allow_fallback, &cache).map_err(|e| format!("qwen3 npu-precompile: {e}"))?;
        require_npu(&compiled_on, inv, "npu-precompile")?;
        Ok(Outcome::new().set("device", json!(compiled_on)).set("seq", json!(seq)).set("seconds", json!(ms / 1e3)).set("cache", json!(cache.display().to_string())))
    }
}

impl Action for NpuAction {
    fn spec(&self) -> ActionSpec {
        match self.kind {
            Kind::Export => export_spec(),
            Kind::Generate => generate_spec(),
            Kind::Precompile => precompile_spec(),
        }
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.kind {
            Kind::Export => self.export(inv),
            Kind::Generate => self.generate(inv),
            Kind::Precompile => self.precompile(inv),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_checkpoint(dir: &Path) -> String {
        let cfg = qwen3::QwenConfig::tiny();
        let init = qwen3::init_weights(&cfg, 3);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg.param_list().into_iter().map(|(name, n)| (name.clone(), vec![n as u64], init[&name].clone())).collect();
        let path = dir.join("tiny.safetensors");
        checkpoint::save(path.to_str().unwrap(), cfg.to_json(), &tensors);
        path.to_string_lossy().into_owned()
    }

    fn run(kind: Kind, weights: Option<String>, inv: Invocation) -> ActionResult {
        let action = NpuAction { kind, weights, tokenizer: None };
        let inv = action.spec().validate(inv)?;
        action.run(&inv, &mut |_| {})
    }

    #[test]
    fn the_manifest_is_the_served_actions_then_the_npu_ones() {
        let names: Vec<String> = manifest().actions.into_iter().map(|a| a.name).collect();
        assert_eq!(&names[names.len() - 4..], ["export-onnx", "npu-generate", "npu-precompile", "toy"]);
        assert!(names.iter().any(|n| n == "generate") && names.iter().any(|n| n == "lora_train"));
    }

    #[test]
    fn export_writes_an_onnx_graph_and_needs_a_brain_checkpoint() {
        let dir = std::env::temp_dir().join(format!("qwen3-npu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("tiny.onnx");
        let inv = || Invocation::new().set("seq", json!(8)).set("out", json!(out.to_str().unwrap()));
        let done = run(Kind::Export, Some(tiny_checkpoint(&dir)), inv()).unwrap();
        assert_eq!(done.outputs["seq"], 8);
        assert!(std::fs::metadata(&out).is_ok_and(|m| m.len() > 1000));
        assert!(run(Kind::Export, None, inv()).unwrap_err().contains("no Qwen3 checkpoint"));
        assert!(run(Kind::Export, Some(dir.to_string_lossy().into_owned()), inv()).unwrap_err().contains("import"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bad_openvino_device_is_named() {
        let dir = std::env::temp_dir().join(format!("qwen3-npu-dev-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = tiny_checkpoint(&dir);
        let inv = Invocation::new().set("seq", json!(8)).set("npu_cache", json!(dir.to_str().unwrap())).set("ov_device", json!("tpu"));
        let err = NpuAction { kind: Kind::Precompile, weights: Some(weights), tokenizer: None }.run(&inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("ov_device"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
