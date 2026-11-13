// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The NPU route as actions beside the served ones: `export` (the decoder as an
//! ONNX graph with dense-expert MoE, fp32 or INT8 weight-only) and
//! `npu-generate` (greedy decoding through OpenVINO, the Intel NPU by default).
//!
//! They are actions, not workflows, so their flags are typed and bounded from
//! one spec like every served action's, `--help` and `--json` come for free,
//! and the ONNX file is a named output (`--out onnx=PATH`). The program's
//! manifest is the model's served one plus these; [`provider`] answers both.
//!
//! `export` is pure Rust and runs anywhere; `npu-generate` needs OpenVINO and
//! an Intel NPU at run time and says so when it has neither. Both read a brain
//! checkpoint (`import` converts a Hugging Face release into one).
//!
//! Swedish Embedded AB implements decoder deployment to NPUs and other edge
//! accelerators for its clients. If your team needs expertise in exporting
//! language models for OpenVINO then you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use data::tokenizer::{CharTokenizer, Tokenizer};
use npu::openvino::NpuDevice;
use serde_json::json;

/// Context length the exported graph is compiled for when the caller names none.
const DEFAULT_SEQ: i64 = 32;

fn int8_param() -> ParamSpec {
    ParamSpec::new("int8", ParamType::Bool, "INT8 weight-only graph instead of fp32").default(json!(false))
}

pub fn export_spec() -> ActionSpec {
    ActionSpec::new("export", "export the checkpoint as an ONNX decoder graph (dense-expert MoE) for OpenVINO and the NPU")
        .param(ParamSpec::new("seq", ParamType::Int, "context length the graph is compiled for").min(1.0).max(capability::bounds::CONTEXT_TOKENS).default(json!(DEFAULT_SEQ)))
        .param(int8_param())
        .output(BlobSpec::new("onnx", Media::Bytes, "the ONNX decoder graph").required())
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("npu-generate", "greedy text generation through OpenVINO (the Intel NPU by default): the checkpoint exported in memory, cache-free over the padded context")
        .param(ParamSpec::new("data", ParamType::Str, "a dataset directory on this machine whose meta.json holds the vocabulary, for a checkpoint that does not embed one").host_resolved())
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (empty: a newline)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").min(0.0).max(capability::bounds::TOKENS).default(json!(128)))
        .param(ParamSpec::new("seq", ParamType::Int, "context length the graph is compiled for; 0 = the prompt plus max_new").min(0.0).max(capability::bounds::CONTEXT_TOKENS).default(json!(0)))
        .param(int8_param())
        .param(ParamSpec::new("ov_device", ParamType::Enum(["npu", "cpu", "gpu", "auto"].map(String::from).to_vec()), "OpenVINO device the graph compiles for").default(json!("npu")))
        .param(ParamSpec::new("allow_fallback", ParamType::Bool, "fall back to CPU/GPU when the requested device is absent").default(json!(true)))
        .output(BlobSpec::new("text", Media::Text, "the prompt and what was generated"))
}

/// The model's served manifest plus the NPU actions.
pub fn manifest() -> Manifest {
    let mut m = glmdsa::caps::manifest();
    m.actions.extend([export_spec(), generate_spec()]);
    m
}

/// The served provider, extended with the NPU actions over the same checkpoint.
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    let served = glmdsa::caps::provider(assembly)?;
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
        let kind = match name {
            "export" => Kind::Export,
            "npu-generate" => Kind::Generate,
            _ => return self.served.action(name),
        };
        Some(Arc::new(NpuAction { kind, weights: self.weights.clone() }))
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Export,
    Generate,
}

struct NpuAction {
    kind: Kind,
    weights: Option<String>,
}

impl NpuAction {
    /// The brain checkpoint the graph is built from: the graph builder reads a
    /// checkpoint file, not a Hugging Face directory.
    fn checkpoint(&self, action: &str) -> Result<&str, String> {
        let weights = self.weights.as_deref().ok_or_else(|| format!("glmdsa {action}: no GLM checkpoint (import one, train one, or name it with --weights)"))?;
        if Path::new(weights).is_dir() {
            return Err(format!("glmdsa {action}: {weights} is a Hugging Face directory; `import` it into a brain checkpoint first"));
        }
        Ok(weights)
    }

    fn export(&self, inv: &Invocation) -> ActionResult {
        let weights = self.checkpoint("export")?;
        let seq = inv.get_i64("seq").unwrap_or(DEFAULT_SEQ).max(1) as usize;
        let int8 = inv.get_bool("int8").unwrap_or(false);
        let (bytes, _config) = if int8 { npu::glm_export::build_glm_int8_bytes(weights, seq) } else { npu::glm_export::build_glm_fp32_bytes(weights, seq) }.map_err(|e| format!("glmdsa export: {e}"))?;
        Ok(Outcome::new().set("bytes", json!(bytes.len())).set("seq", json!(seq)).set("int8", json!(int8)).blob("onnx", Blob::new(Media::Bytes, bytes)))
    }

    fn generate(&self, inv: &Invocation) -> ActionResult {
        let weights = self.checkpoint("npu-generate")?;
        let device_name = inv.get_str("ov_device").unwrap_or_else(|| "npu".into());
        let device = NpuDevice::parse(&device_name).ok_or_else(|| format!("glmdsa npu-generate: ov_device expects npu|cpu|gpu|auto (got {device_name:?})"))?;
        let data = inv.get_str("data").map(PathBuf::from);
        let tok = CharTokenizer::from_itos(glmdsa::caps::char_vocabulary(glmdsa::Glm::load_itos(weights), data.as_deref())?);
        let prompt = inv.get_str("prompt").filter(|p| !p.is_empty()).unwrap_or_else(|| "\n".into());
        let max_new = inv.get_i64("max_new").unwrap_or(128).max(0) as usize;
        let seq = inv.get_i64("seq").filter(|&n| n > 0).map(|n| n as usize);
        let run = npu::glm_decode::generate(weights, &tok.encode(&prompt), max_new, device, inv.get_bool("allow_fallback").unwrap_or(true), seq, inv.get_bool("int8").unwrap_or(false))
            .map_err(|e| format!("glmdsa npu-generate: {e}"))?;
        let text = format!("{prompt}{}", tok.decode(&run.tokens));
        Ok(Outcome::new().set("device", json!(run.device)).set("load_ms", json!(run.load_ms)).set("gen_ms", json!(run.gen_ms)).blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

impl Action for NpuAction {
    fn spec(&self) -> ActionSpec {
        match self.kind {
            Kind::Export => export_spec(),
            Kind::Generate => generate_spec(),
        }
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        match self.kind {
            Kind::Export => self.export(inv),
            Kind::Generate => self.generate(inv),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_checkpoint(dir: &Path) -> String {
        let cfg = glmdsa::GlmConfig::tiny();
        let init = glmdsa::init_weights(&cfg, 3);
        let itos: Vec<char> = ('a'..='z').take(cfg.vocab as usize).collect();
        let path = dir.join("tiny.safetensors");
        glmdsa::Glm::new(cfg.clone(), 1, cfg.block_size, &init).save_with_itos(path.to_str().unwrap(), Some(&itos));
        path.to_string_lossy().into_owned()
    }

    fn run(kind: Kind, weights: Option<String>, inv: Invocation) -> ActionResult {
        let action = NpuAction { kind, weights };
        let inv = action.spec().validate(inv)?;
        action.run(&inv, &mut |_| {})
    }

    #[test]
    fn the_manifest_is_the_served_actions_then_the_npu_ones() {
        let names: Vec<String> = manifest().actions.into_iter().map(|a| a.name).collect();
        assert_eq!(names, ["train", "finetune", "eval", "generate", "import", "export", "npu-generate"]);
    }

    #[test]
    fn export_writes_an_onnx_graph_and_needs_a_brain_checkpoint() {
        let dir = std::env::temp_dir().join(format!("glmdsa-npu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = run(Kind::Export, Some(tiny_checkpoint(&dir)), Invocation::new().set("seq", json!(8))).unwrap();
        let bytes = &out.blobs["onnx"].bytes;
        assert!(bytes.len() > 1000 && out.outputs["bytes"] == bytes.len());
        assert!(run(Kind::Export, None, Invocation::new()).unwrap_err().contains("no GLM checkpoint"));
        assert!(run(Kind::Export, Some(dir.to_string_lossy().into_owned()), Invocation::new()).unwrap_err().contains("import"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn npu_generate_names_a_bad_device() {
        let dir = std::env::temp_dir().join(format!("glmdsa-npu-dev-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weights = tiny_checkpoint(&dir);
        let err = NpuAction { kind: Kind::Generate, weights: Some(weights) }.run(&Invocation::new().set("ov_device", json!("tpu")), &mut |_| {}).unwrap_err();
        assert!(err.contains("ov_device"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
