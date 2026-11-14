// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The conversion actions of `brain/qwen3`: `import` (a Hugging Face release
//! into a brain checkpoint), `export` (the resolved checkpoint as a
//! `transformers` directory or a llama.cpp GGUF) and `export_peft` (a LoRA
//! adapter as a PEFT adapter directory).
//!
//! A converted checkpoint carries the `qwen` card, so the model store
//! resolves it afterwards. The ONNX export for OpenVINO and the NPU needs the
//! `npu` crate and is an action of `samples/models/qwen3`.
//!
//! Swedish Embedded AB implements checkpoint conversion between training and
//! deployment formats for its clients. If your team needs expertise in moving
//! models between Hugging Face, llama.cpp and edge runtimes then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::Path;

use capability::{ActionResult, ActionSpec, Invocation, Outcome, ParamSpec, ParamType, Progress};
use serde_json::json;

use crate::export::{export_gguf, export_hf, export_peft as write_peft, GgufDtype, GgufExport, HfDtype, HfExport};
use crate::served::{guarded, path_param, Resolved};

const FORMATS: [&str; 2] = ["hf", "gguf"];
const DTYPES: [&str; 3] = ["f32", "bf16", "f16"];

pub(crate) fn import_spec() -> ActionSpec {
    ActionSpec::new("import", "convert a Hugging Face Qwen3 checkpoint into a brain checkpoint, carded for the model store")
        .cancellable()
        .from_scratch()
        .param(ParamSpec::new("source", ParamType::Str, "the Hugging Face release directory (config.json and its safetensors)").host_resolved().required())
        .param(ParamSpec::new("dest", ParamType::Str, "where to write the brain checkpoint (.safetensors)").host_resolved().required())
        .param(ParamSpec::new("block", ParamType::Int, "override the context the checkpoint is built with, tokens (a smaller window is a cheaper fine-tune; RoPE sets no hard limit)").min(1.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(ParamSpec::new("id", ParamType::Str, "the model id recorded on the card (default: the destination's base name)"))
}

pub(crate) fn export_spec() -> ActionSpec {
    ActionSpec::new("export", "write the resolved checkpoint as a Hugging Face `transformers` directory or a llama.cpp GGUF")
        .cancellable()
        .param(ParamSpec::new("weights", ParamType::Str, "the checkpoint to export (.safetensors, .gguf or a Hugging Face release); overrides the model store's own pick when set").host_resolved())
        .param(ParamSpec::new("out", ParamType::Str, "where to write: a directory for `hf`, a .gguf file for `gguf`").host_resolved().required())
        .param(ParamSpec::new("format", ParamType::Enum(FORMATS.map(String::from).to_vec()), "hf: a `transformers` directory; gguf: a llama.cpp file embedding the tokenizer").default(json!("hf")))
        .param(ParamSpec::new("dtype", ParamType::Enum(DTYPES.map(String::from).to_vec()), "stored element type: f32, bf16 or f16 for hf (default bf16); f32 or f16 for gguf (default f16)"))
        .param(ParamSpec::new("tokenizer", ParamType::Str, "the tokenizer.json to ship beside the weights; omit to use the model store's own pick").host_resolved())
        .param(ParamSpec::new("adapter", ParamType::Str, "a LoRA adapter (.safetensors) to fold into the exported weights (hf only)").host_resolved())
}

pub(crate) fn export_peft_spec() -> ActionSpec {
    ActionSpec::new("export_peft", "write a LoRA adapter as a PEFT adapter directory")
        .cancellable()
        .from_scratch()
        .param(ParamSpec::new("adapter", ParamType::Str, "the adapter to convert (.safetensors, `lora_train`'s output)").host_resolved().required())
        .param(ParamSpec::new("out", ParamType::Str, "the PEFT adapter directory to write").host_resolved().required())
        .param(ParamSpec::new("base_model", ParamType::Str, "the Hugging Face id recorded as the adapter's base model"))
}

pub(crate) fn import(_: &Resolved, inv: &Invocation, _: &mut dyn FnMut(Progress)) -> ActionResult {
    let source = path_param(inv, "source", "import")?;
    let dest = path_param(inv, "dest", "import")?;
    let block = inv.get_i64("block").map(|b| b.max(1) as u32);
    let id = inv.get_str("id").filter(|i| !i.is_empty());
    guarded("import", || crate::import::import_as(&source.to_string_lossy(), &dest.to_string_lossy(), block, id.as_deref()))?.map_err(|e| format!("qwen import: {e}"))?;
    Ok(Outcome::new().set("saved", json!(dest.display().to_string())))
}

pub(crate) fn export(resolved: &Resolved, inv: &Invocation, _: &mut dyn FnMut(Progress)) -> ActionResult {
    let weights = resolved.weights(inv, "export")?;
    let out = path_param(inv, "out", "export")?;
    let format = inv.get_str("format").unwrap_or_else(|| "hf".into());
    let dtype = inv.get_str("dtype");
    let tokenizer_dir = resolved.tokenizer(inv, &weights).parent().map(Path::to_path_buf);
    let adapter = inv.get_str("adapter").filter(|a| !a.is_empty());
    let (cfg, src) = crate::open_checkpoint(&weights).map_err(|e| format!("qwen export: {e}"))?;
    match format.as_str() {
        "hf" => {
            let dtype = match dtype.as_deref().unwrap_or("bf16") {
                "f32" => HfDtype::F32,
                "bf16" => HfDtype::Bf16,
                "f16" => HfDtype::F16,
                other => return Err(format!("qwen export: dtype {other:?}: expected f32, bf16 or f16 for format hf")),
            };
            let opts = HfExport { dtype, tokenizer_dir: tokenizer_dir.as_deref().filter(|d| d.join("tokenizer.json").is_file()), adapter: adapter.as_deref(), ..Default::default() };
            export_hf(&*src, &cfg, &out, &opts).map_err(|e| format!("qwen export: {e}"))?;
        }
        "gguf" => {
            if adapter.is_some() {
                return Err("qwen export: adapter is folded into an hf export only".into());
            }
            let dtype = match dtype.as_deref().unwrap_or("f16") {
                "f32" => GgufDtype::F32,
                "f16" => GgufDtype::F16,
                other => return Err(format!("qwen export: dtype {other:?}: expected f32 or f16 for format gguf")),
            };
            let tokenizer_dir = tokenizer_dir.ok_or("qwen export: format gguf embeds the tokenizer; name one")?;
            let name = out.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();
            export_gguf(&*src, &cfg, &out, &GgufExport { dtype, tokenizer_dir: &tokenizer_dir, name: &name }).map_err(|e| format!("qwen export: {e}"))?;
        }
        other => return Err(format!("qwen export: format {other:?}: expected hf or gguf")),
    }
    Ok(Outcome::new().set("saved", json!(out.display().to_string())).set("format", json!(format)))
}

pub(crate) fn export_peft(_: &Resolved, inv: &Invocation, _: &mut dyn FnMut(Progress)) -> ActionResult {
    let adapter = path_param(inv, "adapter", "export_peft")?;
    let out = path_param(inv, "out", "export_peft")?;
    let base_model = inv.get_str("base_model").filter(|b| !b.is_empty());
    write_peft(&adapter.to_string_lossy(), &out, base_model.as_deref()).map_err(|e| format!("qwen export_peft: {e}"))?;
    Ok(Outcome::new().set("saved", json!(out.display().to_string())))
}
