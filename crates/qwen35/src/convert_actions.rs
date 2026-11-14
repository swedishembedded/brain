// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `import` action of `brain/qwen35`: a release as it is published into
//! the form the model's own trainers and servers read.
//!
//! - A GGUF (`general.architecture = "qwen35"`) becomes a brain checkpoint
//!   (`.safetensors`, carded for the model store), MTP head included - the
//!   importer the generic `brain import-gguf` dispatches to.
//! - A Hugging Face release directory (the FP8 release included) becomes a
//!   converted base directory: its layer matrices are re-encoded once at
//!   `weight_format` (nf4 by default), tile by tile with bounded host memory,
//!   so a LoRA run on it copies the stored codes instead of requantizing the
//!   FP8 on every load.
//!
//! Swedish Embedded AB implements checkpoint conversion between published
//! formats and the formats a training and serving stack reads, for its
//! clients. If your team needs expertise in loading very large quantized
//! models within a fixed memory budget then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use capability::{ActionResult, ActionSpec, Invocation, Outcome, ParamSpec, ParamType, Progress};
use serde_json::json;

pub(crate) fn import_spec() -> ActionSpec {
    ActionSpec::new(
        "import",
        "convert a Qwen3.8-27B release into the form the trainers and servers read: a GGUF into a brain checkpoint (MTP head included), or a Hugging Face release directory into a base re-encoded once at a narrow format",
    )
    .from_scratch()
    .param(ParamSpec::new("source", ParamType::Str, "path to the release: a .gguf file, or a Hugging Face directory (config.json and its safetensors)").host_resolved().required())
    .param(ParamSpec::new("dest", ParamType::Str, "path to write: the brain checkpoint (.safetensors) for a GGUF, a new directory for a Hugging Face release").host_resolved().required())
    .param(ParamSpec::new("id", ParamType::Str, "the model id recorded on the checkpoint's card (a GGUF source; default qwen35)"))
    .param(ParamSpec::new(
        "weight_format",
        ParamType::Str,
        "the storage format of the layer matrices of a Hugging Face source: nf4 (default), bf16, f32, ..., or nf4 with per-tensor rules such as nf4,self_attn=f8e4m3,norm=f32",
    ))
}

pub(crate) fn import(inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let fail = |why: String| format!("qwen35 import: {why}");
    let path = |name: &str| inv.get_str(name).filter(|p| !p.is_empty()).map(PathBuf::from).ok_or_else(|| fail(format!("missing required param '{name}'")));
    let (source, dest) = (path("source")?, path("dest")?);
    if !source.exists() {
        return Err(fail(format!("source not found at '{}'", source.display())));
    }
    progress(Progress::step(0, 1, format!("converting {}", source.display())));
    let outcome = if source.is_dir() {
        convert_hf(&source, &dest, inv.get_str("weight_format").filter(|f| !f.is_empty()).as_deref().unwrap_or("nf4")).map_err(&fail)?
    } else if source.extension().is_some_and(|e| e == "gguf") {
        let id = inv.get_str("id").filter(|i| !i.is_empty());
        let stats = crate::gguf_import::import_gguf(&source.to_string_lossy(), &dest.to_string_lossy(), id.as_deref()).map_err(&fail)?;
        Outcome::new().set("tensors", json!(stats.written))
    } else {
        return Err(fail(format!("source '{}' is neither a .gguf file nor a Hugging Face directory", source.display())));
    };
    progress(Progress::step(1, 1, "converted"));
    Ok(outcome.set("saved", json!(dest.display().to_string())))
}

/// A Hugging Face release directory re-encoded into `dest`.
fn convert_hf(source: &Path, dest: &Path, weight_format: &str) -> Result<Outcome, String> {
    let cfg = crate::fp8_source::hf_config(source)?;
    let block = crate::fp8_source::hf_fp8_block_size(source)?;
    let tiers = model::ops::TierPolicy::parse(weight_format).map_err(|e| format!("weight_format {weight_format:?}: {e}"))?;
    let report = crate::fp8_source::write_converted_base(source, &cfg, block, dest, &tiers)?;
    Ok(Outcome::new()
        .set("tensors", json!(report.tensors))
        .set("converted", json!(report.converted))
        .set("payload_bytes", json!(report.payload_bytes_written))
        .set("peak_staging_bytes", json!(report.peak_staging_bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(params: &[(&str, serde_json::Value)]) -> ActionResult {
        let inv = params.iter().fold(Invocation::new(), |inv, (name, value)| inv.set(name, value.clone()));
        import(&import_spec().validate(inv)?, &mut |_| {})
    }

    #[test]
    fn a_gguf_becomes_a_carded_brain_checkpoint_with_its_mtp_head() {
        let dir = std::env::temp_dir().join(format!("qwen35-import-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("tiny.gguf");
        crate::gguf_import::testing::write_synthetic_gguf(gguf.to_str().unwrap());
        let dest = dir.join("tiny.safetensors");
        let out = run(&[("source", json!(gguf.to_str().unwrap())), ("dest", json!(dest.to_str().unwrap())), ("id", json!("lab/tiny"))]).unwrap();
        assert_eq!(out.outputs["saved"], json!(dest.display().to_string()));
        let card = checkpoint::st::read_card(dest.to_str().unwrap()).unwrap().expect("the checkpoint carries a card");
        assert_eq!((card.id.as_str(), card.family.as_str()), ("lab/tiny", "qwen35"));
        let st = checkpoint::st::load_safetensors(dest.to_str().unwrap()).unwrap();
        assert!(st.tensors.keys().any(|n| n.starts_with("mtp.")), "the MTP head is imported");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_source_that_is_neither_a_gguf_nor_a_directory_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("qwen35-import-refused-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let odd = dir.join("weights.bin");
        std::fs::write(&odd, b"x").unwrap();
        let err = run(&[("source", json!(odd.to_str().unwrap())), ("dest", json!("out"))]).unwrap_err();
        assert!(err.contains("neither a .gguf file nor a Hugging Face directory"), "{err}");
        let err = run(&[("source", json!("/nonexistent/qwen35.gguf")), ("dest", json!("out"))]).unwrap_err();
        assert!(err.contains("not found"), "{err}");
        let err = run(&[("source", json!(dir.to_str().unwrap())), ("dest", json!(dir.join("out").to_str().unwrap())), ("weight_format", json!("nf9"))]).unwrap_err();
        assert!(err.contains("config") || err.contains("weight_format"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
