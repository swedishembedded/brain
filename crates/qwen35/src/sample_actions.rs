// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `sample_adapter` action of `brain/qwen35`: what a trained LoRA adapter
//! makes Qwen3.8-27B say, on a base that is streamed onto the device at a
//! narrow format with the adapter overlaid - never folded into a host copy,
//! which the 27B does not fit in the job's memory.
//!
//! `mode` picks which decodes each prompt gets: `adapter` (the overlay on),
//! `base` (the overlay switched off, the base model exactly) or `compare`
//! (both, so a change is read off the two answers). Decoding is greedy, by
//! repeated full forwards at one fixed length with no KV cache: slow, but
//! exactly the forward the adapter was trained through, which is what a
//! content check of an adapter needs. The text-generation action with a KV
//! cache is `generate`, which takes the same adapter folded into the weights.
//!
//! Swedish Embedded AB implements solutions for showing that a fine-tuned
//! language model learned what its dataset taught it, for its clients. If your
//! team needs expertise in evaluating adapters on very large models then you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::select::Dtype;
use serde_json::json;

use crate::adapter_sample::AdapterSampler;
use crate::base_source::TRAIN_BASES;

const DEFAULT_MAX_NEW: i64 = 32;
/// Most prompts one call decodes: each is a slow full-forward decode.
const MAX_PROMPTS: usize = 256;
const MODES: [&str; 3] = ["adapter", "base", "compare"];

pub(crate) fn sample_adapter_spec() -> ActionSpec {
    ActionSpec::new(
        "sample_adapter",
        "greedy answers of a streamed base (nf4, bf16, fp8 or fp32) with a trained LoRA adapter overlaid: the adapter's, the base's, or both side by side (full forwards at a fixed length, no KV cache)",
    )
    .streaming()
    .cancellable()
    .param(
        ParamSpec::new("weights", ParamType::Str, "path to the base: a Hugging Face release directory (FP8 included) or a brain checkpoint (.safetensors); overrides the model store's own pick when set")
            .host_resolved(),
    )
    .param(
        ParamSpec::new("tokenizer", ParamType::Str, "path to the base's tokenizer.json; omit for the tokenizer.json inside the base directory, else the model store's own pick")
            .host_resolved(),
    )
    .param(ParamSpec::new("adapter", ParamType::Str, "path to a LoRA adapter (.safetensors) written by `finetune`; needed unless `mode` is base").host_resolved())
    .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue; `prompts` adds more"))
    .param(ParamSpec::new("prompts", ParamType::Str, "JSON array of prompts, decoded after `prompt`"))
    .param(
        ParamSpec::new("base_format", ParamType::Enum(TRAIN_BASES.bases.iter().map(|d| model::policy::format_name(*d).to_string()).collect()), "the format the frozen base is held in on the device")
            .default(json!("nf4")),
    )
    .param(ParamSpec::new("mode", ParamType::Enum(MODES.map(String::from).to_vec()), "adapter: the overlay on; base: the overlay off; compare: both answers per prompt").default(json!("adapter")))
    .param(ParamSpec::new("max_new", ParamType::Int, "tokens to generate per prompt; the forward length is the longest prompt plus these").default(json!(DEFAULT_MAX_NEW)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
    .param(ParamSpec::new("chat", ParamType::Bool, "wrap each prompt in the chat template").default(json!(false)))
    .output(BlobSpec::new("text", Media::Text, "one `[adapter] <prompt><answer>` or `[base] ...` line per decode"))
}

/// Which decodes each prompt gets.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum SampleWhich {
    AdapterOnly,
    BaseOnly,
    Both,
}

/// Everything a `sample_adapter` invocation asks for.
#[derive(Debug)]
pub(crate) struct SampleRequest {
    pub prompts: Vec<String>,
    pub base_dtype: Dtype,
    pub max_new: usize,
    pub chat: bool,
    pub which: SampleWhich,
}

impl SampleRequest {
    /// The request `inv` describes. `adapter` says whether an adapter was named.
    pub(crate) fn from_invocation(inv: &Invocation, adapter: bool) -> Result<SampleRequest, String> {
        let fail = |why: String| format!("qwen35 sample_adapter: {why}");
        let mut prompts: Vec<String> = inv.get_str("prompt").filter(|p| !p.is_empty()).into_iter().collect();
        if let Some(more) = inv.get_str("prompts").filter(|p| !p.is_empty()) {
            let more: Vec<String> = serde_json::from_str(&more).map_err(|e| fail(format!("prompts must be a JSON array of strings: {e}")))?;
            prompts.extend(more);
        }
        if prompts.is_empty() {
            return Err(fail("at least one prompt is required (`prompt` or `prompts`)".into()));
        }
        if prompts.len() > MAX_PROMPTS {
            return Err(fail(format!("{} prompts is more than the {MAX_PROMPTS} one call decodes", prompts.len())));
        }
        let base_dtype = qwen3::lora_params::base_format(inv, &TRAIN_BASES, Dtype::NF4).map_err(&fail)?;
        let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW);
        if max_new < 1 {
            return Err(fail("max_new must be at least 1".into()));
        }
        let which = match inv.get_str("mode").as_deref().unwrap_or("adapter") {
            "adapter" => SampleWhich::AdapterOnly,
            "base" => SampleWhich::BaseOnly,
            "compare" => SampleWhich::Both,
            other => return Err(fail(format!("mode {other:?}: expected adapter, base or compare"))),
        };
        if !adapter && which != SampleWhich::BaseOnly {
            return Err(fail("`adapter` is required unless mode base names the base alone".into()));
        }
        Ok(SampleRequest { prompts, base_dtype, max_new: max_new as usize, chat: inv.get_bool("chat").unwrap_or(false), which })
    }
}

/// The tokenizer of the base at `weights`: a named one, else the
/// `tokenizer.json` inside the base directory (or beside a checkpoint file),
/// else the model store's own pick.
fn tokenizer_path(inv: &Invocation, weights: &Path, default_tokenizer: Option<&str>) -> PathBuf {
    if let Some(named) = inv.get_str("tokenizer").filter(|p| !p.is_empty()) {
        return PathBuf::from(named);
    }
    let beside = (if weights.is_dir() { weights } else { weights.parent().unwrap_or(Path::new(".")) }).join("tokenizer.json");
    match default_tokenizer {
        Some(resolved) if !beside.is_file() => PathBuf::from(resolved),
        _ => beside,
    }
}

pub(crate) fn sample_adapter(default_weights: Option<&str>, default_tokenizer: Option<&str>, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let fail = |why: String| format!("qwen35 sample_adapter: {why}");
    let adapter = inv.get_str("adapter").filter(|a| !a.is_empty()).map(PathBuf::from);
    let request = SampleRequest::from_invocation(inv, adapter.is_some())?;
    let weights = inv
        .get_str("weights")
        .filter(|w| !w.is_empty())
        .or_else(|| default_weights.map(String::from))
        .map(PathBuf::from)
        .ok_or_else(|| fail("no base checkpoint (pass 'weights', or pull one into the models directory)".into()))?;
    if !weights.exists() {
        return Err(fail(format!("weights not found at '{}'", weights.display())));
    }
    let tok_path = tokenizer_path(inv, &weights, default_tokenizer);
    let tok = QwenBpe::from_file(&tok_path.to_string_lossy()).map_err(|e| fail(format!("tokenizer {}: {e}", tok_path.display())))?;
    let encoded: Vec<Vec<u32>> = request.prompts.iter().map(|p| tok.encode(&if request.chat { tok.apply_chat_template(&[("user", p)], true) } else { p.clone() })).collect();
    let longest = encoded.iter().map(Vec::len).max().unwrap_or(0);
    if longest == 0 {
        return Err(fail("a prompt encodes to no tokens".into()));
    }
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.encode(s).first().copied()).collect();

    let len = u32::try_from(longest + request.max_new).map_err(|_| fail("the forward length is beyond any context".into()))?;
    progress(Progress::step(0, request.prompts.len() as u32, format!("loading the base as {}, forward length {len}", model::policy::format_name(request.base_dtype))));
    let mut sampler = AdapterSampler::open(&weights, adapter.as_deref(), request.base_dtype, len).map_err(&fail)?;
    let modes: &[(&str, bool)] = match request.which {
        SampleWhich::AdapterOnly => &[("adapter", true)],
        SampleWhich::BaseOnly => &[("base", false)],
        SampleWhich::Both => &[("adapter", true), ("base", false)],
    };
    let (mut lines, mut answers) = (Vec::new(), Vec::new());
    for (done, (prompt, ids)) in request.prompts.iter().zip(&encoded).enumerate() {
        if inv.cancel.is_cancelled() {
            return Err(format!("qwen35 sample_adapter: cancelled after {done} of {} prompts", request.prompts.len()));
        }
        let mut answer = serde_json::Map::new();
        answer.insert("prompt".into(), json!(prompt));
        for (label, on) in modes {
            sampler.set_adapter(*on);
            let out = sampler.greedy(ids, request.max_new, &eos).map_err(&fail)?;
            let text = tok.decode(&out);
            lines.push(format!("[{label}] {prompt}{text}"));
            answer.insert((*label).into(), json!(text));
        }
        answers.push(serde_json::Value::Object(answer));
        progress(Progress::step(done as u32 + 1, request.prompts.len() as u32, "prompt"));
    }
    let text = lines.join("\n");
    Ok(Outcome::new().set("answers", json!(answers)).set("text", json!(text)).blob("text", Blob::new(Media::Text, text.into_bytes())))
}

/// A `tokenizer.json` whose 28 ids (a-z, space, newline) fit the tiny vocabulary.
#[cfg(test)]
pub(crate) fn write_letter_tokenizer(dir: &Path) {
    let mut vocab = serde_json::Map::new();
    for (i, b) in (b'a'..=b'z').enumerate() {
        vocab.insert((b as char).to_string(), json!(i));
    }
    vocab.insert('\u{0120}'.to_string(), json!(26)); // space
    vocab.insert('\u{010a}'.to_string(), json!(27)); // newline
    std::fs::write(dir.join("tokenizer.json"), json!({"model": {"vocab": vocab, "merges": []}}).to_string()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(params: &[(&str, serde_json::Value)], adapter: bool) -> Result<SampleRequest, String> {
        let inv = params.iter().fold(Invocation::new(), |inv, (name, value)| inv.set(name, value.clone()));
        SampleRequest::from_invocation(&sample_adapter_spec().validate(inv)?, adapter)
    }

    #[test]
    fn the_sample_params_parse_and_refuse_contradictions() {
        let ok = request(&[("prompt", json!("one")), ("prompts", json!(r#"["two"]"#)), ("max_new", json!(9)), ("mode", json!("compare"))], true).unwrap();
        assert_eq!((ok.prompts.len(), ok.max_new, ok.which, ok.base_dtype), (2, 9, SampleWhich::Both, Dtype::NF4));
        let base_only = request(&[("prompt", json!("q")), ("mode", json!("base"))], false).unwrap();
        assert_eq!(base_only.which, SampleWhich::BaseOnly);
        assert!(request(&[("prompt", json!("q"))], false).unwrap_err().contains("adapter"), "an adapter-on run needs an adapter");
        assert!(request(&[("prompt", json!("q")), ("max_new", json!(0))], true).is_err(), "max_new is at least 1");
        assert!(sample_adapter_spec().validate(Invocation::new().set("prompt", json!("q")).set("typo", json!(1))).is_err(), "an unknown param is refused by name");
        assert!(request(&[("mode", json!("base"))], false).unwrap_err().contains("prompt"));
        assert!(request(&[("prompts", json!("not json"))], true).unwrap_err().contains("JSON array"));
    }

    /// The overlay switched on or off is the `mode` param: both decodes of a prompt come back, labelled.
    #[test]
    fn base_and_adapter_decodes_are_labelled_by_mode() {
        let dir = std::env::temp_dir().join(format!("qwen35-sample-modes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = crate::config::Qwen35Config::tiny();
        let init = crate::init::init_weights(&cfg, 3); // seed: any weights decode; this checks the plumbing
        crate::model::Qwen35::new_on(gpu_core::Gpu::new(crate::model::pipelines()), cfg.clone(), 1, cfg.block_size, &init).save(dir.join("base.safetensors").to_str().unwrap());
        write_letter_tokenizer(&dir);
        let run = |extra: &[(&str, serde_json::Value)]| {
            let inv = extra.iter().fold(Invocation::new().set("weights", json!(dir.join("base.safetensors").to_str().unwrap())).set("prompt", json!("abc")).set("max_new", json!(2)).set("base_format", json!("f32")), |inv, (n, v)| inv.set(n, v.clone()));
            sample_adapter(None, None, &sample_adapter_spec().validate(inv).unwrap(), &mut |_| {})
        };
        let base = run(&[("mode", json!("base"))]).unwrap();
        let answers = base.outputs["answers"].as_array().unwrap();
        assert!(answers[0].get("base").is_some() && answers[0].get("adapter").is_none(), "{answers:?}");
        assert!(run(&[]).unwrap_err().contains("adapter"), "the default mode needs an adapter");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
