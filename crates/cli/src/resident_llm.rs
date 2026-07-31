// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapters for brain's text-generation LLMs — GPT (dense
//! char-level baseline), GLM (MLA + noaux_tc MoE decoder), and Qwen3 (BPE
//! decoder) — behind the residency [`Executor`], mirroring the yolo/z-image
//! adapters in [`crate::resident`].
//!
//! Each model family is one [`ResidentModel`] with a single `"generate"` action.
//! Unlike yolo (which pins itself to the CPU via `Gpu::new_cpu` unless a
//! `--device` was chosen), these models load through `gpu_core::Gpu::new`, i.e.
//! the process-default backend — **wgpu (GPU) unless `BRAIN_DEVICE=cpu`**. So the
//! resident instance holds the model on a GPU (VRAM); dropping it frees the card.
//! `activate` places the build on the assigned card via a scoped device-registry
//! selection ([`on_device`]), exactly like z-image.
//!
//! Config is env-only: `BRAIN_GPT_WEIGHTS`, `BRAIN_GLM_WEIGHTS`,
//! `BRAIN_QWEN_WEIGHTS` + `BRAIN_QWEN_TOKENIZER` (and an optional
//! `BRAIN_QWEN_CTX`, default 2048, sizing Qwen's built context length). Each
//! `from_env` returns `None` when its primary weights var is unset/empty.

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress};
use checkpoint::st::ModelCard;
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;

use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};

// ---------------------------------------------------------------- shared

/// The shared `"generate"` action spec. `chat` adds Qwen's chat contract: the
/// chat-template toggle plus `messages`/`system`/`top_p`/`stop` and per-token
/// streaming (one `Progress::token` delta each accepted token).
fn generate_spec(summary: &str, chat: bool) -> ActionSpec {
    let mut s = ActionSpec::new("generate", summary)
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (or chat message)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(128)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.8)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (0 = disabled)").default(json!(40)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)));
    if chat {
        s = s
            .streaming()
            .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt").default(json!(true)))
            .param(ParamSpec::new("messages", ParamType::Str, "JSON array of {role,content} chat turns (overrides prompt)"))
            .param(ParamSpec::new("system", ParamType::Str, "optional system prompt prepended to the chat"))
            .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)))
            .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"));
    }
    s.output(BlobSpec::new("text", Media::Text, "the generated text"))
}

/// Parse the `messages` param (a JSON array of `{"role","content"}`) into role/
/// content pairs, prepending a leading `system` turn when one is supplied.
fn parse_messages(raw: &str, system: Option<&str>) -> Result<Vec<(String, String)>, String> {
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("qwen: messages JSON: {e}"))?;
    let arr = v.as_array().ok_or("qwen: messages must be a JSON array")?;
    let mut out = Vec::with_capacity(arr.len() + 1);
    if let Some(s) = system.filter(|s| !s.is_empty()) {
        out.push(("system".to_string(), s.to_string()));
    }
    for m in arr {
        let role = m.get("role").and_then(|r| r.as_str()).ok_or("qwen: message.role missing")?;
        let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
        out.push((role.to_string(), content.to_string()));
    }
    Ok(out)
}

/// Parse the `stop` param (a JSON array of strings) into non-empty stop strings.
fn parse_stops(raw: Option<&str>) -> Result<Vec<String>, String> {
    let Some(raw) = raw.filter(|s| !s.is_empty()) else { return Ok(Vec::new()) };
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("qwen: stop JSON: {e}"))?;
    let arr = v.as_array().ok_or("qwen: stop must be a JSON array")?;
    Ok(arr.iter().filter_map(|s| s.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect())
}

/// Split the freshly-decoded `full` text into the fragment safe to emit now and
/// the new "printed" prefix, holding back a trailing replacement char (an
/// incomplete multi-byte UTF-8 sequence awaiting the next token). `full` always
/// extends `printed`, so concatenating every emitted fragment — plus a final
/// flush of any held-back tail — reproduces `full` byte-for-byte.
fn stream_delta(printed: &str, full: &str) -> (String, String) {
    let safe = full.strip_suffix('\u{FFFD}').unwrap_or(full);
    let delta = safe.get(printed.len()..).unwrap_or("").to_string();
    (delta, safe.to_string())
}

/// If `text` ends with any stop string, the byte index where the earliest such
/// match begins (so `text[..idx]` is the truncated output); else `None`.
fn find_stop(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| text.ends_with(s.as_str()))
        .map(|s| text.len() - s.len())
        .min()
}

/// Read the four shared sampling params (with the spec defaults).
fn sampling_params(inv: &Invocation) -> (usize, f32, usize, u64) {
    let max_new = inv.get_i64("max_new").unwrap_or(128).max(0) as usize;
    let temp = inv.get_f64("temp").unwrap_or(0.8) as f32;
    let top_k = inv.get_i64("top_k").unwrap_or(40).max(0) as usize;
    let seed = inv.get_i64("seed").unwrap_or(0).max(0) as u64;
    (max_new, temp, top_k, seed)
}

/// Wrap generated text as a text-output [`Outcome`] (`text` value + `text` blob).
fn text_outcome(text: String) -> Outcome {
    Outcome::new().set("text", json!(text)).blob("text", Blob::new(Media::Text, text.into_bytes()))
}

/// Estimate the Hot VRAM footprint of a checkpoint as ~1.3x its file size.
fn est_vram(path: &str) -> MemCost {
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0).saturating_mul(13) / 10;
    MemCost::new(bytes, 0)
}

/// Run `f` placed on the residency-assigned device: a GPU assignment becomes a
/// scoped (thread-local) selection in the canonical device registry, so every
/// `Gpu::new` inside `f` binds that physical card — race-free across the
/// executor's concurrent activation lanes. Shared by the resident adapters.
pub(crate) fn on_device<R>(device: Device, f: impl FnOnce() -> R) -> Result<R, String> {
    match device {
        Device::Gpu(i) => gpu_core::devices::with_gpu(i, f),
        _ => Ok(f()),
    }
}

// ---------------------------------------------------------------- gpt

/// The dense char-level GPT baseline behind the scheduler (`BRAIN_GPT_WEIGHTS`).
/// The checkpoint must embed its char vocab (trained with vocab embedding).
pub struct GptResident {
    /// Catalog id (the model-card id): the manifest/instance-key key, so two
    /// checkpoints of the same family are two distinct selectable models.
    id: String,
    path: String,
}

impl GptResident {
    pub fn from_env() -> Option<GptResident> {
        let path = std::env::var("BRAIN_GPT_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        // Back-compat: synthesize a card whose id is the family constant.
        Some(Self::from_card(&path, &ModelCard::new("gpt", "gpt"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused — GPT is char-level
    /// (its vocab is embedded in the checkpoint).
    pub fn from_card(path: &str, card: &ModelCard, _tokenizer: Option<&str>) -> GptResident {
        GptResident { id: card.id.clone(), path: path.to_string() }
    }
}

impl ResidentModel for GptResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "text generation (dense char-level GPT)", vec![generate_spec("generate text continuing a prompt (char-level GPT)", false)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Stream weights from the mmap: peak host allocation is ~one tensor, not
        // a whole-model f32 copy on top of the device weights. One reader serves
        // the vocab, the config, and the tensor upload.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("gpt: {e}"))?;
        let itos = gpt::model::Gpt::itos_from_config(&reader.config())
            .ok_or("gpt: checkpoint has no embedded char vocab (BRAIN_GPT_WEIGHTS)")?;
        let tok = CharTokenizer::from_itos(itos);
        let block = gpt::GptConfig::from_json(&reader.config()).block_size;
        let model = on_device(device, || gpt::model::Gpt::from_reader(&reader, 1, block))?;
        Ok(Box::new(GptInstance { model, tok }))
    }
}

struct GptInstance {
    model: gpt::model::Gpt,
    tok: CharTokenizer,
}

impl Instance for GptInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress::step(0, max_new as u32, "generating"));
        let gen = gpt::sample::generate(&self.model, &ids, max_new, temp, top_k, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));
        Ok(text_outcome(text))
    }
}

// ---------------------------------------------------------------- glm

/// The GLM decoder (MLA + sigmoid noaux_tc MoE) behind the scheduler
/// (`BRAIN_GLM_WEIGHTS`). Char-level: the checkpoint must embed its vocab.
pub struct GlmResident {
    id: String,
    path: String,
}

impl GlmResident {
    pub fn from_env() -> Option<GlmResident> {
        let path = std::env::var("BRAIN_GLM_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        Some(Self::from_card(&path, &ModelCard::new("glm", "glm"), None))
    }

    /// Construct under the card's id. `_tokenizer` is unused — GLM is char-level.
    pub fn from_card(path: &str, card: &ModelCard, _tokenizer: Option<&str>) -> GlmResident {
        GlmResident { id: card.id.clone(), path: path.to_string() }
    }
}

impl ResidentModel for GlmResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "text generation (GLM MLA + MoE decoder)", vec![generate_spec("generate text continuing a prompt (GLM decoder)", false)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        // Stream weights from the mmap (see GptResident::activate).
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("glm: {e}"))?;
        let itos = glm::model::Glm::itos_from_config(&reader.config())
            .ok_or("glm: checkpoint has no embedded char vocab (BRAIN_GLM_WEIGHTS)")?;
        let tok = CharTokenizer::from_itos(itos);
        let block = glm::config::GlmConfig::from_json(&reader.config()).block_size;
        let model = on_device(device, || glm::model::Glm::from_reader_inference(&reader, 1, block))?;
        Ok(Box::new(GlmInstance { model, tok }))
    }
}

struct GlmInstance {
    model: glm::model::Glm,
    tok: CharTokenizer,
}

impl Instance for GlmInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = self.tok.encode(&prompt_text);
        let mut rng = Rng::new(seed);
        progress(Progress::step(0, max_new as u32, "generating"));
        let gen = glm::sample::generate(&self.model, &ids, max_new, temp, top_k, None, &mut rng);
        let text = self.tok.decode(&gen);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));
        Ok(text_outcome(text))
    }
}

// ---------------------------------------------------------------- qwen

/// The Qwen3 BPE decoder behind the scheduler (`BRAIN_QWEN_WEIGHTS` +
/// `BRAIN_QWEN_TOKENIZER`). Runs the CPU/GPU forward `generate` path (never the
/// NPU branch). `BRAIN_QWEN_CTX` (default 2048) sizes the built context length.
pub struct QwenResident {
    id: String,
    path: String,
    tokenizer: String,
}

impl QwenResident {
    pub fn from_env() -> Option<QwenResident> {
        let path = std::env::var("BRAIN_QWEN_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let tokenizer = std::env::var("BRAIN_QWEN_TOKENIZER").ok().unwrap_or_default();
        Some(Self::from_card(&path, &ModelCard::new("qwen", "qwen"), Some(&tokenizer)))
    }

    /// Construct under the card's id. `tokenizer` is the sibling `tokenizer.json`
    /// (empty/None defers the "set a tokenizer" error to `activate`).
    pub fn from_card(path: &str, card: &ModelCard, tokenizer: Option<&str>) -> QwenResident {
        QwenResident { id: card.id.clone(), path: path.to_string(), tokenizer: tokenizer.unwrap_or_default().to_string() }
    }

    fn ctx() -> u32 {
        std::env::var("BRAIN_QWEN_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(2048u32).max(1)
    }
}

impl ResidentModel for QwenResident {
    fn manifest(&self) -> Manifest {
        Manifest::new(&self.id, "text generation (Qwen3 BPE decoder)", vec![generate_spec("generate text (Qwen3; chat template optional)", true)])
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(self.id.as_str(), "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        est_vram(&self.path)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if self.tokenizer.is_empty() {
            return Err("qwen: set BRAIN_QWEN_TOKENIZER to the tokenizer.json path".to_string());
        }
        let tok = data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?;
        let eos = tok.encode("<|im_end|>").first().copied();
        // Stream weights from the mmap (see GptResident::activate).
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen: {e}"))?;
        let model = on_device(device, || qwen::model::Qwen::from_reader_inference(&reader, 1, Self::ctx()))?;
        Ok(Box::new(QwenInstance { model, tok, eos }))
    }
}

struct QwenInstance {
    model: qwen::model::Qwen,
    tok: data::qwen_tokenizer::QwenBpe,
    eos: Option<u32>,
}

impl Instance for QwenInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (max_new, temp, top_k, seed) = sampling_params(inv);
        let top_p = inv.get_f64("top_p").unwrap_or(1.0) as f32;
        // Build the prompt text: `messages` (chat template) wins; else the legacy
        // single-`prompt` (+ `chat`) path, preserving existing numerics.
        let text = match inv.get_str("messages").filter(|s| !s.is_empty()) {
            Some(raw) => {
                let msgs = parse_messages(&raw, inv.get_str("system").as_deref())?;
                let refs: Vec<(&str, &str)> = msgs.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
                self.tok.apply_chat_template(&refs, true)
            }
            None => {
                let prompt = inv.get_str("prompt").unwrap_or_default();
                if inv.get_bool("chat").unwrap_or(true) {
                    self.tok.apply_chat_template(&[("user", &prompt)], true)
                } else {
                    prompt
                }
            }
        };
        let ids = self.tok.encode(&text);
        if ids.is_empty() {
            return Err("qwen: empty prompt".to_string());
        }
        let stops = parse_stops(inv.get_str("stop").as_deref())?;

        let mut rng = Rng::new(seed);
        let total = max_new as u32;
        progress(Progress::step(0, total, "generating"));

        // Stream: decode each accepted token to its human-visible delta, honour
        // stop-strings and cancellation, and track usage.
        let tok = &self.tok;
        let cancel = inv.cancel.clone();
        let mut ids_out: Vec<u32> = Vec::with_capacity(max_new);
        let mut printed = String::new();
        let mut stop_at: Option<usize> = None;
        let mut cancelled = false;
        let gen = qwen::sample::generate_kv_stream(
            &self.model,
            &ids,
            max_new,
            temp,
            top_k,
            top_p,
            self.eos,
            &mut rng,
            &mut |i, t| {
                ids_out.push(t);
                let full = tok.decode(&ids_out);
                let (delta, new_printed) = stream_delta(&printed, &full);
                printed = new_printed;
                if !delta.is_empty() {
                    progress(Progress::token(i as u32 + 1, total, delta));
                }
                if let Some(idx) = find_stop(&printed, &stops) {
                    stop_at = Some(idx);
                    return false;
                }
                if cancel.is_cancelled() {
                    cancelled = true;
                    return false;
                }
                true
            },
        );

        // Final text + finish reason. A stop-string truncates the visible text; a
        // clean end flushes any held-back tail so the deltas reconstruct it.
        let (text, finish) = if let Some(idx) = stop_at {
            (printed[..idx].to_string(), "stop_sequence")
        } else {
            let full = self.tok.decode(&ids_out);
            if full.len() > printed.len() {
                progress(Progress::token(ids_out.len() as u32, total, full[printed.len()..].to_string()));
            }
            let reason = if cancelled {
                "stop"
            } else if gen.len() >= max_new {
                "length"
            } else {
                "stop" // eos
            };
            (full, reason)
        };
        progress(Progress::step(total, total, "done"));

        let mut out = text_outcome(text);
        out = out
            .set("prompt_tokens", json!(ids.len() as i64))
            .set("completion_tokens", json!(gen.len() as i64))
            .set("finish_reason", json!(finish));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_parse_with_and_without_system() {
        let raw = r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]"#;
        let m = parse_messages(raw, None).unwrap();
        assert_eq!(m, vec![("user".into(), "hi".into()), ("assistant".into(), "yo".into())]);
        let m = parse_messages(raw, Some("be terse")).unwrap();
        assert_eq!(m[0], ("system".to_string(), "be terse".to_string()));
        assert_eq!(m.len(), 3);
        // an empty system turn is dropped; malformed JSON is a clean error.
        assert_eq!(parse_messages(raw, Some("")).unwrap().len(), 2);
        assert!(parse_messages("not json", None).is_err());
        assert!(parse_messages("{}", None).is_err());
    }

    #[test]
    fn stops_parse_and_match() {
        assert_eq!(parse_stops(None).unwrap(), Vec::<String>::new());
        assert_eq!(parse_stops(Some(r#"["\n\n","END"]"#)).unwrap(), vec!["\n\n".to_string(), "END".to_string()]);
        assert!(parse_stops(Some("nope")).is_err());
        let stops = vec!["END".to_string(), "STOP".to_string()];
        assert_eq!(find_stop("all done END", &stops), Some(9));
        assert_eq!(find_stop("mid END dle", &stops), None); // only a trailing match
        assert_eq!(find_stop("nothing here", &stops), None);
        // earliest boundary wins when stops overlap at the tail.
        let overlap = vec!["done".to_string(), "all done".to_string()];
        assert_eq!(find_stop("all done", &overlap), Some(0));
    }

    /// The delta bookkeeping used by `run`: concatenated per-token deltas (plus a
    /// final flush of a held-back tail) must reproduce the full decoded text —
    /// even when a multi-byte char is split across two tokens (transient U+FFFD).
    #[test]
    fn stream_deltas_reconstruct_full_text() {
        // Simulated per-step `decode(ids_out)` outputs, incl. a split euro sign.
        let steps = ["Hi", "Hi\u{FFFD}", "Hi€", "Hi€!"];
        let mut printed = String::new();
        let mut concat = String::new();
        for full in steps {
            let (delta, np) = stream_delta(&printed, full);
            concat.push_str(&delta);
            printed = np;
        }
        let full = *steps.last().unwrap();
        if full.len() > printed.len() {
            concat.push_str(&full[printed.len()..]); // final flush
        }
        assert_eq!(concat, "Hi€!");
        // No replacement char ever escapes into an emitted fragment.
        assert!(!concat.contains('\u{FFFD}'));
    }
}
