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

use data::qwen_chat::{self, ChatEvent, ChatMessage, ChatScanner, Role, ToolCallMsg};
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
            .param(ParamSpec::new(
                "messages",
                ParamType::Str,
                "JSON array of {role,content,reasoning_content?,tool_calls?,tool_call_id?} chat turns (overrides prompt)",
            ))
            .param(ParamSpec::new("system", ParamType::Str, "optional system prompt prepended to the chat"))
            .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (>= 1 = disabled)").default(json!(1.0)))
            .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"))
            .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema)"))
            .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{...})"))
            .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block").default(json!(true)));
    }
    s.output(BlobSpec::new("text", Media::Text, "the generated text"))
}

/// Parse the `messages` param into [`ChatMessage`]s, prepending a leading `system`
/// turn when one is supplied. Back-compatible with the plain `{"role","content"}`
/// shape (the whole surface before this milestone); additionally reads, when
/// present, `reasoning_content` (string), `tool_call_id` (string, meaningful on
/// `role:"tool"`), and `tool_calls` (an array of `{id,name,arguments}` — the flat
/// shape `crates/apiserve`'s message-flattening emits — or, tolerated for direct
/// callers, the nested OpenAI `{id, function:{name,arguments}}` shape; `arguments`
/// is read as-is if it's already a JSON string, else re-serialized from whatever
/// JSON value is there).
fn parse_chat_messages(raw: &str, system: Option<&str>) -> Result<Vec<ChatMessage>, String> {
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("qwen: messages JSON: {e}"))?;
    let arr = v.as_array().ok_or("qwen: messages must be a JSON array")?;
    let mut out = Vec::with_capacity(arr.len() + 1);
    if let Some(s) = system.filter(|s| !s.is_empty()) {
        out.push(ChatMessage::system(s));
    }
    for m in arr {
        let role_str = m.get("role").and_then(|r| r.as_str()).ok_or("qwen: message.role missing")?;
        let role = match role_str {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            other => return Err(format!("qwen: unknown message.role {other:?}")),
        };
        let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
        let reasoning_content = m.get("reasoning_content").and_then(|c| c.as_str()).map(str::to_string);
        let tool_call_id = m.get("tool_call_id").and_then(|c| c.as_str()).map(str::to_string);
        let tool_calls = match m.get("tool_calls").and_then(|c| c.as_array()) {
            Some(calls) => calls.iter().map(parse_tool_call_msg).collect(),
            None => Vec::new(),
        };
        out.push(ChatMessage { role, content, reasoning_content, tool_calls, tool_call_id });
    }
    Ok(out)
}

/// One `tool_calls[]` element (flat `{id,name,arguments}` or nested OpenAI
/// `{id, function:{name,arguments}}`) into a [`ToolCallMsg`]. `arguments` passes
/// through unchanged if it's already a JSON string (the exact-bytes contract
/// [`ToolCallMsg::arguments`] documents); any other JSON value is re-serialized to
/// text via `serde_json` — acceptable here since this is deserialize-then-reserialize
/// bookkeeping, not the byte-exact rendering path (see `qwen_chat`'s module docs).
fn parse_tool_call_msg(c: &serde_json::Value) -> ToolCallMsg {
    let id = c.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let name = c
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| c.get("function").and_then(|f| f.get("name")).and_then(|v| v.as_str()))
        .unwrap_or_default()
        .to_string();
    let args = c.get("arguments").or_else(|| c.get("function").and_then(|f| f.get("arguments")));
    let arguments = match args {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => "{}".to_string(),
    };
    ToolCallMsg { id, name, arguments }
}

/// Parse the `tools` param (a JSON array of OpenAI-shaped tool-schema objects) into
/// one raw JSON text per tool. Re-serializes each element via plain `serde_json`
/// (NOT `qwen_chat`'s Python-`json.dumps`-exact `json_py`, which is crate-private to
/// `brain-data`) — exact byte-for-byte matching only matters inside the prompt
/// renderer itself, which re-normalizes every tool's text through its own
/// `json_py::dumps` when it emits the `<tools>` block (see `qwen_chat::render`).
fn parse_tools(raw: Option<&str>) -> Result<Vec<String>, String> {
    let Some(raw) = raw.filter(|s| !s.is_empty()) else { return Ok(Vec::new()) };
    let v: serde_json::Value = serde_json::from_str(raw).map_err(|e| format!("qwen: tools JSON: {e}"))?;
    let arr = v.as_array().ok_or("qwen: tools must be a JSON array")?;
    arr.iter().map(|t| serde_json::to_string(t).map_err(|e| format!("qwen: tools JSON: {e}"))).collect()
}

/// Forward a batch of [`ChatEvent`]s from the [`ChatScanner`] as [`Progress`]
/// ticks: visible [`ChatEvent::Content`] becomes [`Progress::token`] (exactly what
/// a no-tools request already streamed); reasoning/tool-call events become
/// [`Progress::event`] with a neutral, consistently-shaped JSON payload —
/// `{"kind":"reasoning","text":...}`, `{"kind":"tool_call_start","index":N,
/// "id":"call_N","name":...}`, `{"kind":"tool_call_args","index":N,"text":...}`,
/// `{"kind":"tool_call_end","index":N}`. The `id` mints as `call_<index>`,
/// matching [`ChatScanner::tool_calls`]'s own id convention (see
/// `qwen_chat::ChatScanner::close_call`) so a streamed id and the final
/// non-streaming `tool_calls` id always agree.
fn emit_chat_events(events: &[ChatEvent], progress: &mut dyn FnMut(Progress), step: u32, total: u32) {
    for ev in events {
        match ev {
            ChatEvent::Content(text) => {
                if !text.is_empty() {
                    progress(Progress::token(step, total, text.clone()));
                }
            }
            ChatEvent::Reasoning(text) => {
                progress(Progress::event(step, total, json!({ "kind": "reasoning", "text": text })));
            }
            ChatEvent::ToolCallStart { index, name } => {
                progress(Progress::event(
                    step,
                    total,
                    json!({ "kind": "tool_call_start", "index": index, "id": format!("call_{index}"), "name": name }),
                ));
            }
            ChatEvent::ToolCallArgs { index, fragment } => {
                progress(Progress::event(step, total, json!({ "kind": "tool_call_args", "index": index, "text": fragment })));
            }
            ChatEvent::ToolCallEnd { index } => {
                progress(Progress::event(step, total, json!({ "kind": "tool_call_end", "index": index })));
            }
        }
    }
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
        // Back-compat: synthesize a card whose id is the canonical brain/
        // fallback (see crates/modelref/src/alias.rs's module docs) -- a
        // checkpoint loaded straight from an env var carries no upstream
        // vendor/repo provenance to build a fully-qualified ref from.
        Some(Self::from_card(&path, &ModelCard::new("brain/gpt", "gpt"), None))
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
        // See GptResident::from_env's comment: env-loaded, no upstream provenance.
        Some(Self::from_card(&path, &ModelCard::new("brain/glm", "glm"), None))
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
        // See GptResident::from_env's comment: env-loaded, no upstream provenance.
        Some(Self::from_card(&path, &ModelCard::new("brain/qwen", "qwen"), Some(&tokenizer)))
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
        // Stream weights from the mmap (see GptResident::activate). Open first so
        // a GGUF can supply its own embedded tokenizer.
        let reader = checkpoint::weightio::WeightReader::open(&self.path).map_err(|e| format!("qwen: {e}"))?;
        // Tokenizer precedence: an explicit sibling `tokenizer.json` (safetensors
        // path, or an override) wins; else a `.gguf` builds from its embedded
        // `tokenizer.ggml.*` KV; else there is nothing to tokenize with.
        let tok = if !self.tokenizer.is_empty() {
            data::qwen_tokenizer::QwenBpe::from_file(&self.tokenizer)?
        } else if let Some(gt) = reader.tokenizer() {
            data::qwen_tokenizer::QwenBpe::from_gguf(&gt).map_err(|e| format!("qwen: {e}"))?
        } else {
            return Err("qwen: no tokenizer (set BRAIN_QWEN_TOKENIZER, or use a GGUF with an embedded tokenizer)".to_string());
        };
        let eos = tok.encode("<|im_end|>").first().copied();
        // Decode-only KV-cache load (see `Qwen::from_reader_decode`'s doc comment):
        // this instance only ever drives `generate_kv_stream` (incremental
        // `step`/`prefill`), never the batched forward `from_reader_inference` sizes
        // buffers for.
        let model = on_device(device, || qwen::model::Qwen::from_reader_decode(&reader, Self::ctx()))?;
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
        let enable_thinking = inv.get_bool("enable_thinking").unwrap_or(true);
        let tools = parse_tools(inv.get_str("tools").as_deref())?;
        // Build the prompt text via the byte-exact Jinja renderer (`qwen_chat`, not
        // the older plain-string `QwenBpe::apply_chat_template` — see that module's
        // doc comment on why they coexist): `messages` (chat template, tool-calling
        // and reasoning-aware) wins; else the legacy single-`prompt` (+ `chat`) path,
        // wrapped as a one-turn user message so it goes through the SAME renderer.
        let text = match inv.get_str("messages").filter(|s| !s.is_empty()) {
            Some(raw) => {
                let msgs = parse_chat_messages(&raw, inv.get_str("system").as_deref())?;
                qwen_chat::render_for_generation(&msgs, &tools, enable_thinking)?
            }
            None => {
                let prompt = inv.get_str("prompt").unwrap_or_default();
                if inv.get_bool("chat").unwrap_or(true) {
                    let msgs = [ChatMessage::user(prompt)];
                    qwen_chat::render_for_generation(&msgs, &tools, enable_thinking)?
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

        // `generate_kv_stream` wants a stop-id SET, not an `Option`.
        let eos_arr: [u32; 1];
        let eos_slice: &[u32] = match self.eos {
            Some(e) => {
                eos_arr = [e];
                &eos_arr
            }
            None => &[],
        };

        // Stream: decode each accepted token to its human-visible delta, run it
        // through the chat scanner (splitting `<think>`/`<tool_call>` markup out of
        // the raw text) so visible content, reasoning, and tool calls each stream as
        // their own event kind, honour stop-strings and cancellation, and track
        // usage.
        let tok = &self.tok;
        let cancel = inv.cancel.clone();
        let mut ids_out: Vec<u32> = Vec::with_capacity(max_new);
        let mut printed = String::new();
        let mut stop_at: Option<usize> = None;
        let mut cancelled = false;
        // `thinking_open=false`: the model must emit its own literal `<think>` tag
        // to think (no prefilled-open-think prefix on this path).
        let mut scan = ChatScanner::new(false);
        let mut evs: Vec<ChatEvent> = Vec::new();
        let gen = qwen::sample::generate_kv_stream(
            &self.model,
            &ids,
            max_new,
            temp,
            top_k,
            top_p,
            eos_slice,
            &mut rng,
            &mut |i, t| {
                ids_out.push(t);
                let full = tok.decode(&ids_out);
                let (delta, new_printed) = stream_delta(&printed, &full);
                printed = new_printed;
                if !delta.is_empty() {
                    evs.clear();
                    scan.push(&delta, &mut evs);
                    emit_chat_events(&evs, progress, i as u32 + 1, total);
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

        // Final text + finish reason. A stop-string truncates the visible RAW text
        // (existing behavior, kept as-is — a stop-string cutting mid-generation
        // takes precedence over any in-flight tool call, which is left unclosed and
        // unreported). A clean end flushes any held-back multi-byte tail through the
        // scanner, then [`ChatScanner::finish`] closes out a still-open tool call
        // (truncated by `max_new`) rather than silently dropping it.
        let (text, finish) = if let Some(idx) = stop_at {
            (printed[..idx].to_string(), "stop_sequence")
        } else {
            let full = self.tok.decode(&ids_out);
            if full.len() > printed.len() {
                let tail = full[printed.len()..].to_string();
                evs.clear();
                scan.push(&tail, &mut evs);
                emit_chat_events(&evs, progress, ids_out.len() as u32, total);
            }
            evs.clear();
            scan.finish(&mut evs);
            emit_chat_events(&evs, progress, ids_out.len() as u32, total);
            let reason = if !scan.tool_calls().is_empty() {
                "tool_calls"
            } else if cancelled {
                "stop"
            } else if gen.len() >= max_new {
                "length"
            } else {
                "stop" // eos
            };
            (scan.content().to_string(), reason)
        };
        progress(Progress::step(total, total, "done"));

        let mut out = text_outcome(text);
        out = out
            .set("prompt_tokens", json!(ids.len() as i64))
            .set("completion_tokens", json!(gen.len() as i64))
            .set("finish_reason", json!(finish))
            .set("reasoning_content", json!(scan.reasoning()));
        if !scan.tool_calls().is_empty() {
            let calls: Vec<serde_json::Value> =
                scan.tool_calls().iter().map(|c| json!({ "id": c.id, "name": c.name, "arguments": c.arguments })).collect();
            out = out.set("tool_calls", json!(serde_json::to_string(&calls).unwrap_or_else(|_| "[]".to_string())));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_parse_with_and_without_system() {
        let raw = r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]"#;
        let m = parse_chat_messages(raw, None).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!((m[0].role, m[0].content.as_str()), (Role::User, "hi"));
        assert_eq!((m[1].role, m[1].content.as_str()), (Role::Assistant, "yo"));
        let m = parse_chat_messages(raw, Some("be terse")).unwrap();
        assert_eq!((m[0].role, m[0].content.as_str()), (Role::System, "be terse"));
        assert_eq!(m.len(), 3);
        // an empty system turn is dropped; malformed JSON is a clean error.
        assert_eq!(parse_chat_messages(raw, Some("")).unwrap().len(), 2);
        assert!(parse_chat_messages("not json", None).is_err());
        assert!(parse_chat_messages("{}", None).is_err());
    }

    /// Back-compat + the new fields: `reasoning_content`, `tool_calls` (both the
    /// flat `{id,name,arguments}` shape and the nested OpenAI `{id,function:{...}}`
    /// shape), and `tool_call_id` all round-trip onto the parsed [`ChatMessage`].
    #[test]
    fn messages_parse_reads_reasoning_and_tool_calls() {
        let raw = r#"[
            {"role":"user","content":"weather?"},
            {"role":"assistant","content":"","reasoning_content":"thinking",
             "tool_calls":[{"id":"call_0","name":"get_weather","arguments":"{\"c\":\"Paris\"}"}]},
            {"role":"tool","content":"22C","tool_call_id":"call_0"},
            {"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"f","arguments":{"x":1}}}]}
        ]"#;
        let m = parse_chat_messages(raw, None).unwrap();
        assert_eq!(m.len(), 4);
        assert_eq!(m[1].reasoning_content.as_deref(), Some("thinking"));
        assert_eq!(m[1].tool_calls.len(), 1);
        assert_eq!(m[1].tool_calls[0].name, "get_weather");
        assert_eq!(m[1].tool_calls[0].arguments, r#"{"c":"Paris"}"#);
        assert_eq!(m[2].role, Role::Tool);
        assert_eq!(m[2].tool_call_id.as_deref(), Some("call_0"));
        // nested OpenAI shape with a JSON object `arguments` re-serializes to text.
        assert_eq!(m[3].tool_calls[0].name, "f");
        let args: serde_json::Value = serde_json::from_str(&m[3].tool_calls[0].arguments).unwrap();
        assert_eq!(args["x"], 1);
    }

    #[test]
    fn tools_parse_json_array_to_per_tool_text() {
        assert_eq!(parse_tools(None).unwrap(), Vec::<String>::new());
        assert_eq!(parse_tools(Some("")).unwrap(), Vec::<String>::new());
        let raw = r#"[{"type":"function","function":{"name":"a"}},{"type":"function","function":{"name":"b"}}]"#;
        let tools = parse_tools(Some(raw)).unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools[0].contains("\"a\""));
        assert!(tools[1].contains("\"b\""));
        assert!(parse_tools(Some("not json")).is_err());
        assert!(parse_tools(Some("{}")).is_err());
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
