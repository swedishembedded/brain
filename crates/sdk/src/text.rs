// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`TextGenerationPipeline`]: brain's text-generation surface, over qwen3
//! (the most complete decoder LM in this workspace) - loaded from EITHER a
//! literal local checkpoint path OR a `<vendor>/<repo>` hub id, through the
//! SAME [`TextGenerationPipeline::from_pretrained`] call (rule 2: never a
//! separate API for "load from disk" vs. "load from the hub").
//!
//! **How the two are told apart, unambiguously**: a string that names a
//! real file already on disk (`Path::new(s).is_file()`) is ALWAYS a local
//! path - never guessed at, never re-interpreted as a hub id even if it
//! happens to look like one (a relative path with exactly one `/`, e.g.
//! `out/qwen3-4b.safetensors`, would otherwise parse as a syntactically
//! valid `<vendor>/<repo>` reference). Only a string that is NOT an
//! existing local file is tried as a hub id, through
//! `qwen3::spec::Qwen3Spec` (the SAME resolver `brain do qwen3
//! chat_generate`/`brain qwen3 chat` use) - fetched under
//! `DownloadPolicy::IfMissing` when nothing local already resolves it,
//! mirroring every other pipeline in this crate.
//!
//! Built on the SAME sequence `qwen3::caps::GenerateAction` (the served
//! `brain do qwen3 chat_generate` / HTTP `/v1/chat/completions` path) runs -
//! `chat::parse_request` (chat-template rendering, sampling-param parsing,
//! stop-strings) + `chat::SeqState` (streaming/finalization, cancellation,
//! `finish_reason`) + `sample::generate_kv_stream_cancellable` - constructed
//! in-process from a plain [`capability::Invocation`] built here, with no
//! capability-dispatch machinery, no scheduler and no paged KV cache in the
//! loop. This is deliberately the SAME code the served path runs, not a
//! second implementation of chat templating/sampling/stop-strings.
//!
//! [`crate::ChatPipeline`] is the multi-turn, tool-calling half of this
//! surface: it runs on the model a [`TextGenerationPipeline`] loaded
//! (`ChatPipeline::from(pipe)`), through the same generation loop.
//!
//! A checkpoint's own on-disk shape decides how much you need to pass:
//!
//! ```no_run
//! // A .gguf ships its own embedded tokenizer - one argument is enough,
//! // whether given as a local path or (as here) a hub id resolved through
//! // Qwen3Spec.
//! let pipe = brain::TextGenerationPipeline::from_pretrained("unsloth/Qwen3-4B-GGUF")?;
//! let out = pipe.generate("Explain DMA in one sentence.")?;
//! println!("{}", out.text);
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! A brain-format `.safetensors` checkpoint has no embedded tokenizer, so it
//! needs one named explicitly (a local path override always wins over
//! whatever a hub id's own resolved `tokenizer` role would supply):
//!
//! ```no_run
//! let pipe = brain::TextGenerationPipeline::builder("/models/qwen3-4b.safetensors")
//!     .tokenizer("/models/qwen3-4b/tokenizer.json")
//!     .load()?;
//! # Ok::<(), brain::Error>(())
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::json;

use crate::{Device, Error, Result};

/// One generated completion: the detokenized text plus the accounting a
/// caller needs to size a request or explain why it stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedText {
    pub text: String,
    /// Tokens the rendered prompt (after chat templating) consumed.
    pub prompt_tokens: u32,
    /// Tokens actually generated.
    pub completion_tokens: u32,
    /// Why generation stopped: `"stop"` (EOS), `"length"` (hit
    /// `max_new_tokens`), `"stop_sequence"`, `"tool_calls"`,
    /// `"tool_choice_unmet"`, or `"cancelled"` (the caller's
    /// [`capability::CancelToken`] fired mid-generation - the text is then
    /// PARTIAL, and this field is the only thing that says so) - the exact
    /// vocabulary `qwen3::chat::SeqState::finish` reports on the served path
    /// too. `"cancelled"` cannot occur through THIS pipeline:
    /// [`TextGenerationPipeline::generate_with`] runs with an unarmed token
    /// because it exposes no way to pass one, so a generation started here
    /// runs to its own stop condition. It is listed because the value is part
    /// of the shared vocabulary, and a caller matching on this field should
    /// handle it rather than assume the set is closed.
    pub finish_reason: String,
}

/// The sampling knobs the text and chat surfaces share, and their ONE
/// mapping onto the invocation `qwen3::chat::parse_request` reads - every
/// field left unset keeps whatever that function already does.
#[derive(Clone, Debug, Default)]
pub(crate) struct Sampling {
    pub(crate) max_new_tokens: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) seed: Option<u64>,
    pub(crate) stop: Vec<String>,
    pub(crate) thinking: Option<bool>,
}

impl Sampling {
    pub(crate) fn apply(&self, mut inv: capability::Invocation) -> Result<capability::Invocation> {
        if let Some(v) = self.max_new_tokens {
            inv = inv.set("max_new", json!(v));
        }
        if let Some(v) = self.temperature {
            inv = inv.set("temp", json!(v));
        }
        if let Some(v) = self.top_k {
            inv = inv.set("top_k", json!(v));
        }
        if let Some(v) = self.top_p {
            inv = inv.set("top_p", json!(v));
        }
        if let Some(v) = self.seed {
            inv = inv.set("seed", json!(v));
        }
        if !self.stop.is_empty() {
            let raw = serde_json::to_string(&self.stop).map_err(|e| Error::Backend(format!("qwen3: encoding stop strings: {e}")))?;
            inv = inv.set("stop", json!(raw));
        }
        if let Some(v) = self.thinking {
            inv = inv.set("enable_thinking", json!(v));
        }
        Ok(inv)
    }
}

/// Generation knobs layered over `qwen3::chat::parse_request`'s own spec
/// defaults - every field left unset here keeps whatever that function
/// already does (greedy-ish 0.8 temperature, 128 new tokens, chat-template
/// rendering on). See that function's own doc for the full default set.
#[derive(Clone, Debug, Default)]
pub struct TextGenerationOptions {
    sampling: Sampling,
    chat: Option<bool>,
}

impl TextGenerationOptions {
    pub fn new() -> TextGenerationOptions {
        TextGenerationOptions::default()
    }

    pub fn max_new_tokens(mut self, n: u32) -> Self {
        self.sampling.max_new_tokens = Some(n);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.sampling.temperature = Some(t);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.sampling.top_k = Some(k);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.sampling.top_p = Some(p);
        self
    }

    /// Reproducible decoding. Left unset, every call gets a real random seed
    /// (never a fixed default) - two back-to-back unseeded calls must not
    /// decode the same sequence, matching `qwen3::chat::sampling_params`'s
    /// own contract.
    pub fn seed(mut self, seed: u64) -> Self {
        self.sampling.seed = Some(seed);
        self
    }

    /// Add a stop string; generation ends the moment the decoded text
    /// contains it. May be called more than once.
    pub fn stop(mut self, s: impl Into<String>) -> Self {
        self.sampling.stop.push(s.into());
        self
    }

    /// Whether to render `prompt` through the chat template (the default) or
    /// send it to the model completion-style, verbatim.
    pub fn chat(mut self, on: bool) -> Self {
        self.chat = Some(on);
        self
    }

    /// Whether a hybrid reasoning model deliberates before answering.
    ///
    /// On by default, which is right for a chat turn and wrong for anything
    /// with a token budget: the model spends it inside `<think>` and never
    /// reaches the answer. A caller extracting data from a model wants the
    /// answer.
    pub fn thinking(mut self, on: bool) -> Self {
        self.sampling.thinking = Some(on);
        self
    }

    fn into_invocation(self, prompt: &str) -> Result<capability::Invocation> {
        let mut inv = self.sampling.apply(capability::Invocation::new().set("prompt", json!(prompt)))?;
        if let Some(v) = self.chat {
            inv = inv.set("chat", json!(v));
        }
        Ok(inv)
    }
}

/// `brain`'s text-generation pipeline. See this module's doc for how it
/// tells a local checkpoint path apart from a hub id.
pub struct TextGenerationPipeline {
    engine: Engine,
}

impl std::fmt::Debug for TextGenerationPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextGenerationPipeline").field("capacity", &self.engine.capacity).field("identity", &self.engine.identity).finish()
    }
}

impl TextGenerationPipeline {
    /// `builder(weights_path).load()`.
    pub fn from_pretrained(weights_path: impl AsRef<str>) -> Result<TextGenerationPipeline> {
        TextGenerationPipeline::builder(weights_path).load()
    }

    pub fn builder(weights_path: impl AsRef<str>) -> TextGenerationPipelineBuilder {
        TextGenerationPipelineBuilder {
            weights: weights_path.as_ref().to_string(),
            adapter: None,
            tokenizer: None,
            device: Device::default(),
            capacity: DEFAULT_CAPACITY,
            download_policy: loader::DownloadPolicy::default(),
            precision: None,
        }
    }

    /// Generate one completion from `prompt`, at every default
    /// [`TextGenerationOptions`] leaves unset.
    pub fn generate(&self, prompt: &str) -> Result<GeneratedText> {
        self.generate_with(prompt, TextGenerationOptions::default())
    }

    /// [`TextGenerationPipeline::generate`] plus [`TextGenerationOptions`].
    /// Runs the SAME `chat::parse_request` + `chat::SeqState` + streaming
    /// KV-cached sampler sequence the served path does - see this module's
    /// doc.
    pub fn generate_with(&self, prompt: &str, opts: TextGenerationOptions) -> Result<GeneratedText> {
        let inv = opts.into_invocation(prompt)?;
        let req = qwen3::chat::parse_request_as(&self.engine.tok, &self.engine.format, &inv).map_err(Error::Backend)?;

        let need = req.ids.len() as u64 + req.max_new as u64;
        if need > self.engine.capacity as u64 {
            return Err(Error::Backend(format!(
                "qwen3: prompt + max_new_tokens ({need} tokens) exceeds this pipeline's built capacity ({} tokens) -- rebuild with .capacity({need}) or larger",
                self.engine.capacity
            )));
        }

        // Nothing here can cancel, so the whole prompt is one prefill chunk:
        // the single prefill call, with no per-chunk readback to pay for.
        let outcome = self.engine.run(&req, &capability::CancelToken::default(), req.ids.len(), &mut |_| {});
        generated_text_from_outcome(outcome)
    }

    /// What this pipeline loaded, by content - see [`crate::chat::ModelIdentity`].
    pub fn identity(&self) -> &crate::chat::ModelIdentity {
        &self.engine.identity
    }

    /// The storage tier the decoder's linears were built at: `"fp32"` or
    /// `"int8"`.
    pub fn precision(&self) -> &'static str {
        self.engine.precision.name()
    }

    /// Apply the LoRA adapter at `path` to the resident base from the next
    /// generation on, replacing any adapter already applied - no reload. See
    /// [`crate::ChatPipeline::attach_adapter`].
    pub fn attach_adapter(&mut self, path: impl AsRef<str>) -> Result<()> {
        self.engine.attach_adapter(path.as_ref())
    }

    /// Remove the applied adapter: generations run on exactly the base
    /// again. `false` when none was applied. See
    /// [`crate::ChatPipeline::detach_adapter`].
    pub fn detach_adapter(&mut self) -> Result<bool> {
        self.engine.detach_adapter()
    }

    /// The loaded engine, for the chat surface built on this same load.
    pub(crate) fn into_engine(self) -> Engine {
        self.engine
    }
}

/// One loaded Qwen3 decoder and everything a generation reads besides the
/// request: the tokenizer, the stop ids, the context it was
/// built for and what it was built from. Both the text and the chat surface
/// run their generations through [`Engine::run`], so the two cannot diverge
/// on templating, sampling, stop strings or cancellation.
pub(crate) struct Engine {
    model: qwen3::Qwen,
    pub(crate) tok: data::qwen_tokenizer::QwenBpe,
    /// The ids that end a generation: the checkpoint's own
    /// (`data::generation::stop_ids`, from its `generation_config.json`,
    /// tokenizer config or GGUF) plus its chat format's end-of-turn token.
    eos: Vec<u32>,
    /// How a request becomes prompt text: the checkpoint's own chat template
    /// (`qwen3::chat::ChatFormat::for_checkpoint`).
    pub(crate) format: qwen3::chat::ChatFormat,
    /// The storage tier the decoder's linears were built at.
    precision: Precision,
    /// The context budget (prompt + completion, in tokens) the KV cache was
    /// BUILT for - fixed at construction, like `s3dit`'s build-time size
    /// (see `ImagePipelineBuilder::size`'s own doc for the same asymmetry).
    pub(crate) capacity: u32,
    pub(crate) identity: crate::chat::ModelIdentity,
    /// The checkpoint the base was loaded from, which a folded adapter's
    /// linears are restored from.
    weights: String,
    /// The linears whose resident weights carry a folded adapter
    /// ([`Engine::fold_adapter`]); empty when none is folded.
    folded: Vec<String>,
}

impl Engine {
    /// Fold the adapter at `path` into the resident base, one targeted linear
    /// at a time: each is read from `src`, corrected by
    /// `qwen3::lora::fold_adapter_into` - the same arithmetic serving uses -
    /// and written over its device weight. Only for an fp32 base, where the
    /// fold is exact; decode then pays nothing for the adapter.
    fn fold_adapter(&mut self, path: &str, src: &dyn checkpoint::TensorSource) -> Result<()> {
        let identity = adapter_identity(path)?;
        let names: Vec<String> = qwen3::lora::read_adapter(path).map_err(|e| Error::Backend(format!("{path}: {e}")))?.sites.into_iter().map(|s| s.base).collect();
        let mut linears = std::collections::HashMap::new();
        for name in &names {
            let mut data = None;
            src.with_tensor(name, &mut |t| data = Some(t.to_vec()));
            linears.insert(name.clone(), data.ok_or_else(|| Error::Backend(format!("{path}: the base has no weight named {name}")))?);
        }
        qwen3::lora::fold_adapter_into(&mut linears, path).map_err(|e| Error::Backend(format!("{path}: {e}")))?;
        for (name, w) in &linears {
            self.model.write_weight(name, w);
        }
        self.folded = names;
        self.identity.adapter = Some(identity);
        Ok(())
    }

    /// Put the base's own weights back under a folded adapter, from the
    /// checkpoint - the linears it touched and nothing else.
    fn unfold(&mut self) -> Result<()> {
        if self.folded.is_empty() {
            return Ok(());
        }
        let (_, src) = qwen3::open_checkpoint(&self.weights).map_err(|e| Error::Backend(format!("qwen3: {}: restoring the base under a folded adapter: {e}", self.weights)))?;
        for name in &self.folded {
            let mut data = None;
            src.with_tensor(name, &mut |t| data = Some(t.to_vec()));
            let data = data.ok_or_else(|| Error::Backend(format!("qwen3: {}: no tensor {name} to restore", self.weights)))?;
            self.model.write_weight(name, &data);
        }
        self.folded.clear();
        self.identity.adapter = None;
        Ok(())
    }

    /// Apply the adapter at `path` beside the resident base
    /// (`qwen3::Qwen::attach_adapter`) and name it in the identity. The file
    /// is read, hashed and validated against the base before anything
    /// changes, so a refused adapter leaves the engine serving what it
    /// served before. A folded adapter is taken back out of the base first.
    pub(crate) fn attach_adapter(&mut self, path: &str) -> Result<()> {
        let identity = adapter_identity(path)?;
        self.model.attach_adapter(path).map_err(Error::Backend)?;
        if let Err(e) = self.unfold() {
            self.model.detach_adapter();
            return Err(e);
        }
        self.identity.adapter = Some(identity);
        Ok(())
    }

    /// Serve exactly the base again; `false` when no adapter was applied.
    pub(crate) fn detach_adapter(&mut self) -> Result<bool> {
        let folded = !self.folded.is_empty();
        self.unfold()?;
        let attached = self.model.detach_adapter();
        self.identity.adapter = None;
        Ok(folded || attached)
    }

    /// One generation: the served path's `SeqState` over the cancellable,
    /// chunk-prefilling KV sampler. `cancel` is polled between prefill
    /// chunks of `prefill_chunk` tokens and after every decoded token;
    /// `progress` sees every visible-text delta and scanner event as it is
    /// produced, including the tail `finish` flushes, so the streamed text
    /// and the outcome's text are the same.
    pub(crate) fn run(&self, req: &qwen3::chat::ParsedRequest, cancel: &capability::CancelToken, prefill_chunk: usize, progress: &mut dyn FnMut(capability::Progress)) -> capability::Outcome {
        let mut rng = data::rng::Rng::new(req.seed);
        let mut seq = qwen3::chat::SeqState::new(req, cancel.clone());
        let mut ids_out: Vec<u32> = Vec::with_capacity(req.max_new);
        let generated = qwen3::sample::generate_kv_stream_on_device(
            &self.model,
            &req.ids,
            req.max_new,
            req.temp,
            req.top_k,
            req.top_p,
            &self.eos,
            &mut rng,
            cancel,
            prefill_chunk,
            &mut |_i, t| {
                ids_out.push(t);
                // `advance` answers "should we stop?"; the sampler asks
                // "keep going?".
                !seq.advance(&self.tok, &ids_out, progress)
            },
        );
        seq.finish(&self.tok, &generated, progress)
    }
}

/// An adapter file's identity: its card id and content digest.
fn adapter_identity(path: &str) -> Result<crate::chat::WeightsIdentity> {
    let card = checkpoint::st::read_card(path).map_err(|e| Error::Backend(format!("{path}: reading the adapter card: {e}")))?;
    crate::chat::WeightsIdentity::of_file(path, card.map(|card| card.id))
}

/// `pub(crate)`: also reused by `crate::vlm`, which shares the exact same
/// `Outcome` shape (`qwen3::chat::SeqState::finish`'s own
/// `text`/`prompt_tokens`/`completion_tokens`/`finish_reason` fields) since
/// `qwen3vl::caps::Resident::generate` runs that SAME shared function - one
/// implementation, not two.
///
/// Both counts are always measured on a generation; one missing is a broken
/// outcome and an error here, never a zero that reads as a measurement.
pub(crate) fn generated_text_from_outcome(o: capability::Outcome) -> Result<GeneratedText> {
    let text = o.outputs.get("text").and_then(|v| v.as_str()).ok_or_else(|| Error::Backend("qwen3: generation outcome carries no text".to_string()))?.to_string();
    let count = |key: &str| -> Result<u32> {
        o.outputs.get(key).and_then(|v| v.as_u64()).and_then(|n| u32::try_from(n).ok()).ok_or_else(|| Error::Backend(format!("qwen3: generation outcome carries no {key}")))
    };
    let prompt_tokens = count("prompt_tokens")?;
    let completion_tokens = count("completion_tokens")?;
    let finish_reason = o.outputs.get("finish_reason").and_then(|v| v.as_str()).unwrap_or("").to_string();
    Ok(GeneratedText { text, prompt_tokens, completion_tokens, finish_reason })
}

/// The context budget a pipeline gets when [`TextGenerationPipelineBuilder::capacity`]
/// is never called - generous enough for a real conversation, small enough
/// that the KV cache this reserves does not surprise an embedder who never
/// thought about it. Override for a longer context or a tighter memory
/// budget.
const DEFAULT_CAPACITY: u32 = 4096;

/// Builds a [`TextGenerationPipeline`]. `.tokenizer(...)`/`.device(...)`/
/// `.capacity(...)` are the only knobs this milestone exposes.
pub struct TextGenerationPipelineBuilder {
    weights: String,
    adapter: Option<String>,
    tokenizer: Option<String>,
    device: Device,
    capacity: u32,
    download_policy: loader::DownloadPolicy,
    precision: Option<Precision>,
}

/// The decoder's storage tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precision {
    Fp32,
    /// Group-wise int8 linears with dynamic activation quantization - lossy.
    Int8,
}

impl Precision {
    fn from_name(name: &str) -> Result<Precision> {
        match name {
            "fp32" => Ok(Precision::Fp32),
            "int8" => Ok(Precision::Int8),
            other => Err(Error::Backend(format!("precision {other:?}: expected \"fp32\" or \"int8\""))),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Int8 => "int8",
        }
    }
    fn dtype(self) -> qwen3::Dtype {
        match self {
            Precision::Fp32 => qwen3::Dtype::F32,
            Precision::Int8 => qwen3::Dtype::I8,
        }
    }
}

impl TextGenerationPipelineBuilder {
    /// A tokenizer to use instead of one embedded in the checkpoint. Required
    /// for a `.safetensors` checkpoint (brain's own format carries no
    /// embedded tokenizer); ignored - never silently overridden - for a
    /// `.gguf` that already embeds one, since the checkpoint's own tokenizer
    /// is what the weights were actually trained/measured against.
    pub fn tokenizer(mut self, path: impl AsRef<str>) -> Self {
        self.tokenizer = Some(path.as_ref().to_string());
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Serve `weights` with a LoRA adapter applied.
    ///
    /// Without this a caller can train an adapter with this engine and has
    /// no way to serve it through this surface, which makes the whole
    /// fine-tune unmeasurable from here. On an fp32 base the adapter's delta
    /// is folded into the targeted linears at load - exact, and decode then
    /// costs what the base's does. On an int8 base a fold would round the
    /// delta back onto the weight grid, so the adapter runs beside the base
    /// instead ([`TextGenerationPipeline::attach_adapter`]). Either way the
    /// pipeline can switch or drop the adapter later without reloading. Rank
    /// and alpha come from the adapter's own `ModelCard` rather than from the
    /// caller, so an adapter cannot be served at a shape it was not trained
    /// at.
    pub fn adapter(mut self, path: impl Into<String>) -> Self {
        self.adapter = Some(path.into());
        self
    }

    /// The context budget (prompt + completion, in tokens) to build the KV
    /// cache for. See [`TextGenerationPipeline`]'s own doc for why this is
    /// fixed at build time rather than resized per call.
    pub fn capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// The decoder's storage tier: `"fp32"` (exact) or `"int8"` (LOSSY:
    /// group-wise int8 linears, a quarter of the memory). Unset, a checkpoint
    /// of 6B parameters or more is loaded int8 - its fp32 weights do not fit
    /// one 24 GB card - and a smaller one fp32. `Err` for any other spelling.
    pub fn precision(mut self, precision: impl AsRef<str>) -> Result<Self> {
        self.precision = Some(Precision::from_name(precision.as_ref())?);
        Ok(self)
    }

    /// How [`TextGenerationPipelineBuilder::load`] may use the network to
    /// resolve a hub-id `weights_path` (unused for a literal local path).
    /// Defaults to [`loader::DownloadPolicy::IfMissing`] -- see that type's
    /// own doc for what each variant means.
    pub fn download_policy(mut self, policy: loader::DownloadPolicy) -> Self {
        self.download_policy = policy;
        self
    }

    /// Load `weights_path` (a local path or a hub id - see this module's
    /// doc for how the two are told apart) and build a real
    /// [`TextGenerationPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::apply`]
    ///    - the same call every other `resolve`-tier pipeline in this crate
    ///      makes, now that a hub id genuinely can reach `crates/loader`'s
    ///      model-store resolution below).
    /// 2. If `weights_path` does not name a real local file,
    ///    [`resolve_hub_weights`] resolves it as a hub id through
    ///    `qwen3::spec::Qwen3Spec`, fetching it first if nothing local
    ///    already resolves it.
    /// 3. The checkpoint is opened (`checkpoint::weightio::WeightReader::open`)
    ///    to read its config AND to fail cleanly here, before any GPU work,
    ///    on a bad path - `qwen3::Qwen::load_inference` itself panics on
    ///    open failure, so this facade never calls it on an unopened path.
    /// 4. The tokenizer resolves: an explicit [`TextGenerationPipelineBuilder::tokenizer`]
    ///    wins; else the checkpoint's own embedded GGUF tokenizer; else, for
    ///    a hub id, the resolved `tokenizer` role from step 2; else a named
    ///    [`Error::MissingArgument`].
    /// 5. `qwen3::footprint::place_and_build` picks a device by real VRAM
    ///    budget (within whatever step 1 already narrowed the ambient
    ///    selection to) and builds the inference-only model.
    pub fn load(self) -> Result<TextGenerationPipeline> {
        let TextGenerationPipelineBuilder { weights, adapter, tokenizer, device, capacity, download_policy, precision } = self;

        crate::device::apply(&device)?;

        let (weights, resolved_tokenizer) = if Path::new(&weights).exists() {
            check_local_weights_architecture(&weights)?;
            (weights, None)
        } else if let Some(local) = resolve_in_store(&weights) {
            // A `vendor/repo` reference the local model store already holds.
            // Taken BEFORE the hub policy so that one string means one model
            // across this SDK: the reader and the study resolve
            // `Qwen/Qwen3-0.6B` through the store, and a text pipeline that
            // instead scanned for candidates would answer a different
            // question about the same argument.
            local
        } else {
            resolve_hub_weights(&weights, download_policy)?
        };

        // Any format the decoder reads: a Hugging Face checkpoint directory
        // (Llama and Qwen2 included), a GGUF or a brain file, read as it is.
        let (cfg, src) = qwen3::open_checkpoint(&weights).map_err(|e| Error::Backend(format!("qwen3: {e}")))?;
        // A single file may carry its own tokenizer (a GGUF) and card.
        let file = Path::new(&weights)
            .is_file()
            .then(|| checkpoint::weightio::WeightReader::open(&weights).map_err(|e| Error::Backend(format!("qwen3: {weights}: {e}"))))
            .transpose()?;
        let embedded = file.as_ref().and_then(|r| r.tokenizer());
        let beside = Path::new(&weights).is_dir().then(|| Path::new(&weights).join("tokenizer.json")).filter(|t| t.is_file()).map(|t| t.to_string_lossy().into_owned());

        let (tok, tok_file) = if let Some(t) = &tokenizer {
            (data::qwen_tokenizer::QwenBpe::from_file(t).map_err(Error::Backend)?, Some(t.clone()))
        } else if let Some(gt) = &embedded {
            (data::qwen_tokenizer::QwenBpe::from_gguf(gt).map_err(Error::Backend)?, None)
        } else if let Some(rt) = resolved_tokenizer.or(beside) {
            (data::qwen_tokenizer::QwenBpe::from_file(&rt).map_err(Error::Backend)?, Some(rt))
        } else {
            return Err(Error::MissingArgument(format!("{weights}: no tokenizer embedded (not a .gguf), none beside it, none resolved, and none given; call .tokenizer(path)")));
        };
        // The tokenizer's directory holds the checkpoint's chat template and
        // generation config.
        let tok_dir = tok_file.as_deref().and_then(|t| Path::new(t).parent()).map(Path::to_path_buf);
        let format = qwen3::chat::ChatFormat::for_checkpoint(tok_dir.as_deref(), &tok);
        let eos = data::generation::stop_ids(tok_dir.as_deref(), &tok, embedded.as_ref().and_then(|g| g.eos), format.end_of_turn()).map_err(|e| Error::Backend(format!("qwen3: {e}")))?;
        let precision = precision.unwrap_or(if qwen3::footprint::int8_by_default(&cfg) { Precision::Int8 } else { Precision::Fp32 });
        let dt = precision.dtype();

        // Built for KV-cache DECODE, which is the only thing a generation
        // ever drives (`Engine::run`: prefill, then one token at a time).
        // A decode build has no batched logits buffer - the LM head is applied
        // per token on the device (`Qwen::decode_logits`, which allocates its
        // one `[vocab]` row on first use) - and its KV cache is the only thing that scales with the
        // context: the batched constructor's `t*vocab` logits buffer alone
        // exceeds the 2 GiB binding limit for Qwen3-0.6B at 4096 tokens.
        let shard = qwen3::Shard::whole(cfg.n_layers as usize);
        let base_id = file.as_ref().and_then(|r| r.card()).map(|card| card.id);
        let model = qwen3::footprint::place_and_build(&cfg, &shard, dt, 1, capacity, false, true, "qwen3", || {
            qwen3::Qwen::new_shard_dt_decode(cfg.clone(), capacity, &*src, shard.clone(), dt)
        })
        .map_err(Error::Backend)?;
        drop(file);

        // What was loaded, by content: hashed after the build, so a file that
        // could not be loaded is never reported, and before the pipeline is
        // handed out, so the digest describes the bytes this load read.
        let identity = crate::chat::ModelIdentity { base: crate::chat::WeightsIdentity::of_path(&weights, base_id)?, adapter: None };
        let mut engine = Engine { model, tok, eos, format, precision, capacity, identity, weights: weights.clone(), folded: Vec::new() };
        match &adapter {
            // An fp32 base takes the adapter's delta exactly, so it is folded
            // in and decode costs what the base's does.
            Some(path) if precision == Precision::Fp32 => engine.fold_adapter(path, &*src)?,
            // A quantized base would round the delta back onto its grid, so
            // the adapter runs beside it instead.
            Some(path) => engine.attach_adapter(path)?,
            None => {}
        }
        Ok(TextGenerationPipeline { engine })
    }
}

/// A LOCAL path's own counterpart to what `Qwen3Spec::classify` already
/// guarantees for free on the hub-id path: refuse a checkpoint by name that
/// positively declares a DIFFERENT architecture, rather than reaching
/// `qwen3::Qwen::load_inference`'s panic-on-mismatched-tensor-names deep
/// inside the model build (the same class of unchecked-panic gap this
/// module's `WeightReader::open`-before-`load_inference` ordering already
/// guards against for a bad PATH -- this guards the same call for a bad
/// ARCHITECTURE at a real, openable path). Only refuses on POSITIVE
/// evidence of a mismatch (`general.architecture`/`ModelCard.family` both
/// present and different from qwen3's own) -- a checkpoint that carries
/// neither marker at all (an unlabeled `.safetensors`, or a `.gguf` with no
/// `general.architecture` KV) is let through unchanged, exactly like
/// `Qwen3Spec::classify_gguf`/`classify_safetensors` themselves only assert
/// a POSITIVE match rather than reject an absence of one.
pub(crate) fn check_local_weights_architecture(weights: &str) -> Result<()> {
    if weights.ends_with(".gguf") {
        if let Ok(g) = checkpoint::gguf::MmapGguf::open(weights) {
            if let Some(arch) = g.kv().get("general.architecture").and_then(|v| v.as_str()) {
                if arch != qwen3::spec::GGUF_ARCHITECTURE {
                    return Err(Error::Backend(format!("{weights}: general.architecture is {arch:?}, not {:?} -- this is not a qwen3 checkpoint", qwen3::spec::GGUF_ARCHITECTURE)));
                }
            }
        }
    } else if let Ok(Some(card)) = checkpoint::st::read_card(weights) {
        if card.family != qwen3::spec::CARD_FAMILY {
            return Err(Error::Backend(format!("{weights}: ModelCard.family is {:?}, not {:?} -- this is not a qwen3 checkpoint", card.family, qwen3::spec::CARD_FAMILY)));
        }
    }
    Ok(())
}

/// `(weights, tokenizer)` for a reference the LOCAL model store already
/// holds, or `None` for anything else.
///
/// Taken between the local-file check and `resolve_hub_weights` so that one
/// model reference means one model across this SDK: without it a
/// `vendor/repo` string resolves through the store on the reader and study
/// surfaces and through a candidate scan here, and the same argument names
/// two different checkpoints.
fn resolve_in_store(reference: &str) -> Option<(String, Option<String>)> {
    let root = loader::model_dir::resolve(None)?;
    let r = brain_modelref::ModelRef::parse(reference).ok()?;
    let local = brain_modelstore::Store::new(&root).local(&r)?;
    // A compound entry's `weights` is its manifest, not a checkpoint: which
    // file plays which role is the spec-aware resolver's question, so it
    // goes there (`resolve_hub_weights`) rather than being opened as one.
    if local.format == brain_modelstore::Format::Compound {
        return None;
    }
    let tokenizer = local.dir.join("tokenizer.json");
    Some((
        local.weights.to_string_lossy().into_owned(),
        tokenizer.exists().then(|| tokenizer.to_string_lossy().into_owned()),
    ))
}

/// Resolve `model_id` as a `<vendor>/<repo>` hub reference through
/// `qwen3::spec::Qwen3Spec`, returning the resolved `weights` path and,
/// when the assembly also carries a `tokenizer` role, that path too (a
/// caller's own [`TextGenerationPipelineBuilder::tokenizer`] still wins over
/// this - see [`TextGenerationPipelineBuilder::load`]'s own doc).
///
/// Tries resolution FIRST, before ever consulting `Store::local`/`plan` -
/// deliberately the reverse of the naive "check `Store::local`, then
/// fetch-if-missing, then resolve" order, for the same real reason
/// `crate::tts::TtsPipelineBuilder::load` does: `Qwen3Spec::classify` reads
/// raw file content (a GGUF's own `general.architecture` KV), a strictly
/// wider net than `Store::local`'s narrow "a compound `brain.manifest.json`,
/// or a bare `model.brain.safetensors`" shapes - and a real GGUF release
/// (the common case for qwen3, unlike a converted `.safetensors` checkpoint)
/// has neither, so `Store::local` would otherwise never recognize even an
/// already-downloaded release and `plan()` would fall through to
/// `TransformersRecipe`'s catch-all, which does not know how to read a bare
/// GGUF's `config.json` (it doesn't have one) and fails outright.
/// `model_id` named neither an existing local file (the caller's own check,
/// before this is reached) nor resolves as a hub id: [`Error::ModelNotFound`]
/// from [`crate::resolve_policy::resolve_with_policy`]'s own `ModelRef::parse`
/// already names `model_id` and the parse failure, so this does not add a
/// second, redundant "not a valid reference" wrapper around it.
pub(crate) fn resolve_hub_weights(model_id: &str, download_policy: loader::DownloadPolicy) -> Result<(String, Option<String>)> {
    let overrides: BTreeMap<String, String> = BTreeMap::new();
    let assembly = crate::resolve_policy::resolve_with_policy("qwen3", &qwen3::spec::Qwen3Spec, model_id, &overrides, download_policy)?;

    let weights = assembly.roles.get("weights").ok_or_else(|| Error::Backend(format!("qwen3: resolved assembly {:?} has no weights role", assembly.id)))?;
    if !weights.exists() {
        return Err(Error::Backend(format!("qwen3: resolved assembly {:?} is missing weights at {}", assembly.id, weights.display())));
    }
    let tokenizer = assembly.roles.get("tokenizer").map(|p| p.to_string_lossy().into_owned());
    Ok((weights.to_string_lossy().into_owned(), tokenizer))
}
