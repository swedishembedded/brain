// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`TextGenerationPipeline`]: brain's text-generation surface, over qwen3
//! (the most complete decoder LM in this workspace) - loaded from a literal
//! local checkpoint path, not a `<vendor>/<repo>` hub id.
//!
//! Unlike [`crate::ImagePipeline`]/[`crate::ForecastPipeline`], this
//! pipeline does NOT go through `crates/loader`'s resolver: `crates/qwen3`
//! declares no model-store `ArchSpec` at all (unlike `kronos`/`timesfm3`,
//! which do), so there is nothing for a resolver to resolve a hub id
//! against - this is a real, tracked gap (writing `crates/qwen3/src/spec.rs`
//! is upstream work in that crate, mirroring `crates/qwen35/src/spec.rs`'s
//! own `["weights", "tokenizer"]`-role shape), not something to fake here.
//!
//! Built on the SAME sequence `qwen3::caps::GenerateAction` (the served
//! `brain do qwen3 chat_generate` / HTTP `/v1/chat/completions` path) runs -
//! `chat::parse_request` (chat-template rendering, sampling-param parsing,
//! stop-strings) + `chat::SeqState` (streaming/finalization, cancellation,
//! `finish_reason`) + `sample::generate_kv_stream` - constructed in-process
//! from a plain [`capability::Invocation`] built here, with no
//! capability-dispatch machinery, no scheduler and no paged KV cache in the
//! loop. This is deliberately the SAME code the served path runs, not a
//! second implementation of chat templating/sampling/stop-strings.
//!
//! A checkpoint's own on-disk shape decides how much you need to pass:
//!
//! ```no_run
//! // A .gguf ships its own embedded tokenizer - one argument is enough.
//! let pipe = brain::TextGenerationPipeline::from_pretrained("/models/qwen3-4b-q8_0.gguf")?;
//! let out = pipe.generate("Explain DMA in one sentence.")?;
//! println!("{}", out.text);
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! A brain-format `.safetensors` checkpoint has no embedded tokenizer, so it
//! needs one named explicitly:
//!
//! ```no_run
//! let pipe = brain::TextGenerationPipeline::builder("/models/qwen3-4b.safetensors")
//!     .tokenizer("/models/qwen3-4b/tokenizer.json")
//!     .load()?;
//! # Ok::<(), brain::Error>(())
//! ```

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
    /// Why generation stopped: `"stop"` (EOS/cancellation), `"length"` (hit
    /// `max_new_tokens`), `"stop_sequence"`, `"tool_calls"`, or
    /// `"tool_choice_unmet"` - the exact vocabulary
    /// `qwen3::chat::SeqState::finish` reports on the served path too.
    pub finish_reason: String,
}

/// Generation knobs layered over `qwen3::chat::parse_request`'s own spec
/// defaults - every field left unset here keeps whatever that function
/// already does (greedy-ish 0.8 temperature, 128 new tokens, chat-template
/// rendering on). See that function's own doc for the full default set.
#[derive(Clone, Debug, Default)]
pub struct TextGenerationOptions {
    max_new_tokens: Option<u32>,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Vec<String>,
    chat: Option<bool>,
}

impl TextGenerationOptions {
    pub fn new() -> TextGenerationOptions {
        TextGenerationOptions::default()
    }

    pub fn max_new_tokens(mut self, n: u32) -> Self {
        self.max_new_tokens = Some(n);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Reproducible decoding. Left unset, every call gets a real random seed
    /// (never a fixed default) - two back-to-back unseeded calls must not
    /// decode the same sequence, matching `qwen3::chat::sampling_params`'s
    /// own contract.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add a stop string; generation ends the moment the decoded text
    /// contains it. May be called more than once.
    pub fn stop(mut self, s: impl Into<String>) -> Self {
        self.stop.push(s.into());
        self
    }

    /// Whether to render `prompt` through the chat template (the default) or
    /// send it to the model completion-style, verbatim.
    pub fn chat(mut self, on: bool) -> Self {
        self.chat = Some(on);
        self
    }

    fn into_invocation(self, prompt: &str) -> Result<capability::Invocation> {
        let mut inv = capability::Invocation::new().set("prompt", json!(prompt));
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
        if let Some(v) = self.chat {
            inv = inv.set("chat", json!(v));
        }
        Ok(inv)
    }
}

/// `brain`'s text-generation pipeline. See this module's doc for why it
/// loads from a literal path rather than a hub id.
pub struct TextGenerationPipeline {
    model: qwen3::Qwen,
    tok: data::qwen_tokenizer::QwenBpe,
    /// The context budget (prompt + completion, in tokens) this pipeline was
    /// BUILT for - fixed at construction, like `s3dit`'s build-time size
    /// (see `ImagePipelineBuilder::size`'s own doc for the same asymmetry).
    /// [`TextGenerationPipeline::generate_with`] validates a request against
    /// it rather than silently truncating or rebuilding.
    capacity: u32,
}

impl std::fmt::Debug for TextGenerationPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextGenerationPipeline").field("capacity", &self.capacity).finish()
    }
}

impl TextGenerationPipeline {
    /// `builder(weights_path).load()`.
    pub fn from_pretrained(weights_path: impl AsRef<str>) -> Result<TextGenerationPipeline> {
        TextGenerationPipeline::builder(weights_path).load()
    }

    pub fn builder(weights_path: impl AsRef<str>) -> TextGenerationPipelineBuilder {
        TextGenerationPipelineBuilder { weights: weights_path.as_ref().to_string(), tokenizer: None, device: Device::default(), capacity: DEFAULT_CAPACITY }
    }

    /// Generate one completion from `prompt`, at every default
    /// [`TextGenerationOptions`] leaves unset.
    pub fn generate(&self, prompt: &str) -> Result<GeneratedText> {
        self.generate_with(prompt, TextGenerationOptions::default())
    }

    /// [`TextGenerationPipeline::generate`] plus [`TextGenerationOptions`].
    /// Runs the SAME `chat::parse_request` + `chat::SeqState` +
    /// `sample::generate_kv_stream` sequence the served path does - see
    /// this module's doc.
    pub fn generate_with(&self, prompt: &str, opts: TextGenerationOptions) -> Result<GeneratedText> {
        let inv = opts.into_invocation(prompt)?;
        let req = qwen3::chat::parse_request(&self.tok, &inv).map_err(Error::Backend)?;

        let need = req.ids.len() as u64 + req.max_new as u64;
        if need > self.capacity as u64 {
            return Err(Error::Backend(format!(
                "qwen3: prompt + max_new_tokens ({need} tokens) exceeds this pipeline's built capacity ({} tokens) -- rebuild with .capacity({need}) or larger",
                self.capacity
            )));
        }

        let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| self.tok.special_id(s)).collect();
        let mut rng = data::rng::Rng::new(req.seed);
        let mut seq = qwen3::chat::SeqState::new(&req, capability::CancelToken::default());
        let mut ids_out: Vec<u32> = Vec::with_capacity(req.max_new);
        let gen = qwen3::sample::generate_kv_stream(&self.model, &req.ids, req.max_new, req.temp, req.top_k, req.top_p, &eos, &mut rng, &mut |_i, t| {
            ids_out.push(t);
            !seq.advance(&self.tok, &ids_out, &mut |_| {})
        });
        let outcome = seq.finish(&self.tok, &gen, &mut |_| {});
        generated_text_from_outcome(outcome)
    }
}

fn generated_text_from_outcome(o: capability::Outcome) -> Result<GeneratedText> {
    let text = o.outputs.get("text").and_then(|v| v.as_str()).ok_or_else(|| Error::Backend("qwen3: generation outcome carries no text".to_string()))?.to_string();
    let prompt_tokens = o.outputs.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u32;
    let completion_tokens = o.outputs.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as u32;
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
    tokenizer: Option<String>,
    device: Device,
    capacity: u32,
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

    /// The context budget (prompt + completion, in tokens) to build the KV
    /// cache for. See [`TextGenerationPipeline`]'s own doc for why this is
    /// fixed at build time rather than resized per call.
    pub fn capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Load `weights_path` and build a real [`TextGenerationPipeline`].
    ///
    /// 1. [`Device`] is applied to this process (see [`crate::device::resolve`],
    ///    the `device`-tier function alone, not [`crate::device::apply`]'s
    ///    model-shard placement layer: this pipeline does no model-store
    ///    resolution to place shards for in the first place).
    /// 2. The checkpoint is opened (`checkpoint::weightio::WeightReader::open`)
    ///    to read its config AND to fail cleanly here, before any GPU work,
    ///    on a bad path - `qwen3::Qwen::load_inference` itself panics on
    ///    open failure, so this facade never calls it on an unopened path.
    /// 3. The tokenizer resolves: an explicit [`TextGenerationPipelineBuilder::tokenizer`]
    ///    wins; else the checkpoint's own embedded GGUF tokenizer; else a
    ///    named [`Error::MissingArgument`].
    /// 4. `qwen3::footprint::place_and_build` picks a device by real VRAM
    ///    budget (within whatever step 1 already narrowed the ambient
    ///    selection to) and builds the inference-only model.
    pub fn load(self) -> Result<TextGenerationPipeline> {
        let TextGenerationPipelineBuilder { weights, tokenizer, device, capacity } = self;

        crate::device::resolve(&device)?;

        let reader = checkpoint::weightio::WeightReader::open(&weights).map_err(|e| Error::Backend(format!("qwen3: {weights}: {e}")))?;
        let cfg = qwen3::QwenConfig::from_json(&reader.config());

        let tok = if let Some(t) = &tokenizer {
            data::qwen_tokenizer::QwenBpe::from_file(t).map_err(Error::Backend)?
        } else if let Some(gt) = reader.tokenizer() {
            data::qwen_tokenizer::QwenBpe::from_gguf(&gt).map_err(Error::Backend)?
        } else {
            return Err(Error::MissingArgument(format!("{weights}: no tokenizer embedded (not a .gguf) and none given; call .tokenizer(path)")));
        };
        drop(reader);

        let shard = qwen3::Shard::whole(cfg.n_layers as usize);
        let model = qwen3::footprint::place_and_build(&cfg, &shard, qwen3::Dtype::F32, 1, capacity, false, false, "qwen3", || qwen3::Qwen::load_inference(&weights, 1, capacity)).map_err(Error::Backend)?;

        Ok(TextGenerationPipeline { model, tok, capacity })
    }
}
