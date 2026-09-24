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
//! `finish_reason`) + `sample::generate_kv_stream` - constructed in-process
//! from a plain [`capability::Invocation`] built here, with no
//! capability-dispatch machinery, no scheduler and no paged KV cache in the
//! loop. This is deliberately the SAME code the served path runs, not a
//! second implementation of chat templating/sampling/stop-strings.
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

/// `brain`'s text-generation pipeline. See this module's doc for how it
/// tells a local checkpoint path apart from a hub id.
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
        TextGenerationPipelineBuilder {
            weights: weights_path.as_ref().to_string(),
            tokenizer: None,
            device: Device::default(),
            capacity: DEFAULT_CAPACITY,
            download_policy: loader::DownloadPolicy::default(),
        }
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

/// `pub(crate)`: also reused by `crate::vlm`, which shares the exact same
/// `Outcome` shape (`qwen3::chat::SeqState::finish`'s own
/// `text`/`prompt_tokens`/`completion_tokens`/`finish_reason` fields) since
/// `qwen3vl::caps::Resident::generate` runs that SAME shared function - one
/// implementation, not two.
pub(crate) fn generated_text_from_outcome(o: capability::Outcome) -> Result<GeneratedText> {
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
    download_policy: loader::DownloadPolicy,
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
        let TextGenerationPipelineBuilder { weights, tokenizer, device, capacity, download_policy } = self;

        crate::device::apply(&device)?;

        let (weights, resolved_tokenizer) = if Path::new(&weights).is_file() {
            check_local_weights_architecture(&weights)?;
            (weights, None)
        } else if let Some(local) = crate::study::resolve_in_store(&weights) {
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

        let reader = checkpoint::weightio::WeightReader::open(&weights).map_err(|e| Error::Backend(format!("qwen3: {weights}: {e}")))?;
        let cfg = qwen3::QwenConfig::from_json(&reader.config());

        let tok = if let Some(t) = &tokenizer {
            data::qwen_tokenizer::QwenBpe::from_file(t).map_err(Error::Backend)?
        } else if let Some(gt) = reader.tokenizer() {
            data::qwen_tokenizer::QwenBpe::from_gguf(&gt).map_err(Error::Backend)?
        } else if let Some(rt) = &resolved_tokenizer {
            data::qwen_tokenizer::QwenBpe::from_file(rt).map_err(Error::Backend)?
        } else {
            return Err(Error::MissingArgument(format!("{weights}: no tokenizer embedded (not a .gguf), none resolved, and none given; call .tokenizer(path)")));
        };
        // Built for KV-cache DECODE, which is the only thing `generate_with`
        // ever drives (`qwen3::sample::generate_kv_stream`: prefill, then one
        // token at a time).
        //
        // The batched constructor sizes per-layer activations at `b*t`,
        // attention scores at `n_heads*ctx^2` and a logits buffer at
        // `t*vocab` - none of which a decode reads. On a large-vocabulary
        // model that logits buffer alone is bigger than the device will bind:
        // Qwen3-0.6B at the default 4096 capacity asks for 2.32 GiB against
        // the 2 GiB `maxStorageBufferBindingSize` that Vulkan and WebGPU both
        // standardise, so the default pipeline could not load it at all, and
        // failed inside a bind group rather than anywhere that named a
        // capacity. A decode build has no logits buffer - the LM head is
        // applied host-side - and its KV cache is the only thing that scales
        // with the context at all.
        let shard = qwen3::Shard::whole(cfg.n_layers as usize);
        let model = qwen3::footprint::place_and_build(&cfg, &shard, qwen3::Dtype::F32, 1, capacity, false, true, "qwen3", || {
            qwen3::Qwen::from_reader_decode(&reader, capacity)
        })
        .map_err(Error::Backend)?;
        drop(reader);

        Ok(TextGenerationPipeline { model, tok, capacity })
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
