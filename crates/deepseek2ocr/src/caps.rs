// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability::Provider` for DeepSeek-OCR: a document image + an instruction
//! in, decoded text out, streamed token by token.
//!
//! One action, `generate`, in the SAME chat-capable shape `crates/qwenvl/src/
//! caps.rs` and `crates/omni/src/caps.rs` use (`messages`/`prompt`,
//! `.streaming()`, a `Media::Text` output). That is required, not conventional:
//! `apiserve::catalog::api_caps` classifies a model chat-capable only on that
//! exact quadruple, and both HTTP handlers always populate `messages`, never a
//! bare `prompt`.
//!
//! ## What is real here
//!
//! * **Real preprocessing.** [`crate::preprocess::preprocess_image`] turns the
//!   request's decoded pixels into the `[3, 1024, 1024]` normalized tensor with
//!   the checkpoint's own `mean = std = 0.5` affine and the reference's
//!   aspect-preserving fit-and-pad - not a stretch, not a borrowed CLIP
//!   normalization.
//! * **The real 273-row prompt.** [`crate::prompt::build_prompt`] assembles
//!   `BOS ++ <image>×273 ++ text` with the LM GGUF's own tokenizer, and the
//!   composite is built through [`crate::DeepseekOcr::new_with_prompt`], so the
//!   16 `image_newline` rows and the one `view_separator` row carry the mmproj's
//!   learned vectors rather than being 17 missing rows.
//! * **Real per-token streaming**, off
//!   [`crate::DeepseekOcr::generate_greedy_kv_from_prompt_stream`], diffed with
//!   `qwen3::chat::stream_delta` so a multi-byte character never escapes
//!   half-decoded.
//! * **Real EOS early stop.** The decode loop returns `false` from its
//!   callback the moment the model emits end-of-sentence, which
//!   `DeepseekV2::generate_greedy_kv_stream` honors by not dispatching any
//!   further `step` calls - wall time now tracks how early the model actually
//!   stopped, not always `max_new`.
//! * **Real n-gram anti-repetition.** `model::serve::apply_no_repeat_ngram` -
//!   a verbatim port of upstream's own `vllm-project/recipes` entry for this
//!   model (`ngram_size = 30`, `window_size = 90`, `<td>`/`</td>` whitelisted)
//!   - runs on every step's logits before argmax, through
//!   `DeepseekOcr::generate_greedy_kv_from_prompt_stream_filtered`.
//! * **Real token accounting.** `prompt_tokens` / `completion_tokens` /
//!   `finish_reason` are set explicitly, because `apiserve::bridge::read_outcome`
//!   defaults them to `0`/`0`/`"stop"` when absent - i.e. an action that omits
//!   them reports zero usage over the OpenAI and Anthropic surfaces. (Both
//!   `qwen3omnimoe::caps` and `qwen3vl::caps` currently do omit them. That is a bug to
//!   fix there, not a precedent to copy.)
//!
//! ## What is not
//!
//! * **Greedy only, batch 1, one contiguous image run.** No sampling, no
//!   Base/Gundam multi-tile layout (the decoder splice takes one run). The
//!   decode IS now KV-cached (`DeepseekV2::generate_greedy_kv`) - the prompt
//!   pays one batched forward, every generated token after that is one
//!   incremental step, not a full re-run of the whole sequence so far.
//! * **Split backend.** [`Session::load`] builds the vision encoder
//!   (SAM+CLIP+glue) on `gpu_core::Gpu::new_wgpu` and the decoder on
//!   `gpu_core::Gpu::new_cpu`, regardless of the ambient device selection.
//!   `crates/sam1`'s tower used to corrupt its per-block buffers on wgpu at
//!   1024x1024 with three or more blocks; that is fixed and confirmed at
//!   real-weight scale (see `crates/sam1/tests/wgpu_real_weight_parity.rs`),
//!   which is what let the vision half move off the CPU backend.

use std::sync::Mutex;

use capability::{
    last_user_text, Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
    Provider,
};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use serde_json::json;

use crate::config::DeepseekOcrConfig;
use crate::import::{self, Files};
use crate::model::DeepseekOcr;
use crate::preprocess::{self, Fit};
use crate::prompt::{self, Prompt};

/// The catalog id. Case-exact, because it is a real upstream repo
/// (`deepseek-ai/DeepSeek-OCR`) rather than one of the `brain/<family>`
/// placeholders `crates/modelref/src/alias.rs` documents - those exist for
/// models whose weights come from an arbitrary env-named checkpoint, which this
/// one's do not: it is exactly the `ggml-org/DeepSeek-OCR-GGUF` pair or nothing.
pub const MODEL: &str = "deepseek-ai/DeepSeek-OCR";

/// The instruction the reference model ships with, and the one
/// `tests/prompt_real.rs` pinned against the real tokenizer. Used when a request
/// carries neither `messages` nor `prompt`.
///
/// The leading [`crate::prompt::GROUNDING`] marker is **semantically
/// load-bearing, not decoration**: it is what switches the model into
/// grounding mode, where the decoded text carries `<|ref|>`/`<|det|>` spans
/// pairing each piece of recognised text with its bounding box on the page.
/// Dropping it does not merely tidy the string up - it turns that output off,
/// which is why this default keeps it despite being a reserved token a person
/// should never have to type. A request that wants plain markdown with no box
/// spans passes its own `prompt` without the marker; see the `prompt` param's
/// own help.
pub const DEFAULT_INSTRUCTION: &str = "<|grounding|>Convert the document to markdown.";

/// Built context length (the KV-cache capacity, `deepseek2::model::Sizes::ctx`
/// - NOT a batched tape width any more, see [`CHUNK_LEN`]).
///
/// **Used to be a fixed 512-token budget** because every extra row cost a
/// `[seq, 129280]` logit slab in a flat batched tape at build time - the
/// 273-row image block plus BOS plus a real instruction is ~283 rows, so
/// `max_new` could not exceed ~229 whatever a caller asked for. Chunked
/// prefill (`deepseek2::DeepseekV2::prefill_chunked`, this composite built
/// with `batched = false` via [`DeepseekOcr::new_with_prompt_devices_sized`])
/// removed that flat tape entirely, so the real limit is now the checkpoint's
/// own architectural ceiling.
///
/// `$BRAIN_DEEPSEEK_OCR_CTX`, mirroring `qwen3vl::caps`'s own
/// `$BRAIN_QWEN3VL_CTX` - an env-level operator knob, not a per-request
/// parameter, because it sizes a real device allocation (the KV cache, plus
/// `chunk`'s prefill scratch) the resident is built with once. Clamped to
/// the real checkpoint's declared `max_position_embeddings` (8192,
/// `DeepseekV2Config::deepseek_ocr`'s shape) - this decode path's KV cache is
/// plain linear fp32, so an operator asking for MORE than the checkpoint was
/// trained on would not be a longer context, just a larger allocation the
/// model was never taught to use.
fn default_ctx_len() -> u32 {
    const CHECKPOINT_MAX: u32 = 8192;
    std::env::var("BRAIN_DEEPSEEK_OCR_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(CHECKPOINT_MAX).clamp(1, CHECKPOINT_MAX)
}

/// Prefill round width (`deepseek2::model::Sizes::chunk`) - see
/// `DeepseekV2::decode_rows`'s own doc for what this trades off: a wider
/// round amortises each MoE layer's fixed 64-expert dispatch cost over more
/// rows (this model's 283-row prompt is 2-3 rounds at 128-256, not 283
/// individual chunks), at the cost of a wider `[chunk, n_heads, ctx]`
/// attention-score slab. 512 comfortably covers the whole real prompt (the
/// 273-row image block + BOS + instruction, ~283 rows) in ONE round, so a
/// real request's prefill is exactly one dispatch pass, same as the old flat
/// batched tape was - only the KV cache, not the prefill itself, is what
/// changed shape.
///
/// `$BRAIN_DEEPSEEK_OCR_CHUNK`, same operator-knob rationale as
/// [`default_ctx_len`].
fn default_chunk_len() -> u32 {
    std::env::var("BRAIN_DEEPSEEK_OCR_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(512u32).max(1)
}

/// Default generated-token budget.
///
/// **This was 128**, calibrated to the OLD fixed 512-token context (the
/// 273-row image block plus BOS plus the instruction left room for at most
/// ~229 generated tokens, so 128 was picked from two real calibration
/// requests to fit comfortably under that ceiling while still producing real
/// multi-sentence markdown - 414 characters measured, 40.1-69.7 s served
/// median for prefill plus up to 128 KV-cached steps).
///
/// Raised now that [`default_ctx_len`] is no longer capped near 512: real
/// documents (tables, code listings, multi-column pages) routinely need
/// 500-2000+ output tokens, which the old ceiling could never reach
/// regardless of what a caller asked for. 2048 is a provisional operator
/// default, not a throughput claim - decode speed itself (currently ~0.5
/// s/token measured, unrelated to this change) is a separate, tracked
/// follow-up; EOS early stop (`DeepseekV2::generate_greedy_kv_stream`) means
/// a real document that finishes sooner does not pay for the unused budget.
/// A caller that wants a different ceiling passes its own `max_new`.
pub const DEFAULT_MAX_NEW: i64 = 2048;

/// `model::serve::apply_no_repeat_ngram`'s `ngram_size` - upstream's own
/// `deepseek-ai/DeepSeek-OCR` vLLM recipe value (`vllm-project/recipes`'
/// `NGramPerReqLogitsProcessor` config), which this crate ports verbatim.
/// `$BRAIN_DEEPSEEK_OCR_NGRAM_SIZE`, `0` disables the filter entirely
/// (`apply_no_repeat_ngram` is defined as a no-op at `ngram_size == 0`).
fn default_ngram_size() -> usize {
    std::env::var("BRAIN_DEEPSEEK_OCR_NGRAM_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(30)
}

/// `model::serve::apply_no_repeat_ngram`'s `window_size` - same upstream
/// recipe value as [`default_ngram_size`]. `$BRAIN_DEEPSEEK_OCR_WINDOW_SIZE`.
fn default_window_size() -> usize {
    std::env::var("BRAIN_DEEPSEEK_OCR_WINDOW_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(90)
}

pub fn generate_spec() -> ActionSpec {
    ActionSpec::new(
        "generate",
        "DeepSeek-OCR: a document image + an instruction in, decoded text out (greedy, streamed per token)",
    )
    .streaming()
    .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
    .param(
        ParamSpec::new(
            "prompt",
            ParamType::Str,
            "what to do with the page, in plain English (alternative to messages). The default's leading <|grounding|> \
             marker is not decoration: it turns on grounding mode, where the output pairs each piece of recognised text \
             with its bounding box. Delete it for plain markdown with no boxes.",
        )
        .default(json!(DEFAULT_INSTRUCTION)),
    )
    .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(DEFAULT_MAX_NEW)))
    .param(
        ParamSpec::new(
            "weights",
            ParamType::Str,
            "checkpoint DIRECTORY holding both DeepSeek-OCR GGUFs (mmproj + LM); overrides the model-store resolver's own pick when set",
        )
        .host_resolved(),
    )
    .input(BlobSpec::new("image", Media::Image, "raw HWC f32 pixels in [0,1], meta {w,h} (capability::blob's wire convention)").required())
    .output(BlobSpec::new("text", Media::Text, "the decoded document text"))
}

pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "DeepSeek-OCR -- document image in, text/markdown out. DeepEncoder (SAM ViT-B + 16x compressor \
         -> CLIP-L) spliced into a DeepSeek-V2 MoE decoder. Greedy, batch 1, vision on wgpu, decoder on CPU.",
        vec![generate_spec()],
    )
    .with_max_context_tokens(default_ctx_len() as u64)
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP):
/// the checkpoint directory is service-side configuration (resolved through
/// the model store, see `crate::spec::Deepseek2ocrSpec`), so
/// the served action carries only real per-request parameters - see
/// `glmdsa::caps::manifest_resident`'s doc for why a static, CLI-facing
/// manifest and a stripped resident one are two different things, not one
/// hidden behind deployment state. `crate::resident_deepseekocr::
/// DeepseekOcrResident::manifest` calls this rather than [`manifest`].
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// A built composite plus everything one request needs around it.
///
/// Public so `crates/cli/src/resident_deepseekocr.rs` can own one directly:
/// the residency adapter and the direct `brain do` provider then run the SAME
/// code, and cannot drift about preprocessing, prompt assembly or token counts.
pub struct Session {
    dir: String,
    cfg: DeepseekOcrConfig,
    model: DeepseekOcr,
    tok: QwenBpe,
    /// The end-of-sentence id the generated ids are truncated at.
    eos: u32,
    /// The device the preprocessor dispatches on - its own handle, because its
    /// kernel list ([`preprocess::PIPELINES`]) is not any model stage's.
    pre: gpu_core::Gpu,
    /// The context `self.model` was actually built at (`default_ctx_len()` at
    /// load time) - read back here rather than recomputed per request, so a
    /// changed `$BRAIN_DEEPSEEK_OCR_CTX` mid-process can never disagree with
    /// what the resident composite was actually sized for.
    ctx: u32,
    /// `model::serve::apply_no_repeat_ngram`'s `(ngram_size, window_size)`,
    /// resolved once at load time (see [`default_ngram_size`]/
    /// [`default_window_size`]) for the same reason `ctx` is.
    ngram: (usize, usize),
    /// Token ids exempted from the n-gram ban - `<td>`/`</td>`
    /// (`prompt::TD_OPEN`/`TD_CLOSE`) when this tokenizer carries them as
    /// reserved tokens (the real checkpoint's does; a toy test tokenizer may
    /// not, and an absent tag is simply not whitelisted rather than an
    /// error - the filter still works, just without that exemption).
    ngram_whitelist: std::collections::HashSet<u32>,
}

impl Session {
    /// Build the whole composite from a checkpoint directory. Minutes, and a
    /// ~22 GiB peak - this is the call `ResidentModel::activate` makes once.
    ///
    /// **The decoder is forced onto the CPU backend**, with `Gpu::new_cpu`, not
    /// by mutating `BRAIN_DEVICE`: this object lives for the life of a server
    /// process, and a process-global env write from inside one model's
    /// activation would silently change the backend every *other* resident
    /// builds on afterwards. It has no wgpu-corruption reason to move (that bug
    /// was `crates/sam1`'s tower, not the decoder) and no measured wgpu benefit
    /// either, so it stays put.
    ///
    /// **The vision encoder (SAM+CLIP+glue) now builds on `Gpu::new_wgpu`.**
    /// `crates/sam1`'s known wgpu corruption at 1024x1024 with three or more
    /// blocks (what pinned this whole model to the CPU backend originally) is
    /// fixed and confirmed at real-weight scale (`crates/sam1/tests/
    /// wgpu_real_weight_parity.rs`, `wgpu_block_count_corruption.rs`) - a prior
    /// pass measured a several-fold CPU-vs-wgpu gap on this tower, so
    /// moving it is a real per-page win, not a defensive no-op. The vision
    /// tower and the decoder are already separate `Gpu` handles - the splice
    /// crosses them as a host `Vec<f32>` (`DeepseekOcr::encode_block`), never a
    /// raw device buffer - so giving them different backends is a
    /// device-selection change, not an architectural one.
    pub fn load(dir: &str) -> Result<Session, String> {
        let t0 = std::time::Instant::now();
        let files = Files::locate(dir)?;
        let cfg = import::config(&files, 1)?;
        let tok = import::tokenizer(&files)?;
        let eos = tok.special_id(prompt::EOS).ok_or_else(|| format!("this tokenizer has no reserved {:?} token", prompt::EOS))?;
        let ngram_whitelist: std::collections::HashSet<u32> = [prompt::TD_OPEN, prompt::TD_CLOSE].into_iter().filter_map(|s| tok.special_id(s)).collect();

        // The prompt the splice is SIZED for: text_before is always empty, so
        // `row0` is 1 and `n_rows` is 273 whatever instruction a request
        // carries. Requests only vary the text AFTER the image block, which
        // does not move the run -- asserted per request by
        // `generate_greedy_from_prompt_cb` itself.
        let shape = Self::build_prompt(&tok, &cfg, DEFAULT_INSTRUCTION)?;
        stage_time("load: config+tokenizer+prompt", t0);

        let t1 = std::time::Instant::now();
        let dev_vision = |k: &'static [(&'static str, &'static str)]| gpu_core::Gpu::new_wgpu(k);
        let dev_decoder = |k: &'static [(&'static str, &'static str)]| gpu_core::Gpu::new_cpu(k);
        let vision = import::encoder_weights_for(&files, &cfg)?;
        stage_time("load: mmproj import (encoder weights)", t1);
        let t2 = std::time::Instant::now();
        let reader = import::decoder_reader(&files)?;
        let decoder = import::decoder_source(&files, &reader, &cfg)?;
        stage_time("load: decoder_reader open", t2);
        let t3 = std::time::Instant::now();
        let ctx = default_ctx_len();
        let model = DeepseekOcr::new_with_prompt_devices_sized(&dev_vision, &dev_decoder, cfg.clone(), &vision, &decoder, 0, ctx, default_chunk_len(), &shape);
        drop(decoder);
        drop(reader);
        drop(vision);
        stage_time("load: DeepseekOcr::new_with_prompt_devices_sized (weight upload + tape build)", t3);

        let pre = gpu_core::Gpu::new_cpu(preprocess::PIPELINES);
        stage_time("load: TOTAL", t0);
        let ngram = (default_ngram_size(), default_window_size());
        Ok(Session { dir: dir.to_string(), cfg, model, tok, eos, pre, ctx, ngram, ngram_whitelist })
    }

    /// Which checkpoint directory this session was built from.
    pub fn dir(&self) -> &str {
        &self.dir
    }

    /// `BOS ++ <image>×273 ++ "\n" ++ instruction`.
    ///
    /// The newline belongs to the reference's own prompt string
    /// (`"<image>\n<|grounding|>Convert the document to markdown."`), i.e. it
    /// sits between the image block and the instruction - which is exactly the
    /// `text_after` side of `build_prompt`'s split.
    fn build_prompt(tok: &QwenBpe, cfg: &DeepseekOcrConfig, instruction: &str) -> Result<Prompt, String> {
        prompt::build_prompt(tok, "", &format!("\n{instruction}"), cfg.token_grid().0)
    }

    /// Run one `generate` invocation.
    pub fn generate(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let instruction = {
            let t = last_user_text(inv);
            if t.trim().is_empty() {
                DEFAULT_INSTRUCTION.to_string()
            } else {
                t
            }
        };
        let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).clamp(1, self.ctx as i64) as u32;
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;

        let prompt = Self::build_prompt(&self.tok, &self.cfg, &instruction)?;
        if prompt.image_run() != self.model.image_run() {
            // Only reachable if an instruction somehow moved the image block,
            // which `text_before = ""` makes impossible -- but the splice would
            // otherwise land on text rows and still decode, so it is checked.
            return Err(format!(
                "deepseek-ocr generate: this instruction moves the image run to {:?}, but the splice is sized at {:?}",
                prompt.image_run(),
                self.model.image_run()
            ));
        }
        if prompt.len() + max_new as usize > self.ctx as usize {
            return Err(format!(
                "deepseek-ocr generate: prompt ({} tokens, incl. the {}-row image block) + max_new ({max_new}) exceeds this model's context {}",
                prompt.len(),
                prompt.n_rows,
                self.ctx
            ));
        }

        // Real preprocessing: any extent -> [3, 1024, 1024], aspect-preserving
        // fit-and-pad, the checkpoint's own normalization.
        let t_pre = std::time::Instant::now();
        let image = preprocess::preprocess_image(&self.pre, &self.cfg, &hwc, w, h, Fit::Pad);
        stage_time("generate: preprocess", t_pre);

        progress(Progress::step(0, max_new, "generating"));
        let t_gen = std::time::Instant::now();
        // Real per-token deltas: re-decode the running id list each token and
        // emit the UTF-8-safe suffix (`qwen3::chat::stream_delta`), the same
        // loop `qwen3vl::caps` runs.
        let mut ids: Vec<u32> = Vec::new();
        let mut printed = String::new();
        let mut step = 0u32;
        let mut stopped = false;
        let (ngram_size, window_size) = self.ngram;
        let out = self.model.generate_greedy_kv_from_prompt_stream_filtered(
            &image,
            &prompt,
            max_new,
            &mut |history, logits| model::serve::apply_no_repeat_ngram(logits, history, ngram_size, window_size, &self.ngram_whitelist),
            &mut |_, tok_id| {
                step += 1;
                if tok_id == self.eos {
                    stopped = true;
                    return false; // real early stop: no further decode steps are dispatched
                }
                ids.push(tok_id);
                let full = self.tok.decode(&ids);
                let (delta, np) = qwen3::chat::stream_delta(&printed, &full);
                printed = np;
                if !delta.is_empty() {
                    progress(Progress::token(step, max_new, delta));
                }
                true
            },
        );
        // An early stop makes `out` SHORTER than the full budget, on purpose -
        // that is the whole point of `generate_greedy_kv_stream` over
        // `_cb`. `stopped` already distinguishes the two cases for
        // `finish`/`completion` below.
        debug_assert!(out.len() <= prompt.len() + max_new as usize);
        stage_time("generate: encode+splice+decode (TOTAL)", t_gen);
        // A resident device never drops, so its BRAIN_PROFILE table would
        // otherwise never print -- same pattern `crates/fastvlm`'s caps.rs uses.
        self.model.gpu().dump_profile();

        let text = self.tok.decode(&ids);
        // "stop" when the model emitted EOS inside the budget, "length" when it
        // ran the budget out -- `qwen3::chat::SeqState::finish`'s own rule,
        // minus the tool-call/cancellation arms this model has no notion of.
        let finish = if stopped { "stop" } else { "length" };
        // Completion length is what the model actually produced as the
        // completion (EOS included, as OpenAI counts it), not the budget: the
        // ids past EOS are recompute this loop cannot skip, not output.
        let completion = ids.len() + usize::from(stopped);
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(Outcome::new()
            .set("text", json!(text.clone()))
            .set("prompt_tokens", json!(prompt.len() as i64))
            .set("completion_tokens", json!(completion as i64))
            .set("finish_reason", json!(finish))
            .blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

use crate::stage_time;

/// A stateless provider: it holds only the checkpoint directory, and builds (and
/// caches) the composite on the first `generate` - construction must stay cheap,
/// because `crates/cli/src/catalog.rs` constructs every provider just to list it.
pub struct DeepseekOcrProvider {
    dir: String,
}

impl DeepseekOcrProvider {
    /// `None` when `dir` is empty or does not hold both shipped GGUFs -
    /// advertising a model whose every call would fail is worse than not
    /// advertising it. `crates/cli/src/catalog.rs` builds `dir` from a
    /// resolved [`capability::Assembly`] (`crate::spec::Deepseek2ocrSpec`'s
    /// `dir` role) rather than an env var.
    pub fn new(dir: impl Into<String>) -> Option<DeepseekOcrProvider> {
        let dir = dir.into();
        if dir.is_empty() {
            return None;
        }
        match Files::locate(&dir) {
            Ok(_) => Some(DeepseekOcrProvider { dir }),
            Err(e) => {
                eprintln!("brain: deepseek-ocr not served ({e})");
                None
            }
        }
    }
}

impl Provider for DeepseekOcrProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<std::sync::Arc<dyn Action>> {
        (name == "generate").then(|| std::sync::Arc::new(GenerateAction { dir: self.dir.clone() }) as std::sync::Arc<dyn Action>)
    }
}

/// One process-wide session, keyed by checkpoint directory - the same shape
/// `qwen3vl::caps`'s `RESIDENT` static uses, and for the same reason: this is a
/// ~24 GiB build, so a second one would not fit beside the first.
static RESIDENT: Mutex<Option<Session>> = Mutex::new(None);

struct GenerateAction {
    dir: String,
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        generate_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let dir = inv.get_str("weights").filter(|s| !s.is_empty()).unwrap_or_else(|| self.dir.clone());
        if dir.is_empty() {
            return Err("deepseek-ocr generate: no checkpoint directory (pass 'weights', or configure one through the models directory)".to_string());
        }
        let mut guard = RESIDENT.lock().map_err(|_| "deepseek-ocr: resident lock poisoned")?;
        if !matches!(&*guard, Some(s) if s.dir == dir) {
            *guard = None; // free the old composite BEFORE building the new one
            *guard = Some(Session::load(&dir)?);
        }
        guard.as_ref().expect("just built").generate(inv, progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape `apiserve::catalog::api_caps` classifies as chat-capable.
    /// Weights-free, so it runs everywhere.
    #[test]
    fn the_manifest_is_chat_capable_shaped() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.model, "deepseek-ai/DeepSeek-OCR", "the id is the case-exact upstream repo");
        assert_eq!(m.actions.len(), 1);
        let a = &m.actions[0];
        assert_eq!(a.name, "generate");
        assert!(a.streaming, "streaming is required for the chat-capable classification");
        assert!(a.params.iter().any(|p| p.name == "messages"));
        assert!(a.params.iter().any(|p| p.name == "prompt"));
        assert!(a.params.iter().any(|p| p.name == "max_new"));
        assert!(a.inputs.iter().any(|b| b.name == "image" && b.media == Media::Image && b.required));
        assert!(a.outputs.iter().any(|b| b.name == "text" && b.media == Media::Text));
        assert_eq!(m.max_context_tokens, Some(default_ctx_len() as u64));
    }

    /// An unconfigured provider must not exist at all, rather than exist and
    /// fail every call.
    #[test]
    fn an_unset_directory_yields_no_provider() {
        assert!(DeepseekOcrProvider::new("").is_none());
        assert!(DeepseekOcrProvider::new("/definitely/not/a/deepseek/dir").is_none());
    }

    /// The provider only answers its one action.
    #[test]
    fn only_generate_resolves() {
        // A provider built directly (the ctor's existence check is the env
        // path's job, not this one's).
        let p = DeepseekOcrProvider { dir: "/tmp".into() };
        assert!(p.action("generate").is_some());
        assert!(p.action("segment").is_none());
    }

    /// The image run must not depend on the instruction: `text_before` is empty,
    /// so the block always starts right after BOS and is always 273 rows. This
    /// is what lets ONE resident composite serve every request.
    #[test]
    fn the_image_run_is_instruction_independent() {
        // Checkpoint-free: a toy tokenizer carrying this model's reserved
        // strings is all `build_prompt` needs.
        let gt = checkpoint::gguf::GgufTokenizer {
            model: "gpt2".into(),
            pre: Some("deepseek-v3".into()),
            tokens: vec![prompt::BOS.into(), prompt::EOS.into(), prompt::IMAGE.into(), prompt::GROUNDING.into(), "a".into(), "\n".into()],
            merges: Vec::new(),
            token_types: vec![3, 3, 3, 3, 1, 1],
            bos: Some(0),
            eos: Some(1),
            unk: None,
            pad: None,
            ..Default::default()
        };
        let tok = QwenBpe::from_gguf(&gt).expect("toy tokenizer");
        let cfg = DeepseekOcrConfig::deepseek_ocr(1);
        let a = Session::build_prompt(&tok, &cfg, DEFAULT_INSTRUCTION).expect("default instruction");
        let (short, long) = (
            Session::build_prompt(&tok, &cfg, "a").expect("short instruction"),
            Session::build_prompt(&tok, &cfg, "aa").expect("longer instruction"),
        );
        for p in [&a, &short, &long] {
            assert_eq!(p.image_run(), (1, 273), "BOS, then the whole 273-row global view -- whatever the instruction");
        }
        // ...and the instruction really does change the prompt, so the run's
        // invariance above is not vacuous. (Only the tail moves; `row0` cannot,
        // because `text_before` is empty by construction.)
        assert!(long.len() > short.len(), "a longer instruction must produce a longer prompt");
        assert!(a.len() + DEFAULT_MAX_NEW as usize <= default_ctx_len() as usize, "the default request must fit the built context");
    }

    /// `default_ngram_size`/`default_window_size` must match upstream's own
    /// `deepseek-ai/DeepSeek-OCR` vLLM recipe values (30, 90) absent an
    /// operator override, and must actually honour
    /// `$BRAIN_DEEPSEEK_OCR_NGRAM_SIZE`/`$BRAIN_DEEPSEEK_OCR_WINDOW_SIZE` when
    /// set - a typo'd env var name here would silently ignore an operator's
    /// override forever. Not parallel-safe against another test reading the
    /// SAME two vars (none in this file does), same caveat
    /// `modelstore`'s own env-mutating tests record.
    #[test]
    fn ngram_defaults_match_the_upstream_recipe_and_honour_env_overrides() {
        // SAFETY: no other test in this crate reads these two vars, so this
        // cannot race a concurrently-running one within this binary.
        unsafe {
            std::env::remove_var("BRAIN_DEEPSEEK_OCR_NGRAM_SIZE");
            std::env::remove_var("BRAIN_DEEPSEEK_OCR_WINDOW_SIZE");
        }
        assert_eq!(default_ngram_size(), 30, "upstream's own recipe value");
        assert_eq!(default_window_size(), 90, "upstream's own recipe value");

        unsafe {
            std::env::set_var("BRAIN_DEEPSEEK_OCR_NGRAM_SIZE", "12");
            std::env::set_var("BRAIN_DEEPSEEK_OCR_WINDOW_SIZE", "40");
        }
        assert_eq!(default_ngram_size(), 12);
        assert_eq!(default_window_size(), 40);
        unsafe {
            std::env::remove_var("BRAIN_DEEPSEEK_OCR_NGRAM_SIZE");
            std::env::remove_var("BRAIN_DEEPSEEK_OCR_WINDOW_SIZE");
        }
    }
}
