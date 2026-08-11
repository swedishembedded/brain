// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability` surface for Qwen3-Omni: text generation (with optional real
//! audio/image input) and now real speech output.
//!
//! `generate` (text prompt in, plus an optional speech clip and/or image
//! spliced in via `crate::mm`, greedy text out —
//! `crate::generate::generate_greedy`/`generate_greedy_multimodal`) and
//! `speak` (text prompt in, response text + a real spoken waveform out —
//! `OmniInner::speak`, chaining Thinker -> Talker -> MTP -> Code2Wav via
//! `crate::talker_prompt`/`crate::talker_generate`) are both declared.
//! `speak`'s own scope: text-only user turn (no audio/image splice on the
//! Talker side yet — `crate::talker_prompt`'s doc), single-turn (no
//! multi-turn Talker context). No `converse`/`transcribe` actions —
//! `converse` would need `speak`'s loop wired onto `generate`'s multimodal
//! input path together (not done); `transcribe` needs a dedicated ASR-shaped
//! prompt this crate hasn't built (use `brain/qwen-asr` directly for
//! transcription today). Declaring an action whose `run()` can't actually do
//! what its spec promises is worse than not declaring it. Video input IS
//! wired into `generate` now: a `video` blob (`Media::Bytes`, N concatenated
//! HWC-f32 RGB frames + `{frames,w,h,c}` meta —
//! `capability::blob::decode_video_hwc`) decodes to the already-decoded-frame
//! list `crate::mm::encode_video_frames` takes. Frame EXTRACTION (demuxing an
//! actual video file into frames) stays out of scope for this crate, same as
//! before — the caller (`brain-py`) supplies frames it already has.
//!
//! `generate` is itself validation-tier, not production: weights are still
//! streamed fresh from the checkpoint via `checkpoint::weightio::WeightReader`
//! per generated token (no resident weights / int8-sharded serving —
//! `crate::generate`'s own module doc has the full reasoning), though the
//! KV-cache DOES make the attention math itself O(cached length), not
//! O(cached length)² (same module doc). Correct, still slow per token.
//!
//! **`generate_spec()`'s shape matches `crate::resident_mock::MockResident`'s
//! `generate` exactly** (`.streaming()` + `messages`/`prompt` + every param
//! the OpenAI/Anthropic/OpenRouter chat handlers set), not by convention but
//! by requirement: `apiserve::catalog::api_caps` classifies a model as
//! chat-capable only when its `generate` action is `streaming` with a
//! `messages`/`prompt`/`text` param and a `Media::Text` output (M10's
//! investigation found this — D-Bus dispatches by whatever `(model, action)`
//! the caller names, generically, but `/v1/chat/completions`/`/v1/messages`
//! hardcode the action name `"generate"` AND gate exposure on this shape),
//! and both handlers always populate `messages` (a JSON-array string), never
//! a bare `prompt` — so a spec that only declared `prompt` would validate
//! but never actually receive the flattened conversation.

use std::path::Path;
use std::sync::Arc;

use capability::blob::{decode_image, decode_video_hwc};
use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress, Provider};
use checkpoint::weightio::WeightReader;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;
use serde_json::json;

use crate::config::OmniConfig;
use crate::generate::{generate_greedy, generate_greedy_multimodal};
use crate::mm::build_multimodal_prompt;
use crate::talker_generate::{self, GenOpts};
use crate::talker_prompt::build_talker_prompt;
use crate::thinker::thinker_pipelines;

/// Model name in the manifest.
pub const MODEL: &str = "brain/omni";

/// The `generate` action schema — see this module's doc for why it mirrors
/// `MockResident::generate_spec()`'s param list exactly.
pub fn generate_spec() -> ActionSpec {
    ActionSpec::new("generate", "Qwen3-Omni Thinker: greedy text completion (validation-tier -- no KV-cache; see this module's doc)")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages; no chat template applied)"))
        .param(ParamSpec::new("system", ParamType::Str, "system prompt (accepted; folded into the flattened prompt only via messages)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "max tokens to generate").default(json!(32)))
        .param(ParamSpec::new("temp", ParamType::Float, "accepted; greedy (argmax) generation ignores it -- see this module's doc"))
        .param(ParamSpec::new("top_p", ParamType::Float, "accepted; greedy generation ignores it"))
        .param(ParamSpec::new("top_k", ParamType::Int, "accepted; greedy generation ignores it"))
        .param(ParamSpec::new("seed", ParamType::Int, "accepted; greedy generation is deterministic already"))
        .param(ParamSpec::new("stop", ParamType::Str, "accepted; not yet applied (JSON array string)"))
        .param(ParamSpec::new("tools", ParamType::Str, "accepted, not implemented"))
        .param(ParamSpec::new("tool_choice", ParamType::Str, "accepted, ignored"))
        .param(ParamSpec::new("enable_thinking", ParamType::Bool, "accepted, ignored"))
        .input(BlobSpec::new("audio", Media::Audio, "optional speech input: raw mono f32 little-endian PCM at 16 kHz (see audio::asr_caps's wire convention)"))
        .input(BlobSpec::new("image", Media::Image, "optional image input: interleaved HWC f32 in [0,1] (capability::blob's wire convention)"))
        .input(BlobSpec::new("video", Media::Bytes, "optional video input: N concatenated interleaved-HWC f32 RGB frames in [0,1], meta {frames,w,h,c=3} (capability::blob::decode_video_hwc's wire convention)"))
        .output(BlobSpec::new("text", Media::Text, "the generated continuation"))
}

/// The `speak` action schema: text in, spoken text + a real waveform out.
/// Chains Thinker (text) -> Talker + MTP + Code2Wav (`crate::caps::OmniInner
/// ::speak`, `crate::talker_generate`'s module doc) -- text-only user turn,
/// no audio/image splice on this path yet (`crate::talker_prompt`'s scope
/// note).
pub fn speak_spec() -> ActionSpec {
    ActionSpec::new("speak", "Qwen3-Omni: text response + spoken waveform (Thinker -> Talker -> MTP -> Code2Wav)")
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "max TEXT tokens to generate").default(json!(32)))
        .param(ParamSpec::new("speaker", ParamType::Str, "voice name from TalkerConfig::speaker_id (chelsie/ethan/aiden); falls back to the first configured voice").default(json!("chelsie")))
        .output(BlobSpec::new("text", Media::Text, "the generated response text"))
        .output(BlobSpec::new("audio", Media::Audio, "the spoken response: raw mono f32 little-endian PCM at Code2WavConfig::output_sample_rate (24 kHz)"))
}

/// The manifest (`generate` + `speak`).
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Qwen3-Omni-30B-A3B -- text generation with optional real audio/image/video input, plus real speech output via speak (validation-tier: streamed weights, no int8/GPU-sharded residency; the multimodal+speech-output combination is not wired in yet).",
        vec![generate_spec(), speak_spec()],
    )
}

/// The shared messages-array → last-user-turn extraction — hoisted to
/// `capability::last_user_text` (this was one of three hand-synced copies);
/// re-exported so existing `omni::caps::last_user_text` callers keep working.
pub use capability::last_user_text;

/// A loaded Thinker, ready to generate — real weights streamed on demand
/// from `reader`, not resident. `embed_table`/`lm_head` are the two tensors
/// every generated token needs (`thinker.model.embed_tokens.weight`,
/// `thinker.lm_head.weight`, untied — `tie_word_embeddings: false`), kept
/// host-resident once at load time rather than re-read from the mmap on
/// every token (unlike the 48 decoder layers, which really are re-streamed
/// per token — see `crate::generate`'s doc for why).
pub struct OmniInner {
    reader: WeightReader,
    gpu: Gpu,
    cfg: OmniConfig,
    tok: QwenBpe,
    embed_table: Vec<f32>,
    lm_head: Vec<f32>,
    eos_ids: Vec<u32>,
}

pub struct OmniProvider {
    inner: Arc<OmniInner>,
}

impl OmniProvider {
    /// Load from a real HF checkpoint directory (sharded or single-file —
    /// `WeightReader::open_hf_dir` handles both). No brain-native import
    /// step involved: this reads the raw checkpoint directly, the same
    /// pattern every real-weight test in this crate already uses.
    pub fn load(dir: &str) -> Result<OmniProvider, String> {
        let reader = WeightReader::open_hf_dir(Path::new(dir)).map_err(|e| format!("omni: open {dir}: {e}"))?;
        let config_json = std::fs::read_to_string(Path::new(dir).join("config.json")).map_err(|e| format!("omni: read config.json: {e}"))?;
        let root: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| format!("omni: parse config.json: {e}"))?;
        let cfg = OmniConfig::from_json(&root);
        let tok = QwenBpe::from_dir(dir)?;
        let embed_table = reader.tensor("thinker.model.embed_tokens.weight").ok_or("omni: missing thinker.model.embed_tokens.weight")?;
        let lm_head = reader.tensor("thinker.lm_head.weight").ok_or("omni: missing thinker.lm_head.weight")?;
        let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].into_iter().filter_map(|s| tok.special_id(s)).collect();
        let gpu = Gpu::new(thinker_pipelines());
        Ok(OmniProvider { inner: Arc::new(OmniInner { reader, gpu, cfg, tok, embed_table, lm_head, eos_ids }) })
    }

    /// The shared inner state — the seam `cli::resident_omni`'s
    /// `ResidentModel` uses to serve the same loaded model without a second
    /// (and much slower) checkpoint open + tokenizer load.
    pub fn inner(&self) -> Arc<OmniInner> {
        self.inner.clone()
    }
}

impl Provider for OmniProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match resolve_action(name).ok()? {
            OmniActionKind::Generate => Some(Arc::new(GenerateAction { inner: self.inner.clone() })),
            OmniActionKind::Speak => Some(Arc::new(SpeakAction { inner: self.inner.clone() })),
        }
    }
}

/// Which handler an action name dispatches to. Every name [`manifest`]
/// declares MUST resolve here (spec-tested below): an advertised action that
/// silently fell through to a different handler is exactly the served-wrong-
/// result bug `cli::resident_omni` shipped before this dispatcher existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmniActionKind {
    /// Thinker text generation ([`GenerateAction`] path — text out).
    Generate,
    /// Thinker -> Talker -> MTP -> Code2Wav ([`SpeakAction`] path — text + audio out).
    Speak,
}

/// Resolve an action name to its handler, or an error naming the declared
/// set. Unknown actions are a hard error, never a fallthrough.
pub fn resolve_action(name: &str) -> Result<OmniActionKind, String> {
    match name {
        "generate" => Ok(OmniActionKind::Generate),
        "speak" => Ok(OmniActionKind::Speak),
        other => Err(format!("omni: unsupported action '{other}' (this model declares: generate, speak)")),
    }
}

/// Run a named action against a loaded [`OmniInner`] — the single dispatch
/// path shared by [`OmniProvider::action`] callers and the residency adapter
/// (`cli::resident_omni::OmniInstance::run`), so the two serving surfaces
/// cannot disagree about what an action name does.
pub fn run_action(inner: &Arc<OmniInner>, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    match resolve_action(action)? {
        OmniActionKind::Generate => GenerateAction { inner: inner.clone() }.run(inv, progress),
        OmniActionKind::Speak => SpeakAction { inner: inner.clone() }.run(inv, progress),
    }
}

impl OmniInner {
    /// The shared generate path (also used by the resident adapter,
    /// `crate::resident_omni` — matches `qwen_asr::caps::QwenAsrInner::transcribe`'s
    /// shared-between-Provider-and-resident shape). `prompt` is already
    /// resolved (via [`last_user_text`]) by the caller. Text-only: no media
    /// splice, plain-sequential M-RoPE positions (`crate::generate`'s doc).
    pub fn generate(&self, prompt: &str, max_new: u32) -> (String, Vec<u32>) {
        let prompt_ids = self.tok.encode(prompt);
        let out_ids = generate_greedy(&self.reader, &self.gpu, &self.cfg.thinker.text, &self.embed_table, &self.lm_head, &prompt_ids, max_new, &self.eos_ids);
        let new_ids = out_ids[prompt_ids.len()..].to_vec();
        let text = self.tok.decode(&new_ids);
        (text, new_ids)
    }

    /// [`Self::generate`], with real audio and/or image and/or video input
    /// spliced into the prompt (`crate::mm::build_multimodal_prompt`) — real
    /// embeddings (`qwen_asr`'s audio tower / `qwenvl`'s vision tower, both
    /// already parity-validated against this checkpoint) and real per-axis
    /// M-RoPE positions, not the plain-sequential text-only path
    /// [`Self::generate`] takes. `audio` is raw 16kHz mono PCM; `image` is
    /// `(hwc, w, h)` (`capability::blob::decode_image`'s output shape);
    /// `video` is per-frame `(hwc, w, h)` in order
    /// (`capability::blob::decode_video_hwc`'s output shape) — each frame
    /// runs through `crate::mm::encode_video_frames`'s single-frame-path
    /// approximation, per that function's own doc.
    pub fn generate_multimodal(
        &self,
        prompt: &str,
        audio: Option<&[f32]>,
        image: Option<(&[f32], u32, u32)>,
        video: Option<&[(Vec<f32>, u32, u32)]>,
        max_new: u32,
    ) -> Result<(String, Vec<u32>), String> {
        let text_ids = self.tok.encode(prompt);
        let mm_prompt = build_multimodal_prompt(&self.reader, &self.gpu, &self.cfg.thinker, &self.embed_table, &text_ids, audio, image, video)?;
        let n_prompt = mm_prompt.token_ids.len();
        let out_ids = generate_greedy_multimodal(&self.reader, &self.gpu, &self.cfg.thinker.text, &self.embed_table, &self.lm_head, &mm_prompt, max_new, &self.eos_ids);
        let new_ids = out_ids[n_prompt..].to_vec();
        let text = self.tok.decode(&new_ids);
        Ok((text, new_ids))
    }

    /// Speech output: [`Self::generate`] for the text, then chains Talker +
    /// MTP + Code2Wav into a real waveform (`crate::talker_generate`'s
    /// module doc). Text-only user turn (no audio/image splice on this
    /// path yet — see `crate::talker_prompt`'s scope note). `speaker` is a
    /// name from `TalkerConfig::speaker_id` (falls back to the first entry,
    /// typically `"chelsie"`, if unrecognized). Returns `(text, wav_samples,
    /// sample_rate)`.
    pub fn speak(&self, prompt: &str, max_new: u32, speaker: &str) -> Result<(String, Vec<f32>, u32), String> {
        let (text, new_ids) = self.generate(prompt, max_new);
        let user_ids = self.tok.encode(prompt);

        let tc = &self.cfg.talker;
        let text_proj = crate::codec_bridge::load_talker_projection(&self.reader, tc, "text_projection")?;
        let codec_embedding = self.reader.tensor("talker.model.codec_embedding.weight").ok_or("omni: missing tensor talker.model.codec_embedding.weight")?;
        let d = tc.text.hidden as usize;
        let codec_embed = |id: u32| codec_embedding[id as usize * d..(id as usize + 1) * d].to_vec();
        let specials = crate::codec_bridge::talker_prompt_specials(&self.cfg);
        let speaker_id = tc
            .speaker_id
            .get(speaker)
            .copied()
            .or_else(|| tc.speaker_id.values().next().copied())
            .ok_or("omni: TalkerConfig::speaker_id has no entries (checkpoint config.json missing the speaker map?)")?;

        let prompt = build_talker_prompt(&text_proj, &codec_embed, &specials, speaker_id, &self.embed_table, self.cfg.thinker.text.hidden as usize, &user_ids, &new_ids);

        let mtp_gpu = self.gpu.new_like(tts::mtp::PIPELINES);
        let mtp = crate::codec_bridge::load_mtp(&self.reader, mtp_gpu, &tc.code_predictor)?;
        let codec_head_w = self.reader.tensor("talker.codec_head.weight").ok_or("omni: missing tensor talker.codec_head.weight")?;

        // Talker's own kernel-index scheme (crate::talker::talker_pipelines,
        // 18 entries) is NOT the same table as self.gpu's (built from
        // thinker_pipelines, 16 entries) -- dispatching Talker's decode-cache
        // kernels (indices 15-17) against self.gpu's table read out of bounds.
        // A real bug this test's own real-weight run caught (`index 16, len
        // 16`, i.e. the thinker-sized table): a fresh Gpu handle on the same
        // device, with Talker's own pipeline table, is required here.
        let talker_gpu = self.gpu.new_like(crate::talker::talker_pipelines());
        let codes = talker_generate::generate_codes(&self.reader, &talker_gpu, &tc.text, &codec_head_w, tc.codec_eos_token_id, &mtp, codec_embed, &prompt, &GenOpts::default())?;

        let codec = crate::codec_bridge::load_codec(&self.reader, &self.cfg.code2wav)?;
        let wav = codec.decode_omni(&codes);
        Ok((text, wav, self.cfg.code2wav.output_sample_rate))
    }
}

struct GenerateAction {
    inner: Arc<OmniInner>,
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        generate_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let prompt = last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err("omni generate: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(32).clamp(1, 4096) as u32;

        let audio = inv.get_blob("audio").map(audio::asr_caps::wav_from_blob).transpose()?;
        let image = inv.get_blob("image").map(|_| decode_image(inv, "image")).transpose()?;
        let video = inv.get_blob("video").map(|_| decode_video_hwc(inv, "video")).transpose()?;

        progress(Progress::step(0, max_new, "generating"));
        let (text, new_ids) = if audio.is_some() || image.is_some() || video.is_some() {
            let image_ref = image.as_ref().map(|(hwc, w, h)| (hwc.as_slice(), *w, *h));
            self.inner.generate_multimodal(&prompt, audio.as_deref(), image_ref, video.as_deref(), max_new)?
        } else {
            self.inner.generate(&prompt, max_new)
        };
        progress(Progress::step(max_new, max_new, text.clone()));
        Ok(Outcome::new().set("text", json!(text)).set("tokens", json!(new_ids)).blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

struct SpeakAction {
    inner: Arc<OmniInner>,
}

impl Action for SpeakAction {
    fn spec(&self) -> ActionSpec {
        speak_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let prompt = last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err("omni speak: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(32).clamp(1, 4096) as u32;
        let speaker = inv.get_str("speaker").unwrap_or_else(|| "chelsie".to_string());

        progress(Progress::step(0, 2, "generating text"));
        let (text, wav, sample_rate) = self.inner.speak(&prompt, max_new, &speaker)?;
        progress(Progress::step(1, 2, "synthesizing speech"));
        let audio_bytes: Vec<u8> = wav.iter().flat_map(|s| s.to_le_bytes()).collect();
        progress(Progress::step(2, 2, "done"));
        Ok(Outcome::new()
            .set("text", json!(text))
            .blob("text", Blob::new(Media::Text, text.into_bytes()))
            .blob("audio", Blob::new(Media::Audio, audio_bytes).with_meta(json!({"sample_rate": sample_rate}))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: `speak` and `generate` dispatch to DIFFERENT handlers — a
    /// served `speak` must never take the text-only generate path (the P0
    /// this dispatcher fixed: `cli::resident_omni` ignored the action name
    /// and always ran generate, returning text with no audio and no error).
    #[test]
    fn speak_and_generate_resolve_to_distinct_paths() {
        let g = resolve_action("generate").expect("generate must resolve");
        let s = resolve_action("speak").expect("speak must resolve");
        assert_eq!(g, OmniActionKind::Generate);
        assert_eq!(s, OmniActionKind::Speak);
        assert_ne!(g, s, "speak must not alias the generate path");
        // The output contracts really differ: speak declares an audio blob.
        let has_audio = |spec: &ActionSpec| spec.outputs.iter().any(|b| b.media == Media::Audio);
        assert!(has_audio(&speak_spec()), "speak_spec must declare an audio output");
        assert!(!has_audio(&generate_spec()), "generate_spec must not declare an audio output");
    }

    /// Spec: every action the manifest advertises resolves to a handler, and
    /// an unknown action is a hard error (never a silent fallthrough).
    #[test]
    fn every_advertised_action_resolves_and_unknown_errors() {
        for spec in manifest().actions {
            resolve_action(&spec.name).unwrap_or_else(|e| panic!("manifest advertises '{}' but dispatch rejects it: {e}", spec.name));
        }
        let err = resolve_action("transcribe").expect_err("undeclared action must error");
        assert!(err.contains("unsupported action"), "error must name the failure: {err}");
    }
}
