// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability` surface for Qwen3-Omni: text generation (with optional real
//! audio/image/video input), real speech output, and now real audio/image/
//! video input FUSED WITH speech output in one call.
//!
//! Three actions: `generate` (text prompt in, plus an optional speech clip
//! and/or image and/or video spliced in via `crate::mm`, greedy text out —
//! `crate::generate::generate_greedy`/`generate_greedy_multimodal`); `speak`
//! (text prompt in, response text + a real spoken waveform out —
//! `OmniInner::speak`, chaining Thinker -> Talker -> MTP -> Code2Wav via
//! `crate::talker_prompt`/`crate::talker_generate` — text-only user turn:
//! the Talker's user segment always takes the text-projection branch); and
//! `converse` (real audio/image/video input AND real speech output, same
//! turn — `OmniInner::converse`, `crate::talker_prompt::UserMediaSplice`'s
//! per-position `hidden_projection`/`text_projection` selection is what
//! makes this correct instead of silently ignoring the multimodal input on
//! the Talker side the way reusing `speak`'s text-only assembly would).
//! `converse` is D-Bus/CLI-reachable only:
//! `apiserve::catalog::api_caps` classifies an action as chat-capable only
//! when it is literally named `generate` (see this doc's own note on
//! `generate_spec()` below), so `converse` never appears on `/v1/chat/
//! completions`/`/v1/messages`. No `transcribe` action — it needs a
//! dedicated ASR-shaped prompt this crate hasn't built (use `brain/qwen-asr`
//! directly for transcription today). Declaring an action whose `run()`
//! can't actually do what its spec promises is worse than not declaring it.
//! All three actions are single-turn (no multi-turn Talker context tracked
//! across calls). Video input: a `video` blob (`Media::Bytes`, N concatenated
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
use crate::generate::{generate_greedy, generate_greedy_multimodal, thinker_hidden_at_layer, EmbedTable, ThinkerStack};
use crate::mm::build_multimodal_prompt;
use crate::talker_generate::{self, GenOpts};
use crate::talker_prompt::{build_talker_prompt, UserMediaSplice};
use crate::thinker::thinker_pipelines;

/// Model name in the manifest.
pub const MODEL: &str = "brain/omni";

/// The chat-shaped `generate` schema every Thinker-backed model shares: the
/// text params, `.streaming()`, and a `Media::Text` output.
///
/// Extracted so `brain/omni-int8-thinker-multi` is reachable through the SAME
/// request contract as `brain/omni` without a second, hand-synced copy of this
/// param list. That shape is not cosmetic: `apiserve::catalog::api_caps` gates
/// `/v1/chat/completions` and `/v1/messages` exposure on exactly
/// `name == "generate"` + `streaming` + a `messages`/`prompt`/`text` param + a
/// `Media::Text` output, so a model missing any of it is invisible to the chat
/// APIs (see this module's doc and `tests/caps_conformance.rs`).
///
/// Media INPUTS are deliberately not here: they are the multimodal path's, and
/// a model that would silently ignore an attached image must not advertise one.
pub fn chat_generate_spec(desc: &str) -> ActionSpec {
    ActionSpec::new("generate", desc)
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
        .output(BlobSpec::new("text", Media::Text, "the generated continuation"))
}

/// The three real media inputs `generate` accepts, wherever a model's
/// `generate` actually splices them in via `crate::mm::build_multimodal_prompt`
/// (today: this model and `brain/omni-int8-thinker-multi`). Factored out so
/// both declarations are built from the SAME `.input(...)` chain rather than
/// two hand-synced copies that could silently drift apart -
/// `tests/caps_conformance.rs` asserts the two models' declared inputs match,
/// but a shared builder is what makes that assertion something other than a
/// convention.
pub fn with_multimodal_inputs(spec: ActionSpec) -> ActionSpec {
    spec.input(BlobSpec::new("audio", Media::Audio, "optional speech input: raw mono f32 little-endian PCM at 16 kHz (see audio::asr_caps's wire convention)"))
        .input(BlobSpec::new("image", Media::Image, "optional image input: interleaved HWC f32 in [0,1] (capability::blob's wire convention)"))
        .input(BlobSpec::new("video", Media::Bytes, "optional video input: N concatenated interleaved-HWC f32 RGB frames in [0,1], meta {frames,w,h,c=3} (capability::blob::decode_video_hwc's wire convention)"))
}

/// The `generate` action schema - see this module's doc for why it mirrors
/// `MockResident::generate_spec()`'s param list exactly. [`chat_generate_spec`]
/// plus this path's real audio/image/video inputs ([`with_multimodal_inputs`]).
pub fn generate_spec() -> ActionSpec {
    with_multimodal_inputs(chat_generate_spec("Qwen3-Omni Thinker: greedy text completion (validation-tier -- no KV-cache; see this module's doc)"))
}

/// The `speak` action schema: text in, spoken text + a real waveform out.
/// Chains Thinker (text) -> Talker + MTP + Code2Wav (`crate::caps::OmniInner
/// ::speak`, `crate::talker_generate`'s module doc) -- text-only user turn,
/// no audio/image splice on this path yet (`crate::talker_prompt`'s scope
/// note). `.streaming()`: audio chunks now arrive mid-run via
/// `Progress::chunk` (`Codec::decode_omni_chunked`, wired in
/// `OmniInner::speak`) for a `Subscribe`-based caller; the terminal
/// `Outcome` still carries the FULL reassembled audio too, unchanged, for a
/// plain `Run` caller that never sees progress frames.
pub fn speak_spec() -> ActionSpec {
    ActionSpec::new("speak", "Qwen3-Omni: text response + spoken waveform (Thinker -> Talker -> MTP -> Code2Wav)")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "max TEXT tokens to generate").default(json!(32)))
        .param(ParamSpec::new("speaker", ParamType::Str, "voice name from TalkerConfig::speaker_id (chelsie/ethan/aiden); falls back to the first configured voice").default(json!("chelsie")))
        .output(BlobSpec::new("text", Media::Text, "the generated response text"))
        .output(BlobSpec::new("audio", Media::Audio, "the spoken response: raw mono f32 little-endian PCM at Code2WavConfig::output_sample_rate (24 kHz)"))
}

/// The `converse` action schema: real audio/image/video input AND real
/// speech output, in one call — see this module's doc for why this needs
/// `crate::talker_prompt::UserMediaSplice` rather than reusing `speak`'s
/// text-only Talker-prompt assembly, and why it is D-Bus/CLI-only (not
/// classified chat-capable by `apiserve::catalog::api_caps`, which keys on
/// the literal action name `generate`).
pub fn converse_spec() -> ActionSpec {
    ActionSpec::new("converse", "Qwen3-Omni: real audio/image/video input, text response + spoken waveform out (Thinker -> Talker -> MTP -> Code2Wav, media-aware user segment). D-Bus/CLI only.")
        .streaming()
        .param(ParamSpec::new("messages", ParamType::Str, "flattened chat messages (JSON array string)"))
        .param(ParamSpec::new("prompt", ParamType::Str, "a raw prompt (alternative to messages)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "max TEXT tokens to generate").default(json!(32)))
        .param(ParamSpec::new("speaker", ParamType::Str, "voice name from TalkerConfig::speaker_id (chelsie/ethan/aiden); falls back to the first configured voice").default(json!("chelsie")))
        .input(BlobSpec::new("audio", Media::Audio, "optional speech input: raw mono f32 little-endian PCM at 16 kHz (see audio::asr_caps's wire convention)"))
        .input(BlobSpec::new("image", Media::Image, "optional image input: interleaved HWC f32 in [0,1] (capability::blob's wire convention)"))
        .input(BlobSpec::new("video", Media::Bytes, "optional video input: N concatenated interleaved-HWC f32 RGB frames in [0,1], meta {frames,w,h,c=3} (capability::blob::decode_video_hwc's wire convention)"))
        .output(BlobSpec::new("text", Media::Text, "the generated response text"))
        .output(BlobSpec::new("audio", Media::Audio, "the spoken response: raw mono f32 little-endian PCM at Code2WavConfig::output_sample_rate (24 kHz)"))
}

/// The manifest (`generate` + `speak` + `converse`).
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Qwen3-Omni-30B-A3B -- text generation with optional real audio/image/video input, real speech output via speak, and real audio/image/video input fused with speech output via converse (validation-tier: streamed weights, no int8/GPU-sharded residency).",
        vec![generate_spec(), speak_spec(), converse_spec()],
    )
}

/// The shared messages-array → last-user-turn extraction — hoisted to
/// `capability::last_user_text` (this was one of three hand-synced copies);
/// re-exported so existing `omni::caps::last_user_text` callers keep working.
pub use capability::last_user_text;

/// A loaded Thinker, ready to generate.
///
/// `gpus` is one handle per device the placement uses and `stack` is the
/// placement itself (`crate::generate::ThinkerStack`): as many decoder layers
/// as genuinely fit are resident across those cards, the rest stream per use
/// in bounded chunks. `embed` is the token embedding, borrowed from the
/// mapping a row at a time where the dtype allows it rather than expanded to a
/// 1.2 GB f32 table; `lm_head` is not held on the host at all any more - it is
/// a device buffer inside `stack`.
pub struct OmniInner {
    reader: WeightReader,
    /// DECLARED BEFORE `gpus`, and that order is load-bearing: struct fields
    /// drop in declaration order, and every buffer in the stack belongs to one
    /// of those devices. Tearing the devices down first leaves the buffers
    /// pointing at destroyed driver state - teardown then faults, turning a
    /// clean SIGTERM into a crash.
    stack: ThinkerStack,
    gpus: Vec<Gpu>,
    cfg: OmniConfig,
    tok: QwenBpe,
    embed: EmbedTable,
    eos_ids: Vec<u32>,
}

pub struct OmniProvider {
    inner: Arc<OmniInner>,
}

/// Bytes kept free per card for activations when the caller has no budget of
/// its own to hand in - matching `brain serve`'s own `--reserve-gb` default,
/// so a standalone `load()` and a scheduled one size the cards the same way.
const DEFAULT_RESERVE_BYTES: u64 = 2 << 30;

impl OmniProvider {
    /// Load from a real HF checkpoint directory (sharded or single-file —
    /// `WeightReader::open_hf_dir` handles both), placing the Thinker across
    /// every GPU this process discovered. No brain-native import step
    /// involved: this reads the raw checkpoint directly.
    pub fn load(dir: &str) -> Result<OmniProvider, String> {
        Self::load_on(dir, &crate::thinker_plan::discovered_devices(DEFAULT_RESERVE_BYTES))
    }

    /// [`Self::load`] against a caller-supplied device budget: `(canonical GPU
    /// index, USABLE bytes)` per card.
    ///
    /// Capacity travels with identity because the split has to RESPECT it -
    /// a 24 GB and an 8 GB card must not get the same number of layers, and a
    /// model that fits one card must not be spread over three. Which of these
    /// devices actually gets used, and how many layers each holds, is decided
    /// by `model::shard`'s capacity-aware planner from the checkpoint's own
    /// per-tensor byte costs (`crate::thinker_plan`).
    pub fn load_on(dir: &str, devices: &[(u32, u64)]) -> Result<OmniProvider, String> {
        let reader = WeightReader::open_hf_dir(Path::new(dir)).map_err(|e| format!("omni: open {dir}: {e}"))?;
        let config_json = std::fs::read_to_string(Path::new(dir).join("config.json")).map_err(|e| format!("omni: read config.json: {e}"))?;
        let root: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| format!("omni: parse config.json: {e}"))?;
        let cfg = OmniConfig::from_json(&root);
        let tok = QwenBpe::from_dir(dir)?;
        let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].into_iter().filter_map(|s| tok.special_id(s)).collect();
        let embed = EmbedTable::open(&reader)?;
        if embed.hidden() != cfg.thinker.text.hidden as usize {
            return Err(format!("omni: the embedding table is [_, {}] but the config says hidden={}", embed.hidden(), cfg.thinker.text.hidden));
        }
        // No device budget at all (no GPU on this box) still has to work:
        // fall back to the ambient single device, which is what every other
        // single-device resident does.
        let (gpus, caps): (Vec<Gpu>, Vec<u64>) = if devices.is_empty() {
            (vec![Gpu::new(thinker_pipelines())], vec![u64::MAX])
        } else {
            let mut gs = Vec::with_capacity(devices.len());
            for &(i, _) in devices {
                gs.push(Gpu::new_on_index(i, thinker_pipelines())?);
            }
            (gs, devices.iter().map(|&(_, c)| c).collect())
        };
        let stack = ThinkerStack::build(&reader, &cfg.thinker.text, &gpus, &caps)?;
        Ok(OmniProvider { inner: Arc::new(OmniInner { reader, stack, gpus, cfg, tok, embed, eos_ids }) })
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
            OmniActionKind::Converse => Some(Arc::new(ConverseAction { inner: self.inner.clone() })),
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
    /// Thinker -> Talker -> MTP -> Code2Wav ([`SpeakAction`] path — text + audio out, text-only user turn).
    Speak,
    /// Thinker -> Talker -> MTP -> Code2Wav ([`ConverseAction`] path — real audio/image/video in AND audio out, same turn).
    Converse,
}

/// Resolve an action name to its handler, or an error naming the declared
/// set. Unknown actions are a hard error, never a fallthrough.
pub fn resolve_action(name: &str) -> Result<OmniActionKind, String> {
    match name {
        "generate" => Ok(OmniActionKind::Generate),
        "speak" => Ok(OmniActionKind::Speak),
        "converse" => Ok(OmniActionKind::Converse),
        other => Err(format!("omni: unsupported action '{other}' (this model declares: generate, speak, converse)")),
    }
}

/// Run a named action against a loaded [`OmniInner`] — the single dispatch
/// path shared by [`OmniProvider::action`] callers and the residency adapter
/// (`cli::resident_omni::OmniInstance::run`), so the two serving surfaces
/// cannot disagree about what an action name does.
///
/// Validates `inv` against the resolved action's own [`ActionSpec`] first
/// (`residency::bridge::ProviderInstance::run`'s exact pattern) — `Action::run`'s
/// own doc says "the invocation is already validated against `Action::spec`",
/// a contract this function used to silently violate: unknown params went
/// unrejected and declared defaults (e.g. `generate_spec`'s `max_new` default
/// of 32) were never filled from the spec, only hand-duplicated per call site.
pub fn run_action(inner: &Arc<OmniInner>, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    match resolve_action(action)? {
        OmniActionKind::Generate => {
            let act = GenerateAction { inner: inner.clone() };
            let inv = act.spec().validate(inv.clone())?;
            act.run(&inv, progress)
        }
        OmniActionKind::Speak => {
            let act = SpeakAction { inner: inner.clone() };
            let inv = act.spec().validate(inv.clone())?;
            act.run(&inv, progress)
        }
        OmniActionKind::Converse => {
            let act = ConverseAction { inner: inner.clone() };
            let inv = act.spec().validate(inv.clone())?;
            act.run(&inv, progress)
        }
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
        let t0 = std::time::Instant::now();
        let out_ids = generate_greedy(&self.stack, &self.gpus, &self.reader, &self.cfg.thinker.text, &self.embed, &prompt_ids, max_new, &self.eos_ids);
        gpu_core::profile::stage_time("omni generate", t0);
        crate::generate::dump_stream_profile(&self.gpus);
        let new_ids = out_ids[prompt_ids.len()..].to_vec();
        let text = self.tok.decode(&new_ids);
        (text, new_ids)
    }

    /// The whole embedding table as f32, for the two callers that genuinely
    /// need it in one piece (the multimodal splice and the Talker prompt).
    /// Borrowed when a host copy already exists, decoded on demand otherwise -
    /// never a second resident copy alongside the first.
    fn embed_host(&self) -> std::borrow::Cow<'_, [f32]> {
        match self.embed.as_host_slice() {
            Some(t) => std::borrow::Cow::Borrowed(t),
            None => std::borrow::Cow::Owned(self.embed.to_host(&self.reader)),
        }
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
        let mm_prompt = build_multimodal_prompt(&self.reader, &self.gpus[0], &self.cfg.thinker, &self.embed_host(), &text_ids, audio, image, video)?;
        let n_prompt = mm_prompt.token_ids.len();
        let out_ids = generate_greedy_multimodal(&self.stack, &self.gpus, &self.reader, &self.cfg.thinker.text, &self.embed, &mm_prompt, max_new, &self.eos_ids);
        let new_ids = out_ids[n_prompt..].to_vec();
        let text = self.tok.decode(&new_ids);
        Ok((text, new_ids))
    }

    /// Speech output: [`Self::generate`] for the text, then chains Talker +
    /// MTP + Code2Wav into a real waveform (`crate::talker_generate`'s
    /// module doc). Text-only user turn (no audio/image splice on this
    /// path yet — see `crate::talker_prompt`'s scope note). `speaker` is a
    /// name from `TalkerConfig::speaker_id` (falls back to the first entry,
    /// typically `"chelsie"`, if unrecognized). Vocodes via
    /// `Codec::decode_omni_chunked` (`SPEAK_CHUNK_FRAMES` at a time),
    /// calling `on_chunk` with each real audio segment as it's decoded — a
    /// `Subscribe`-based caller gets real mid-stream audio
    /// (`ConverseAction`/`SpeakAction::run` turn this into
    /// `Progress::chunk`); `on_chunk` may be a no-op for a caller that only
    /// wants the final reassembled waveform this still returns. Returns
    /// `(text, wav_samples, sample_rate)` — the SAME complete waveform
    /// regardless of whether `on_chunk` streamed it too.
    pub fn speak(&self, prompt: &str, max_new: u32, speaker: &str, mut on_chunk: impl FnMut(&[f32])) -> Result<(String, Vec<f32>, u32), String> {
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

        let prompt = build_talker_prompt(&text_proj, &codec_embed, &specials, speaker_id, &self.embed_host(), self.cfg.thinker.text.hidden as usize, &user_ids, &new_ids, None);

        let mtp_gpu = self.gpus[0].new_like(tts::mtp::PIPELINES);
        let mtp = crate::codec_bridge::load_mtp(&self.reader, mtp_gpu, &tc.code_predictor)?;
        let codec_head_w = self.reader.tensor("talker.codec_head.weight").ok_or("omni: missing tensor talker.codec_head.weight")?;

        // Talker's own kernel-index scheme (crate::talker::talker_pipelines,
        // 18 entries) is NOT the same table as self.gpus[0]'s (built from
        // thinker_pipelines, 16 entries) -- dispatching Talker's decode-cache
        // kernels (indices 15-17) against self.gpus[0]'s table read out of bounds.
        // A real bug this test's own real-weight run caught (`index 16, len
        // 16`, i.e. the thinker-sized table): a fresh Gpu handle on the same
        // device, with Talker's own pipeline table, is required here.
        let talker_gpu = self.gpus[0].new_like(crate::talker::talker_pipelines());
        let codes = talker_generate::generate_codes(&self.reader, &talker_gpu, &tc.text, &codec_head_w, tc.codec_eos_token_id, &mtp, codec_embed, &prompt, &GenOpts::default())?;

        let codec = crate::codec_bridge::load_codec(&self.reader, &self.cfg.code2wav)?;
        let mut wav = Vec::new();
        codec.decode_omni_chunked(&codes, SPEAK_CHUNK_FRAMES, |chunk| {
            wav.extend_from_slice(chunk);
            on_chunk(chunk);
        });
        Ok((text, wav, self.cfg.code2wav.output_sample_rate))
    }

    /// [`Self::speak`], but the user turn carries real audio/image/video
    /// input AND the Talker's own user segment reflects it — the gap
    /// `speak` alone leaves open (its user segment is always
    /// `text_projection(thinker_embed(id))`, `crate::talker_prompt`'s
    /// original scope note). Builds the SAME `MultimodalPrompt`
    /// (`crate::mm::build_multimodal_prompt`) [`Self::generate_multimodal`]
    /// uses, reuses it for both the text generation AND a second Thinker
    /// pass (`crate::generate::thinker_hidden_at_layer`, teacher-forced,
    /// early-exits after `TalkerConfig::accept_hidden_layer`) that captures
    /// the accept-layer hidden state for every USER-segment position, then
    /// feeds `crate::talker_prompt::UserMediaSplice` — a per-position mask
    /// derived directly from `mm_prompt.token_ids` (a position is a media
    /// position iff its token id is `audio_token_id`/`image_token_id`/
    /// `video_token_id`, no separate bookkeeping needed) — into
    /// `build_talker_prompt` so a media position gets
    /// `hidden_projection(hidden)` instead of the text branch. From there,
    /// identical to `speak`: Talker -> MTP -> Code2Wav, chunked/streamed the
    /// same way (see [`Self::speak`]'s doc on `on_chunk`).
    #[allow(clippy::too_many_arguments)]
    pub fn converse(&self, prompt: &str, audio: Option<&[f32]>, image: Option<(&[f32], u32, u32)>, video: Option<&[(Vec<f32>, u32, u32)]>, max_new: u32, speaker: &str, mut on_chunk: impl FnMut(&[f32])) -> Result<(String, Vec<f32>, u32), String> {
        let text_ids = self.tok.encode(prompt);
        let mm_prompt = build_multimodal_prompt(&self.reader, &self.gpus[0], &self.cfg.thinker, &self.embed_host(), &text_ids, audio, image, video)?;
        let n_prompt = mm_prompt.token_ids.len() as u32;
        let out_ids = generate_greedy_multimodal(&self.stack, &self.gpus, &self.reader, &self.cfg.thinker.text, &self.embed, &mm_prompt, max_new, &self.eos_ids);
        let new_ids = out_ids[n_prompt as usize..].to_vec();
        let text = self.tok.decode(&new_ids);

        let tc = &self.cfg.talker;
        let text_proj = crate::codec_bridge::load_talker_projection(&self.reader, tc, "text_projection")?;
        let hidden_proj = crate::codec_bridge::load_talker_projection(&self.reader, tc, "hidden_projection")?;
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

        let user_hidden = thinker_hidden_at_layer(&self.stack, &self.gpus, &self.reader, &self.cfg.thinker.text, &mm_prompt.x_host, &mm_prompt.positions, n_prompt, tc.accept_hidden_layer);
        let tcfg = &self.cfg.thinker;
        let media_mask: Vec<bool> = mm_prompt.token_ids.iter().map(|&t| t == tcfg.audio_token_id || t == tcfg.image_token_id || t == tcfg.video_token_id).collect();
        let splice = UserMediaSplice { hidden_proj: &hidden_proj, hidden: &user_hidden, media_mask: &media_mask };

        let tprompt = build_talker_prompt(&text_proj, &codec_embed, &specials, speaker_id, &self.embed_host(), self.cfg.thinker.text.hidden as usize, &mm_prompt.token_ids, &new_ids, Some(splice));

        let mtp_gpu = self.gpus[0].new_like(tts::mtp::PIPELINES);
        let mtp = crate::codec_bridge::load_mtp(&self.reader, mtp_gpu, &tc.code_predictor)?;
        let codec_head_w = self.reader.tensor("talker.codec_head.weight").ok_or("omni: missing tensor talker.codec_head.weight")?;
        let talker_gpu = self.gpus[0].new_like(crate::talker::talker_pipelines());
        let codes = talker_generate::generate_codes(&self.reader, &talker_gpu, &tc.text, &codec_head_w, tc.codec_eos_token_id, &mtp, codec_embed, &tprompt, &GenOpts::default())?;

        let codec = crate::codec_bridge::load_codec(&self.reader, &self.cfg.code2wav)?;
        let mut wav = Vec::new();
        codec.decode_omni_chunked(&codes, SPEAK_CHUNK_FRAMES, |chunk| {
            wav.extend_from_slice(chunk);
            on_chunk(chunk);
        });
        Ok((text, wav, self.cfg.code2wav.output_sample_rate))
    }
}

/// Code frames (12.5 Hz) per `Codec::decode_omni_chunked` call in
/// `OmniInner::speak`/`converse` — matching the reference's own
/// `chunked_decode(chunk_size=300, ...)` convention (`Codec::
/// decode_omni_chunked`'s own doc has the full reasoning for why this
/// implementation's front/back split doesn't need the reference's
/// `left_context_size` re-decode-and-discard shape).
const SPEAK_CHUNK_FRAMES: usize = 300;

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
        let mut n_chunks = 0u32;
        let (text, wav, sample_rate) = self.inner.speak(&prompt, max_new, &speaker, |chunk| {
            n_chunks += 1;
            let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
            let blob = Blob::new(Media::Audio, bytes).with_meta(json!({"index": n_chunks}));
            progress(Progress::chunk(1, 2, format!("audio chunk {n_chunks}"), "audio", blob));
        })?;
        let audio_bytes: Vec<u8> = wav.iter().flat_map(|s| s.to_le_bytes()).collect();
        progress(Progress::step(2, 2, "done"));
        Ok(Outcome::new()
            .set("text", json!(text))
            .blob("text", Blob::new(Media::Text, text.into_bytes()))
            .blob("audio", Blob::new(Media::Audio, audio_bytes).with_meta(json!({"sample_rate": sample_rate}))))
    }
}

struct ConverseAction {
    inner: Arc<OmniInner>,
}

impl Action for ConverseAction {
    fn spec(&self) -> ActionSpec {
        converse_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let prompt = last_user_text(inv);
        if prompt.trim().is_empty() {
            return Err("omni converse: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
        }
        let max_new = inv.get_i64("max_new").unwrap_or(32).clamp(1, 4096) as u32;
        let speaker = inv.get_str("speaker").unwrap_or_else(|| "chelsie".to_string());

        let audio = inv.get_blob("audio").map(audio::asr_caps::wav_from_blob).transpose()?;
        let image = inv.get_blob("image").map(|_| decode_image(inv, "image")).transpose()?;
        let video = inv.get_blob("video").map(|_| decode_video_hwc(inv, "video")).transpose()?;
        let image_ref = image.as_ref().map(|(hwc, w, h)| (hwc.as_slice(), *w, *h));

        progress(Progress::step(0, 2, "generating text"));
        let mut n_chunks = 0u32;
        let (text, wav, sample_rate) = self.inner.converse(&prompt, audio.as_deref(), image_ref, video.as_deref(), max_new, &speaker, |chunk| {
            n_chunks += 1;
            let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
            let blob = Blob::new(Media::Audio, bytes).with_meta(json!({"index": n_chunks}));
            progress(Progress::chunk(1, 2, format!("audio chunk {n_chunks}"), "audio", blob));
        })?;
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

    /// Spec: `converse` resolves to its own distinct handler (not aliasing
    /// `speak` or `generate`), and its schema really is the union its doc
    /// claims — every multimodal input `generate_spec` declares AND the
    /// audio output `speak_spec` declares. A `converse_spec` missing either
    /// half would validate fine but silently drop real input or promise an
    /// output `ConverseAction::run` never produces.
    #[test]
    fn converse_resolves_distinctly_and_declares_the_full_union_shape() {
        let c = resolve_action("converse").expect("converse must resolve");
        assert_eq!(c, OmniActionKind::Converse);
        assert_ne!(c, OmniActionKind::Generate);
        assert_ne!(c, OmniActionKind::Speak);

        let spec = converse_spec();
        for name in ["audio", "image", "video"] {
            assert!(spec.inputs.iter().any(|b| b.name == name), "converse_spec must declare input '{name}'");
        }
        assert!(spec.outputs.iter().any(|b| b.media == Media::Text), "converse_spec must declare a text output");
        assert!(spec.outputs.iter().any(|b| b.media == Media::Audio), "converse_spec must declare an audio output");
    }
}
