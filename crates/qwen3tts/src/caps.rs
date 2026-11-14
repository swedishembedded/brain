// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS's capabilities behind the generalized [`capability`] interface:
//! what `brain serve`, D-Bus/HTTP clients, `brain qwen3tts <action>` and the
//! `sample-models-qwen3tts` program all drive with the same parameters.
//!
//! Actions, each a thin wrapper over the pipeline in [`crate::pipeline`]:
//!   - `synth` - speaker-free text-to-speech.
//!   - `clone` - voice cloning from a reference WAV (`ref`), x-vector-only or
//!     in-context when `ref_text` is given; `ref_codes` hands over externally
//!     encoded reference codes.
//!   - `design` - instruct-style VoiceDesign and/or CustomVoice preset speakers.
//!   - `batch` - N speaker-free requests decoded interleaved in one call.
//!   - `import` - the Hugging Face release as brain's four component
//!     checkpoints ([`crate::import::import_all`]).
//!   - `finetune` - single-speaker SFT of the Talker on a `text -> codes`
//!     dataset, LoRA or full ([`crate::sft`]).
//!   - `sim` - speaker-embedding cosine similarity of two utterances
//!     ([`crate::reference::speaker_similarity`]).
//!
//! The checkpoint is the provider's own, resolved from the model store through
//! [`crate::spec::Qwen3TtsSpec`]'s `weights_dir` and `ckpt` roles (the roles
//! are the command line's `--weights-dir`/`--ckpt`), never a per-call param.
//! The weights load per call - the load-once seams are
//! [`crate::engine::ResidentEngine`] (the resident adapter's `speak`, `clone`
//! and `design`, streamed in chunks) and the NPU engine in [`crate::serve`].
//!
//! Each generation action forwards the invocation's [`capability::CancelToken`]
//! into the pipeline, which polls it between generated frames, so a caller
//! that hangs up stops the autoregressive loop instead of waiting out
//! `max_frames`.
//!
//! Swedish Embedded AB implements on-device speech synthesis and voice cloning
//! for its clients. If your team needs expertise in text-to-speech on its own
//! hardware then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use serde_json::json;

use crate::reference::Reference;
use crate::{FinetuneOpts, GenOpts, TtsPaths};

/// The model id on the command line (`brain qwen3tts …`) and the event API.
pub const MODEL: &str = "brain/qwen3tts";

/// Qwen3-TTS output sample rate.
const SAMPLE_RATE: u32 = 24_000;

/// Most codec frames (12.5 per second) one generation may run.
const MAX_FRAMES: f64 = 16_384.0;
/// Sampling temperature ceiling; a temperature past it is noise, not variety.
const MAX_TEMPERATURE: f64 = 10.0;
/// Repetition-penalty ceiling.
const MAX_REPETITION_PENALTY: f64 = 10.0;
/// Most codec frames one streamed audio chunk may hold.
const MAX_STREAM_CHUNK: f64 = 1_024.0;
/// Largest LoRA alpha a run may ask for.
const MAX_ALPHA: f64 = 4_096.0;
/// Most examples one fine-tune step may average over.
const MAX_BATCH: f64 = 1_024.0;

const RESOLVED: &str = "unset resolves from the checkpoint's generation_config.json, then the reference default";

/// The generation knobs every speech action shares.
///
/// **The sampling knobs deliberately declare no `default`.**
/// [`capability::ActionSpec::validate`] fills a declared default INTO the
/// invocation before an action ever sees it, so a `ParamSpec` default is
/// indistinguishable from a caller who typed the value - it would shadow the
/// checkpoint's own `generation_config.json` on every single call and make the
/// precedence chain in [`crate::genconfig`] dead code. That is exactly the
/// drift that shipped `repetition_penalty = 1.0` while the checkpoint said
/// `1.05`. Unset here means unset, and `profile` reports the resolved plan.
///
/// `residual_*` are the same three knobs for the MTP's residual codebooks
/// (1..15), which are sampled like codebook 0; `residual_temp 0` pins them to
/// greedy.
fn generation_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("lang", ParamType::Str, "synthesis language").default(json!("english")))
        .param(ParamSpec::new("max_frames", ParamType::Int, "max codec frames (length cap)").min(0.0).max(MAX_FRAMES).default(json!(256)))
        .param(ParamSpec::new("temp", ParamType::Float, &format!("sampling temperature (0 = greedy, degenerate for this model); {RESOLVED} 0.9")).min(0.0).max(MAX_TEMPERATURE))
        .param(ParamSpec::new("top_k", ParamType::Int, &format!("top-k sampling cutoff; {RESOLVED} 50")).min(0.0).max(capability::bounds::TOP_K))
        .param(ParamSpec::new("top_p", ParamType::Float, &format!("nucleus sampling cutoff (0 or 1 = disabled); {RESOLVED} 1.0")).min(0.0).max(capability::bounds::UNIT))
        .param(
            ParamSpec::new(
                "repetition_penalty",
                ParamType::Float,
                &format!("repetition penalty (1.0 = disabled; the resolved value is what keeps codebook-0 out of a silent repetition loop); {RESOLVED} 1.05"),
            )
            .min(0.0)
            .max(MAX_REPETITION_PENALTY),
        )
        .param(ParamSpec::new("residual_temp", ParamType::Float, &format!("sampling temperature of the residual codebooks (0 = greedy); {RESOLVED}")).min(0.0).max(MAX_TEMPERATURE))
        .param(ParamSpec::new("residual_top_k", ParamType::Int, &format!("top-k cutoff of the residual codebooks; {RESOLVED}")).min(0.0).max(capability::bounds::TOP_K))
        .param(ParamSpec::new("residual_top_p", ParamType::Float, &format!("nucleus cutoff of the residual codebooks; {RESOLVED}")).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (reproducible run; omit for random, the outcome echoes the one used)").min(0.0))
        .param(ParamSpec::new("profile", ParamType::Bool, "report the resolved sampling plan and each stage's wall time on the serving machine's stderr").default(json!(false)))
}

/// [`generation_params`] plus the single-request `text`, shared by
/// `synth`/`clone`/`design` (and the resident adapter's actions). `batch`
/// carries its texts inside its own `requests` array and must NOT declare a
/// top-level `text`.
pub fn speech_params(spec: ActionSpec) -> ActionSpec {
    generation_params(spec).param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
}

/// The reference-voice inputs of a clone: the reference WAV and optional
/// externally encoded codes. Raw bytes, not decoded audio: the speaker encoder
/// and the codec encoder each want the reference at its own sample rate.
pub fn reference_inputs(spec: ActionSpec) -> ActionSpec {
    spec.input(BlobSpec::new("ref", Media::Bytes, "the reference voice: a WAV recording of the speaker to clone").required())
}

fn ref_text_param() -> ParamSpec {
    ParamSpec::new("ref_text", ParamType::Str, "transcript of the reference recording - given, runs in-context (ICL) cloning; omitted, x-vector-only").default(json!(""))
}

fn instruct_params(spec: ActionSpec) -> ActionSpec {
    spec.param(ParamSpec::new("instruct", ParamType::Str, "natural-language voice/emotion/prosody description").default(json!("")))
        .param(ParamSpec::new("speaker", ParamType::Str, "CustomVoice preset speaker name").default(json!("")))
}

/// Decode the shared generation knobs. A `seed` that is not a non-negative
/// integer is refused; an absent one is drawn here, so the opts always carry
/// the seed the run uses and the outcome can echo it.
///
/// A param the caller did not pass stays `None` and is answered later by
/// [`crate::genconfig`]'s chain - the invocation IS the "explicit caller
/// override" layer, so anything absent from it must stay absent here.
pub fn gen_opts_from(inv: &Invocation) -> Result<GenOpts, String> {
    let mut opts = GenOpts::default();
    if let Some(f) = inv.get_i64("max_frames") {
        opts.max_frames = f.max(1) as usize;
    }
    let float = |name: &str| inv.get_f64(name).map(|v| v as f32);
    let count = |name: &str| inv.get_i64(name).map(|v| v.max(0) as usize);
    opts.sampling.temperature = float("temp");
    opts.sampling.top_k = count("top_k");
    opts.sampling.top_p = float("top_p");
    opts.sampling.repetition_penalty = float("repetition_penalty");
    opts.sampling.subtalker.temperature = float("residual_temp");
    opts.sampling.subtalker.top_k = count("residual_top_k");
    opts.sampling.subtalker.top_p = float("residual_top_p");
    opts.profile = inv.get_bool("profile").unwrap_or(false);
    opts.seed = inv.seed()?.unwrap_or_else(data::rng::random_seed);
    Ok(opts)
}

/// The terminal outcome of a speech action: the whole clip as a WAV, and the
/// seed it was generated under.
pub fn audio_outcome(wav: Vec<f32>, seed: u64) -> ActionResult {
    let bytes = audio::wav::encode(&wav, SAMPLE_RATE);
    Ok(Outcome::new()
        .set("seed", json!(seed))
        .set("samples", json!(wav.len()))
        .set("sample_rate", json!(SAMPLE_RATE))
        .set("seconds", json!(wav.len() as f64 / SAMPLE_RATE as f64))
        .blob("audio", Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": SAMPLE_RATE, "format": "wav", "channels": 1}))))
}

pub fn synth_spec() -> ActionSpec {
    speech_params(ActionSpec::new("synth", "speaker-free text-to-speech (Talker + MTP + codec, 24 kHz wav)").cancellable())
        .output(BlobSpec::new("audio", Media::Audio, "the synthesized speech as a 24 kHz mono WAV"))
}

pub fn clone_spec() -> ActionSpec {
    reference_inputs(speech_params(ActionSpec::new("clone", "voice cloning from a reference wav (x-vector-only, or in-context when ref_text is given)").cancellable()))
        .param(ref_text_param())
        .input(BlobSpec::new("ref_codes", Media::Bytes, "the reference's [frames,16] codec codes, encoded elsewhere (an 8-byte little-endian count, then u32 little-endian codes); overrides the in-tree encode of the reference"))
        .output(BlobSpec::new("audio", Media::Audio, "the cloned-voice speech as a 24 kHz mono WAV"))
}

pub fn design_spec() -> ActionSpec {
    instruct_params(speech_params(ActionSpec::new("design", "VoiceDesign (instruct) and/or CustomVoice preset speakers - needs a CustomVoice/VoiceDesign checkpoint").cancellable()))
        .output(BlobSpec::new("audio", Media::Audio, "the designed-voice speech as a 24 kHz mono WAV"))
}

/// The `batch` action: N speaker-free requests in one call, decoded
/// **interleaved** by [`crate::batch::run_batch`] (round-robin, one frame per
/// request per round, each request ending independently at its own EOS or
/// `max_frames`) instead of run to completion one after another.
///
/// The requests arrive as a JSON array in the `requests` param rather than a
/// typed array param, because [`capability::ParamType`] has no array kind -
/// the same JSON-in-a-string convention `chat`'s `messages` param already
/// uses across this repo. Each element is an object with a required `text`
/// and optional per-request overrides of the action-level sampling knobs;
/// each produces its own `audio_<i>` WAV output blob, indexed in request
/// order.
pub fn batch_spec() -> ActionSpec {
    generation_params(ActionSpec::new(
        "batch",
        "synthesize N texts in one interleaved (ragged) batch - one 24 kHz WAV per request, in request order",
    ).cancellable())
    .streaming()
    .param(
        ParamSpec::new(
            "requests",
            ParamType::Str,
            "JSON array of request objects: [{\"text\":\"...\",\"lang\":\"english\",\"max_frames\":64,\"seed\":1,\"temp\":0.9,\"top_k\":50,\"top_p\":1.0,\"repetition_penalty\":1.05}, ...]; every key but 'text' is optional and falls back to this call's own value, then to the checkpoint's generation_config.json",
        )
        .required(),
    )
    .output(BlobSpec::new("audio_0", Media::Audio, "request 0's speech as a 24 kHz mono WAV; request i's is 'audio_<i>'"))
}

pub fn import_spec() -> ActionSpec {
    ActionSpec::new("import", "convert the Hugging Face release into brain's four component checkpoints (talker, mtp, codec, speaker); a release with no speaker encoder (CustomVoice, VoiceDesign) skips it")
        .cancellable()
        .from_scratch()
        .param(ParamSpec::new("source", ParamType::Str, "the Hugging Face checkpoint directory (the Talker, the MTP and the speaker encoder), on the serving machine").required().host_resolved())
        .param(ParamSpec::new("codec_source", ParamType::Str, "the speech tokenizer's directory, on the serving machine; omit for the release's speech_tokenizer subdirectory").host_resolved())
        .param(ParamSpec::new("speaker_source", ParamType::Str, "the speaker encoder's directory, on the serving machine; omit for the release itself").host_resolved())
        .param(ParamSpec::new("dest", ParamType::Str, "the directory to write talker.safetensors, mtp.safetensors, codec.safetensors and speaker.safetensors to, on the serving machine").required().host_resolved())
}

pub fn finetune_spec() -> ActionSpec {
    let defaults = FinetuneOpts::default();
    ActionSpec::new("finetune", "single-speaker SFT of the Talker on a text -> codes dataset: LoRA adapters over a frozen base, or every weight with `full`")
        .training()
        .streaming()
        .cancellable()
        .param(ParamSpec::new("data", ParamType::Str, "the dataset folder holding train.u32.bin and val.u32.bin, on the serving machine").required().host_resolved())
        .param(ParamSpec::new("save", ParamType::Str, "where to write the fine-tuned checkpoint (.safetensors) on the serving machine").required().host_resolved())
        .param(ParamSpec::new("full", ParamType::Bool, "train every Talker weight (Qwen's documented single-speaker fine-tuning) instead of the LoRA attention adapters").default(json!(false)))
        .param(ParamSpec::new("steps", ParamType::Int, "optimizer steps").default(json!(defaults.steps)).min(1.0).max(capability::bounds::TRAIN_STEPS))
        .param(ParamSpec::new("lr", ParamType::Float, "peak learning rate (cosine schedule)").default(json!(defaults.lr)).min(0.0).max(capability::bounds::UNIT))
        .param(ParamSpec::new("rank", ParamType::Int, "LoRA rank; ignored by `full`").default(json!(defaults.rank)).min(1.0).max(capability::bounds::LORA_RANK))
        .param(ParamSpec::new("alpha", ParamType::Float, "LoRA alpha (the delta is scaled by alpha/rank); ignored by `full`").default(json!(defaults.alpha)).min(0.0).max(MAX_ALPHA))
        .param(ParamSpec::new("batch", ParamType::Int, "examples averaged into each step").default(json!(defaults.batch)).min(1.0).max(MAX_BATCH))
        .param(ParamSpec::new("block", ParamType::Int, "training context, tokens").default(json!(defaults.block)).min(1.0).max(capability::bounds::CONTEXT_TOKENS))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed of the adapter init and the batch order (omit for random, the outcome echoes the one used)").min(0.0))
        .param(ParamSpec::new("activations", ParamType::Enum(model::saved::SavedPolicy::ACTIVATIONS_CHOICES.map(String::from).to_vec()), "what the backward keeps of the forward: every unit's intermediates, or only each unit's input, rebuilt in the backward; the Talker recomputes each layer, so keep is refused by name before the first step (default: the model's own)"))
}

pub fn sim_spec() -> ActionSpec {
    ActionSpec::new("sim", "speaker-embedding cosine similarity of two utterances (ECAPA x-vectors): the timbre-preservation metric, e.g. a quantized Talker's clone against its reference")
        .cancellable()
        .input(BlobSpec::new("a", Media::Bytes, "the first utterance: a WAV recording").required())
        .input(BlobSpec::new("b", Media::Bytes, "the second utterance: a WAV recording").required())
}

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "Qwen3-TTS voice synthesis - text to 24 kHz speech (Talker + MTP + codec), voice cloning, import, fine-tuning and speaker similarity.",
        vec![synth_spec(), clone_spec(), design_spec(), batch_spec(), import_spec(), finetune_spec(), sim_spec()],
    )
}

/// The resident adapter's variant of a speech action: it streams its audio.
/// Each decoded codec chunk is a `Progress::chunk` named `audio` (raw mono f32
/// little-endian PCM at 24 kHz), which D-Bus `Subscribe` turns into a real
/// out-of-band blob frame, while the terminal `Outcome` still carries the FULL
/// waveform as a WAV for a plain `Run` caller.
pub fn streamed(spec: ActionSpec) -> ActionSpec {
    spec.streaming()
        .param(ParamSpec::new("stream_chunk", ParamType::Int, "codec frames decoded per streamed audio chunk; omit for the engine's default").min(1.0).max(MAX_STREAM_CHUNK))
}

/// The resident (`brain serve`) manifest: [`manifest`], with the speech
/// actions the hot engine runs ([`RESIDENT_ACTIONS`]) in their [`streamed`]
/// form. The rest (`batch`, `import`, `finetune`, `sim`) are the provider's.
pub fn resident_manifest() -> Manifest {
    let mut m = manifest();
    m.actions = m.actions.into_iter().map(|a| if RESIDENT_ACTIONS.contains(&a.name.as_str()) { streamed(a) } else { a }).collect();
    m
}

/// The actions the resident engine runs hot, load-once, streamed.
pub const RESIDENT_ACTIONS: [&str; 3] = ["synth", "clone", "design"];

/// The reference voice of a `clone` invocation, and its optional external codes.
pub fn reference_from(inv: &Invocation, action: &str) -> Result<(Reference, Option<Vec<u32>>), String> {
    let blob = inv.get_blob("ref").ok_or_else(|| format!("tts {action}: missing required input 'ref' (the reference WAV)"))?;
    let reference = Reference::from_bytes(&blob.bytes).map_err(|e| format!("tts {action}: {e}"))?;
    let codes = inv.get_blob("ref_codes").map(|b| crate::reference::parse_codes(&b.bytes).map_err(|e| format!("tts {action}: ref_codes: {e}"))).transpose()?;
    Ok((reference, codes))
}

/// Qwen3-TTS over the checkpoint of `assembly`'s `weights_dir` and `ckpt`
/// roles; an assembly without them serves only the actions that need no
/// checkpoint (`import`).
pub fn provider(assembly: &capability::Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(TtsProvider::new(TtsPaths::from_assembly(assembly).ok())))
}

/// The executable TTS model behind the manifest. Stateless: the pipeline owns
/// loading (per call), so construction is free and there is no hot cache.
#[derive(Default)]
pub struct TtsProvider {
    paths: Option<TtsPaths>,
}

impl TtsProvider {
    pub fn new(paths: Option<TtsPaths>) -> TtsProvider {
        TtsProvider { paths }
    }
}

impl Provider for TtsProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        let paths = self.paths.clone();
        match name {
            "synth" => Some(Arc::new(SynthAction { paths })),
            "clone" => Some(Arc::new(CloneAction { paths })),
            "design" => Some(Arc::new(DesignAction { paths })),
            "batch" => Some(Arc::new(BatchAction { paths })),
            "import" => Some(Arc::new(ImportAction)),
            "finetune" => Some(Arc::new(FinetuneAction { paths })),
            "sim" => Some(Arc::new(SimAction { paths })),
            _ => None,
        }
    }
}

/// The provider's checkpoint, or the instruction to name one. `need_speaker`
/// additionally requires `speaker.safetensors` (only `clone` and `sim` do -
/// CustomVoice/VoiceDesign checkpoints don't ship a speaker encoder). Fails
/// cleanly (not a panic in the loaders) when a component is absent.
fn checkpoint(paths: &Option<TtsPaths>, action: &str, need_speaker: bool) -> Result<TtsPaths, String> {
    let paths = paths.clone().ok_or_else(|| format!("tts {action}: no Qwen3-TTS checkpoint (pull one, or name it with --weights-dir and --ckpt)"))?;
    paths.require(need_speaker).map_err(|e| format!("tts {action}: {e}"))?;
    Ok(paths)
}

/// The checkpoint plus the request every speech action shares: non-empty
/// `text`, `lang` and the generation knobs. Public so a front end that runs
/// the same request on another engine (the NPU route) reads it identically.
pub fn speech_request(paths: &Option<TtsPaths>, inv: &Invocation, action: &str, need_speaker: bool) -> Result<(TtsPaths, String, String, GenOpts), String> {
    let text = inv.get_str("text").unwrap_or_default();
    if text.trim().is_empty() {
        return Err(format!("tts {action}: 'text' must be non-empty"));
    }
    let lang = inv.get_str("lang").unwrap_or_else(|| "english".to_string());
    let opts = gen_opts_from(inv).map_err(|e| format!("tts {action}: {e}"))?;
    Ok((checkpoint(paths, action, need_speaker)?, text, lang, opts))
}

struct SynthAction {
    paths: Option<TtsPaths>,
}

impl Action for SynthAction {
    fn spec(&self) -> ActionSpec {
        synth_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let (paths, text, lang, opts) = speech_request(&self.paths, inv, "synth", false)?;
        let wav = crate::pipeline::synth(&paths, &opts, &text, &lang, &inv.cancel)?;
        audio_outcome(wav, opts.seed)
    }
}

struct CloneAction {
    paths: Option<TtsPaths>,
}

impl Action for CloneAction {
    fn spec(&self) -> ActionSpec {
        clone_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let (paths, text, lang, opts) = speech_request(&self.paths, inv, "clone", true)?;
        let (reference, ref_codes) = reference_from(inv, "clone")?;
        let ref_text = inv.get_str("ref_text").unwrap_or_default();
        let wav = crate::pipeline::clone(&paths, &opts, &text, &reference, &ref_text, &lang, ref_codes, &inv.cancel)?;
        audio_outcome(wav, opts.seed)
    }
}

struct DesignAction {
    paths: Option<TtsPaths>,
}

impl Action for DesignAction {
    fn spec(&self) -> ActionSpec {
        design_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let (paths, text, lang, opts) = speech_request(&self.paths, inv, "design", false)?;
        let instruct = inv.get_str("instruct").unwrap_or_default();
        let speaker = inv.get_str("speaker").filter(|s| !s.is_empty());
        let wav = crate::pipeline::design(&paths, &opts, &text, &lang, &instruct, speaker.as_deref(), &inv.cancel)?;
        audio_outcome(wav, opts.seed)
    }
}

struct BatchAction {
    paths: Option<TtsPaths>,
}

impl Action for BatchAction {
    fn spec(&self) -> ActionSpec {
        batch_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let raw = inv.get_str("requests").ok_or("tts batch: missing required param 'requests'")?;
        let items: Vec<serde_json::Value> =
            serde_json::from_str(&raw).map_err(|e| format!("tts batch: 'requests' must be a JSON array of objects: {e}"))?;
        // Action-level knobs are the per-request defaults, so a caller that
        // wants one setting for the whole batch sets it once.
        let base_lang = inv.get_str("lang").unwrap_or_else(|| "english".to_string());
        let base = gen_opts_from(inv).map_err(|e| format!("tts batch: {e}"))?;
        let mut reqs = Vec::with_capacity(items.len());
        let mut seeds = Vec::with_capacity(items.len());
        for (i, it) in items.iter().enumerate() {
            let text = it["text"].as_str().unwrap_or_default().to_string();
            if text.trim().is_empty() {
                return Err(format!("tts batch: request {i} needs a non-empty 'text'"));
            }
            let mut opts = base.clone();
            if let Some(v) = it["max_frames"].as_i64() {
                opts.max_frames = v.max(1) as usize;
            }
            // A per-request key present here is that request's own explicit
            // choice; absent, it inherits the call-level one, which is itself
            // `None` unless the caller passed it. Nothing here invents a value.
            if let Some(v) = it["temp"].as_f64() {
                opts.sampling.temperature = Some(v as f32);
            }
            if let Some(v) = it["top_k"].as_i64() {
                opts.sampling.top_k = Some(v.max(0) as usize);
            }
            if let Some(v) = it["top_p"].as_f64() {
                opts.sampling.top_p = Some(v as f32);
            }
            if let Some(v) = it["repetition_penalty"].as_f64() {
                opts.sampling.repetition_penalty = Some(v as f32);
            }
            match &it["seed"] {
                serde_json::Value::Null => {}
                v => opts.seed = v.as_u64().ok_or_else(|| format!("tts batch: request {i}: seed {v} is not a seed (a non-negative integer)"))?,
            }
            let lang = it["lang"].as_str().unwrap_or(&base_lang).to_string();
            seeds.push(opts.seed);
            reqs.push(crate::batch::BatchRequest { text, lang, opts });
        }
        // An empty batch is the batch's own error, before any weight lookup.
        if reqs.is_empty() {
            return Err("tts batch: needs at least one request".to_string());
        }
        let paths = checkpoint(&self.paths, "batch", false)?;
        let n = reqs.len() as u32;
        progress(Progress::step(0, n, format!("generating {n} requests, interleaved")));
        let wavs = crate::batch::synth_batch(&paths, &reqs, &inv.cancel)?;
        progress(Progress::step(n, n, "decoded"));

        let mut outcome = Outcome::new()
            .set("count", json!(wavs.len()))
            .set("seeds", json!(seeds))
            .set("sample_rate", json!(SAMPLE_RATE))
            .set("samples", json!(wavs.iter().map(|w| w.len()).collect::<Vec<_>>()))
            .set("seconds", json!(wavs.iter().map(|w| w.len() as f64 / SAMPLE_RATE as f64).collect::<Vec<_>>()));
        for (i, wav) in wavs.iter().enumerate() {
            let bytes = audio::wav::encode(wav, SAMPLE_RATE);
            outcome = outcome.blob(
                &format!("audio_{i}"),
                Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": SAMPLE_RATE, "format": "wav", "channels": 1, "index": i})),
            );
        }
        Ok(outcome)
    }
}

struct ImportAction;

impl Action for ImportAction {
    fn spec(&self) -> ActionSpec {
        import_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let source = inv.get_str("source").ok_or("tts import: missing required param 'source'")?;
        let dest = inv.get_str("dest").ok_or("tts import: missing required param 'dest'")?;
        let sources = crate::import::Sources::new(&source, inv.get_str("codec_source").as_deref(), inv.get_str("speaker_source").as_deref());
        let imported = crate::import::import_all(&sources, &dest).map_err(|e| format!("tts import: {e}"))?;
        Ok(Outcome::new().set("dest", json!(dest)).set("speaker_skipped", json!(imported.speaker_skipped)))
    }
}

struct FinetuneAction {
    paths: Option<TtsPaths>,
}

impl Action for FinetuneAction {
    fn spec(&self) -> ActionSpec {
        finetune_spec()
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let base = checkpoint(&self.paths, "finetune", false)?.talker;
        let data = inv.get_str("data").ok_or("tts finetune: missing required param 'data'")?;
        let save = inv.get_str("save").ok_or("tts finetune: missing required param 'save'")?;
        let defaults = FinetuneOpts::default();
        let opts = FinetuneOpts {
            steps: inv.get_i64("steps").map_or(defaults.steps, |v| v.max(1) as u32),
            lr: inv.get_f64("lr").map_or(defaults.lr, |v| v as f32),
            rank: inv.get_i64("rank").map_or(defaults.rank, |v| v.max(1) as u32),
            alpha: inv.get_f64("alpha").map_or(defaults.alpha, |v| v as f32),
            batch: inv.get_i64("batch").map_or(defaults.batch, |v| v.max(1) as u32),
            block: inv.get_i64("block").map_or(defaults.block, |v| v.max(1) as u32),
            seed: inv.seed().map_err(|e| format!("tts finetune: {e}"))?.unwrap_or(defaults.seed),
            saved: inv.get_str("activations").map(|a| model::saved::SavedPolicy::parse_activations(&a)).transpose().map_err(|e| format!("tts finetune: {e}"))?,
            ..defaults
        };
        let full = inv.get_bool("full").unwrap_or(false);
        if let Some(parent) = Path::new(&save).parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| format!("tts finetune: {}: {e}", parent.display()))?;
        }
        let mut on_step = |r: &model::StepReport| {
            progress(Progress::step(r.step, r.steps, format!("loss {:.4}", r.loss)));
            !inv.cancel.is_cancelled()
        };
        let trained = if full {
            crate::sft::finetune_full(&base, Path::new(&data), &save, &opts, &mut on_step)
        } else {
            crate::sft::finetune_lora(&base, Path::new(&data), &save, &opts, &mut on_step)
        };
        let (initial, last) = trained.map_err(|e| format!("tts finetune: {e}"))?;
        Ok(Outcome::new()
            .set("mode", json!(if full { "full" } else { "lora" }))
            .set("initial_loss", json!(initial))
            .set("last_loss", json!(last))
            .set("seed", json!(opts.seed))
            .set("saved", json!(save)))
    }
}

struct SimAction {
    paths: Option<TtsPaths>,
}

impl Action for SimAction {
    fn spec(&self) -> ActionSpec {
        sim_spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        let paths = checkpoint(&self.paths, "sim", true)?;
        let utterance = |name: &str| {
            let blob = inv.get_blob(name).ok_or_else(|| format!("tts sim: missing required input '{name}'"))?;
            Reference::from_bytes(&blob.bytes).map_err(|e| format!("tts sim: input '{name}': {e}"))
        };
        let (a, b) = (utterance("a")?, utterance("b")?);
        let cosine = crate::reference::speaker_similarity(&paths.speaker, &a, &b);
        Ok(Outcome::new().set("cosine", json!(cosine)))
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;
    use capability::Registry;

    fn registry(paths: Option<TtsPaths>) -> Registry {
        let mut reg = Registry::new();
        reg.register(Arc::new(TtsProvider::new(paths)));
        reg
    }

    fn missing_paths() -> TtsPaths {
        TtsPaths::new("/nonexistent/tts", "/nonexistent/ckpt")
    }

    fn wav_blob(samples: &[f32]) -> Blob {
        Blob::new(Media::Bytes, audio::wav::encode(samples, 24_000))
    }

    #[test]
    fn a_seed_that_is_not_a_non_negative_integer_is_refused_before_any_work() {
        for bad in [json!(-1), json!(1.5), json!("7")] {
            let inv = Invocation::new().set("text", json!("hi")).set("seed", bad.clone());
            let err = SynthAction { paths: Some(missing_paths()) }.run(&inv, &mut |_| {}).unwrap_err();
            assert!(err.contains("seed"), "seed {bad} must be refused as a seed, got: {err}");
        }
    }

    #[test]
    fn a_seedless_request_draws_a_seed_and_the_outcome_echoes_the_one_used() {
        let inv = Invocation::new();
        let (a, b) = (gen_opts_from(&inv).unwrap().seed, gen_opts_from(&inv).unwrap().seed);
        assert_ne!(a, b, "an absent seed must be drawn per request");
        let given = gen_opts_from(&Invocation::new().set("seed", json!(41))).unwrap();
        assert_eq!(given.seed, 41);
        let out = audio_outcome(vec![0.0; 4], 41).unwrap();
        assert_eq!(out.outputs["seed"], json!(41));
    }

    /// Every knob of the CLI verbs is a served param: the sampling set for
    /// both codebook streams reaches [`GenOpts`], and nothing is invented for
    /// a knob the caller left out.
    #[test]
    fn every_sampling_knob_reaches_the_generation_options_and_an_absent_one_stays_unset() {
        let none = gen_opts_from(&Invocation::new()).unwrap();
        assert_eq!((none.sampling.temperature, none.sampling.top_k, none.sampling.top_p, none.sampling.repetition_penalty), (None, None, None, None));
        assert_eq!((none.sampling.subtalker.temperature, none.sampling.subtalker.top_k, none.sampling.subtalker.top_p), (None, None, None));
        assert!(!none.profile);

        let inv = Invocation::new()
            .set("max_frames", json!(64))
            .set("temp", json!(0.7))
            .set("top_k", json!(20))
            .set("top_p", json!(0.9))
            .set("repetition_penalty", json!(1.2))
            .set("residual_temp", json!(0.0))
            .set("residual_top_k", json!(8))
            .set("residual_top_p", json!(0.5))
            .set("profile", json!(true));
        let o = gen_opts_from(&inv).unwrap();
        assert_eq!(o.max_frames, 64);
        assert_eq!((o.sampling.temperature, o.sampling.top_k, o.sampling.top_p, o.sampling.repetition_penalty), (Some(0.7), Some(20), Some(0.9), Some(1.2)));
        // `residual_temp 0` is an explicit greedy pin, distinct from unset.
        assert_eq!((o.sampling.subtalker.temperature, o.sampling.subtalker.top_k, o.sampling.subtalker.top_p), (Some(0.0), Some(8), Some(0.5)));
        assert!(o.profile);
    }

    #[test]
    fn the_manifest_declares_every_action_with_no_weights_params() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["synth", "clone", "design", "batch", "import", "finetune", "sim"]);
        for a in &m.actions {
            // The checkpoint is the provider's (the store's roles), never a param.
            for gone in ["weights_dir", "ckpt"] {
                assert!(!a.params.iter().any(|p| p.name == gone), "{}: {gone} is a role, not a param", a.name);
            }
            assert!(a.params.iter().all(|p| p.host_env.is_none()), "{}: no env-resolved param", a.name);
        }
        for a in &m.actions[..3] {
            assert!(!a.streaming, "{}: one-shot (the wav is the single artifact)", a.name);
            assert!(a.params.iter().any(|p| p.name == "text" && p.required), "{}: text must be required", a.name);
            assert_eq!(a.params.iter().find(|p| p.name == "max_frames").unwrap().default, Some(json!(256)));
            assert_eq!(a.outputs[0].media, Media::Audio);
        }
        // `batch` carries its texts inside `requests`, so it must NOT declare a
        // top-level `text` at all - a caller that sets one would otherwise be
        // silently ignored rather than told it is an unknown param.
        let batch = &m.actions[3];
        assert!(!batch.params.iter().any(|p| p.name == "text"), "batch must not declare a scalar 'text'");
        assert!(batch.params.iter().any(|p| p.name == "requests" && p.required), "batch: 'requests' must be required");
        assert_eq!(m.to_json()["actions"][0]["name"], "synth");

        let clone = &m.actions[1];
        assert!(clone.inputs.iter().any(|i| i.name == "ref" && i.required), "clone: the reference wav is a required input");
        assert!(clone.inputs.iter().any(|i| i.name == "ref_codes" && !i.required));
        assert!(clone.params.iter().any(|p| p.name == "ref_text" && !p.required));
        let design = &m.actions[2];
        assert!(design.params.iter().any(|p| p.name == "instruct" && !p.required));
        assert!(design.params.iter().any(|p| p.name == "speaker" && !p.required));
    }

    /// The model's lifecycle actions: import runs with no checkpoint resolved
    /// (it writes one), fine-tuning is a training stream, `sim` takes two
    /// utterances, and a fine-tune carries every knob of its CLI verb.
    #[test]
    fn import_finetune_and_sim_carry_every_knob_of_their_verbs() {
        let m = manifest();
        let get = |n: &str| m.actions.iter().find(|a| a.name == n).unwrap();
        assert!(get("import").from_scratch, "import writes the checkpoint, so it needs none");
        for p in ["source", "codec_source", "speaker_source", "dest"] {
            assert!(get("import").params.iter().any(|q| q.name == p && q.host_resolved), "import: {p}");
        }
        let ft = get("finetune");
        assert!(ft.training && ft.streaming && !ft.from_scratch);
        for p in ["data", "save", "full", "steps", "lr", "rank", "alpha", "batch", "block", "seed", "activations"] {
            assert!(ft.params.iter().any(|q| q.name == p), "finetune: {p}");
        }
        for p in ["data", "save"] {
            assert!(ft.params.iter().any(|q| q.name == p && q.required && q.host_resolved), "finetune: {p} is a required host path");
        }
        let inputs: Vec<&str> = get("sim").inputs.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(inputs, ["a", "b"]);
    }

    /// The sampling knobs declare no default - the checkpoint owns them - yet
    /// every served number carries a ceiling, seeds excepted.
    #[test]
    fn sampling_knobs_have_no_default_and_every_number_but_a_seed_has_a_ceiling() {
        let synth = synth_spec();
        let inv = synth.validate(Invocation::new().set("text", json!("hi"))).unwrap();
        for knob in ["temp", "top_k", "top_p", "repetition_penalty", "residual_temp", "residual_top_k", "residual_top_p"] {
            assert_eq!(inv.params.get(knob), None, "'{knob}' must not be defaulted into the invocation");
            assert_eq!(synth.params.iter().find(|p| p.name == knob).unwrap().default, None, "'{knob}' must declare no ParamSpec default - the checkpoint owns it");
        }
        // An explicit value still arrives untouched, even one equal to a reference default.
        let pinned = synth.validate(Invocation::new().set("text", json!("hi")).set("repetition_penalty", json!(1.0))).unwrap();
        assert_eq!(pinned.get_f64("repetition_penalty"), Some(1.0));
        assert!(synth.validate(Invocation::new()).is_err(), "text is required");
        for a in manifest().actions.iter().chain(resident_manifest().actions.iter()) {
            for p in a.params.iter().filter(|p| matches!(p.ty, ParamType::Int | ParamType::Float) && p.max.is_none()) {
                assert_eq!(p.name, "seed", "{}:{} is an unbounded number", a.name, p.name);
            }
        }
    }

    /// Missing weights must surface as a clean `ActionResult` error, not a
    /// panic in a loader, on EVERY action that reads the checkpoint.
    #[test]
    fn a_missing_checkpoint_is_a_clean_error_on_every_action() {
        let reg = registry(Some(missing_paths()));
        let base = || Invocation::new().set("text", json!("hello"));
        for action in ["synth", "design"] {
            let err = reg.run(MODEL, action, base(), &mut |_| {}).unwrap_err();
            assert!(err.contains("not found"), "{action} got: {err}");
        }
        let err = reg.run(MODEL, "clone", base().blob("ref", wav_blob(&[0.0; 8])), &mut |_| {}).unwrap_err();
        assert!(err.contains("not found"), "clone got: {err}");
        let batch = Invocation::new().set("requests", json!(r#"[{"text":"hello"}]"#));
        let err = reg.run(MODEL, "batch", batch, &mut |_| {}).unwrap_err();
        assert!(err.contains("not found"), "batch got: {err}");
        let sim = Invocation::new().blob("a", wav_blob(&[0.0; 8])).blob("b", wav_blob(&[0.0; 8]));
        assert!(reg.run(MODEL, "sim", sim, &mut |_| {}).unwrap_err().contains("not found"));
        let ft = Invocation::new().set("data", json!("/nonexistent/data")).set("save", json!("/nonexistent/out.safetensors"));
        assert!(reg.run(MODEL, "finetune", ft, &mut |_| {}).unwrap_err().contains("not found"));
    }

    /// With no checkpoint resolved at all, the actions that need one say how to
    /// name it rather than complaining about an internal structure.
    #[test]
    fn a_provider_with_no_checkpoint_asks_for_one() {
        let reg = registry(None);
        let err = reg.run(MODEL, "synth", Invocation::new().set("text", json!("hi")), &mut |_| {}).unwrap_err();
        assert!(err.contains("--weights-dir"), "got: {err}");
    }

    /// `batch`'s request array is parsed and validated BEFORE anything is
    /// loaded, so a malformed or empty batch is a typed error rather than a
    /// long wait ending in a loader panic.
    #[test]
    fn batch_validates_its_request_array_up_front() {
        let reg = registry(Some(missing_paths()));
        let run = |requests: &str| reg.run(MODEL, "batch", Invocation::new().set("requests", json!(requests)), &mut |_| {}).unwrap_err();
        assert!(run("not json").contains("JSON array"), "malformed JSON must say so");
        assert!(run(r#"[{"lang":"english"}]"#).contains("non-empty 'text'"), "a request without text must be rejected");
        assert!(run("[]").contains("at least one"), "an empty batch must be rejected");
    }

    #[test]
    fn clone_refuses_a_missing_or_undecodable_reference_and_malformed_codes() {
        let dir = std::env::temp_dir().join(format!("qwen3tts-caps-ref-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["talker.safetensors", "mtp.safetensors", "codec.safetensors", "speaker.safetensors"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        let reg = registry(Some(TtsPaths::new(&dir, "c")));
        let inv = || Invocation::new().set("text", json!("hi"));
        assert!(reg.run(MODEL, "clone", inv(), &mut |_| {}).unwrap_err().contains("'ref'"));
        let err = reg.run(MODEL, "clone", inv().blob("ref", Blob::new(Media::Bytes, b"not a wav".to_vec())), &mut |_| {}).unwrap_err();
        assert!(err.contains("reference audio"), "got: {err}");
        let err = reg.run(MODEL, "clone", inv().blob("ref", wav_blob(&[0.0; 8])).blob("ref_codes", Blob::new(Media::Bytes, vec![1, 2])), &mut |_| {}).unwrap_err();
        assert!(err.contains("ref_codes"), "got: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `clone` needs a `speaker.safetensors` (the ECAPA x-vector encoder);
    /// `synth`/`design` don't, matching CustomVoice/VoiceDesign checkpoints
    /// that ship no speaker encoder at all. Uses a weights dir with
    /// talker/mtp/codec present but speaker.safetensors absent, so the two
    /// `need_speaker` values are only distinguishable by that one file.
    #[test]
    fn only_clone_and_sim_require_the_speaker_encoder() {
        let dir = std::env::temp_dir().join(format!("qwen3tts-caps-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["talker.safetensors", "mtp.safetensors", "codec.safetensors"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        let paths = Some(TtsPaths::new(&dir, "c"));
        assert!(checkpoint(&paths, "synth", false).is_ok(), "synth must not require speaker.safetensors");
        for action in ["clone", "sim"] {
            match checkpoint(&paths, action, true) {
                Err(e) => assert!(e.contains("speaker.safetensors"), "{action} got: {e}"),
                Ok(_) => panic!("{action} must require speaker.safetensors"),
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn import_of_a_missing_release_is_a_clean_error_and_needs_no_checkpoint() {
        let dest = std::env::temp_dir().join(format!("qwen3tts-caps-import-{}", std::process::id()));
        let inv = Invocation::new().set("source", json!("/nonexistent/release")).set("dest", json!(dest.to_str().unwrap()));
        let err = registry(None).run(MODEL, "import", inv, &mut |_| {}).unwrap_err();
        assert!(err.contains("tts import") && err.contains("talker"), "got: {err}");
        std::fs::remove_dir_all(&dest).ok();
    }

    /// The resident surface streams the speech actions the hot engine runs,
    /// and a client discovers that from the manifest; the others are the
    /// provider's, unchanged.
    #[test]
    fn the_resident_manifest_streams_the_hot_actions_and_keeps_the_rest() {
        let (m, plain) = (resident_manifest(), manifest());
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["synth", "clone", "design", "batch", "import", "finetune", "sim"]);
        for a in &m.actions {
            let hot = RESIDENT_ACTIONS.contains(&a.name.as_str());
            let chunk = a.params.iter().find(|p| p.name == "stream_chunk");
            assert_eq!(a.streaming, hot || a.name == "batch" || a.name == "finetune", "{}: streaming", a.name);
            assert_eq!(chunk.is_some(), hot, "{}: only a hot action takes a chunk size", a.name);
            if hot {
                assert!(a.outputs.iter().any(|o| o.media == Media::Audio), "{}: must declare an audio output", a.name);
                assert!(chunk.unwrap().max.is_some(), "{}: chunk size is bounded", a.name);
            }
        }
        for name in ["batch", "import", "finetune", "sim"] {
            let (r, p) = (m.actions.iter().find(|a| a.name == name).unwrap(), plain.actions.iter().find(|a| a.name == name).unwrap());
            assert_eq!(r.to_json(), p.to_json(), "{name} is the provider's own spec");
        }
    }

    /// End-to-end: a real generation through the SAME `capability::Registry`
    /// path D-Bus/HTTP/`brain serve` use, against the checkpoint the model
    /// store resolves. Skips cleanly when the store holds none.
    #[test]
    fn design_action_produces_real_audio_through_the_registry() {
        let Some(paths) = crate::testsupport::store_paths() else {
            brain_testutil::skip("no Qwen3-TTS checkpoint in the model store");
            return;
        };
        let inv = Invocation::new().set("text", json!("Reached through the generic action surface now.")).set("max_frames", json!(24));
        let outcome = registry(Some(paths)).run(MODEL, "design", inv, &mut |_| {}).expect("design action should succeed");
        let samples = outcome.outputs["samples"].as_u64().expect("samples field");
        assert!(samples > 0, "design produced no audio");
        let audio = outcome.blobs.get("audio").expect("audio blob");
        assert!(!audio.bytes.is_empty(), "audio blob is empty");
        eprintln!("design-action e2e: {samples} samples, {} audio bytes", audio.bytes.len());
    }
}
