// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Voice-synthesis **prompt assembly** for the Qwen3-TTS Talker.
//!
//! Reproduces the input-embedding prefix that `Qwen3TTSForConditionalGeneration.
//! generate` builds in `qwen3_tts_model.py` for the *Base* (voice-clone) model.
//! Every prompt but the in-context clone is first planned as data
//! ([`PromptPlan`]: per prefix row, the codec rows, text tokens and x-vector it
//! sums) and then materialised from the tables, so inference and training read
//! one description of the rows. Two text layouts exist ([`TextLayout`]), as in
//! the reference's `non_streaming_mode`.
//!
//! Two conditioning modes are supported, matching the reference:
//!
//!   * **x-vector-only** ([`build_xvector_prompt`]) — the speaker timbre comes
//!     from a 1024-d x-vector (from `ecapatdnn`); the target text drives content.
//!     The whole path is pure brain (no reference audio codes required), so this
//!     is the one the end-to-end CLI demo runs.  Passing `speaker = None` yields
//!     the speaker-free "synth" prompt.
//!   * **ICL** ([`build_icl_prompt`]) — in-context cloning that additionally
//!     conditions on the reference transcript *and the reference audio's codec
//!     codes* (`ref_code`, `[T_ref, 16]`). brain's codec has a real in-tree
//!     encoder (`mimi::Codec::encode`, exact vs the reference tokenizer): when
//!     a caller passes `--ref-text` without `--ref-codes`,
//!     [`crate::pipeline::clone`] encodes the reference wav itself to produce
//!     `ref_code`; an explicit `--ref-codes` file still overrides that.
//!
//! ### Text-conditioning path (resolved)
//! The text embedding table lives **inside the Talker**:
//! `talker.model.text_embedding` is an `nn.Embedding[text_vocab=151936,
//! text_hidden=2048]` indexed by the raw Qwen text-token id.  Each text token is
//! looked up there and passed through `talker.text_projection` (a 2-layer
//! `fc2(silu(fc1(x)))` MLP, 2048→2048→1024) to the Talker `d_model`, then summed
//! onto the codec-token embedding stream.  Both tensors are imported alongside the
//! decoder (`text_embedding.weight`, `text_projection.*`) and exposed through
//! [`crate::talker::TextProjection`] / [`crate::gen::TalkerGen`].

use std::collections::HashMap;

use crate::talker::TextProjection;

/// CPU-table view a prompt builder needs from a Talker: `d_model`, the text
/// projection front-end, and the codebook-0 input-embedding table. Implemented by
/// the full [`crate::gen::TalkerGen`] (GPU/CPU engine) and by the lightweight
/// [`crate::npu_gen::TalkerTables`] (NPU host path, no `gpu_core` decoder), so the
/// same assembly serves every backend.
pub trait TalkerHost {
    fn d(&self) -> usize;
    fn text(&self) -> &TextProjection;
    fn codec_embed(&self, id: u32) -> &[f32];
    /// `text_projection(text_embedding(ids))`, `[ids.len(), d]`: every row on
    /// its own, so a host that caches projected tokens (the text path is
    /// frozen in training) returns the same bits.
    fn project_text(&self, ids: &[u32]) -> Vec<f32> {
        self.text().project(&self.text().embed_text(ids))
    }
}

/// CPU-table view a prompt builder needs from an MTP: the residual-codebook
/// input-embedding tables. Implemented by [`crate::mtp::MtpModel`] and the
/// CPU-cached [`crate::gen_kv_mtp::CpuMtp`].
pub trait MtpHost {
    fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32];
}

/// Special token ids for the prompt, read from the checkpoint `config.json`.
#[derive(Clone, Debug)]
pub struct TtsSpecials {
    pub tts_bos: u32,
    pub tts_eos: u32,
    pub tts_pad: u32,
    pub codec_nothink: u32,
    pub codec_think: u32,
    pub codec_think_bos: u32,
    pub codec_think_eos: u32,
    pub codec_pad: u32,
    pub codec_bos: u32,
    pub codec_eos: u32,
    pub lang: HashMap<String, u32>,
    /// Preset speaker name -> codec-token id (CustomVoice `talker_config.spk_id`).
    /// Empty for Base / VoiceDesign.
    pub spk_id: HashMap<String, u32>,
    /// Which released checkpoint this is (`tts_model_type`, `tts_model_size`):
    /// it decides what a request may steer ([`Self::check_steering`]). `None`
    /// for a config that does not say, which is held to nothing.
    pub checkpoint: Option<Checkpoint>,
}

/// A released Qwen3-TTS checkpoint's kind: each was trained for one way of
/// choosing the voice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointKind {
    /// Clones a reference recording (it ships the speaker encoder).
    Base,
    /// Speaks as one of its preset speakers, the larger size also following an
    /// instruction.
    CustomVoice,
    /// Speaks in a voice an instruction describes.
    VoiceDesign,
}

/// The checkpoint's kind and size, as its `config.json` states them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub kind: CheckpointKind,
    /// `tts_model_size` (`0b6`, `1b7`).
    pub size: String,
}

/// How a request steers the voice, for [`TtsSpecials::check_steering`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Steering {
    /// A reference recording to imitate (`clone`).
    Reference,
    /// An instruction and/or a preset speaker (`design`).
    Design { instruct: bool, speaker: bool },
}

/// The smallest released size, whose CustomVoice checkpoint was not trained to
/// follow an instruction.
const SMALL_SIZE: &str = "0b6";

impl Checkpoint {
    fn from_config(v: &serde_json::Value) -> Result<Option<Checkpoint>, String> {
        let Some(kind) = v["tts_model_type"].as_str() else { return Ok(None) };
        let kind = match kind {
            "base" => CheckpointKind::Base,
            "custom_voice" => CheckpointKind::CustomVoice,
            "voice_design" => CheckpointKind::VoiceDesign,
            other => return Err(format!("config.json: unknown tts_model_type {other:?} (expected base, custom_voice or voice_design)")),
        };
        Ok(Some(Checkpoint { kind, size: v["tts_model_size"].as_str().unwrap_or_default().to_string() }))
    }
}

impl TtsSpecials {
    /// Parse from the HF `config.json` (top-level tts ids + `talker_config`).
    pub fn from_config_dir(dir: &str) -> Result<TtsSpecials, String> {
        let s = std::fs::read_to_string(std::path::Path::new(dir).join("config.json"))
            .map_err(|e| format!("read config.json: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&s).map_err(|e| format!("parse config.json: {e}"))?;
        Self::from_config(&v)
    }

    /// [`Self::from_config_dir`] on a `config.json` already parsed.
    pub fn from_config(v: &serde_json::Value) -> Result<TtsSpecials, String> {
        let t = &v["talker_config"];
        let gu = |o: &serde_json::Value, k: &str, d: u32| {
            o[k].as_u64().map(|x| x as u32).unwrap_or(d)
        };
        let mut lang = HashMap::new();
        if let Some(m) = t["codec_language_id"].as_object() {
            for (k, val) in m {
                if let Some(id) = val.as_u64() {
                    lang.insert(k.to_lowercase(), id as u32);
                }
            }
        }
        let mut spk_id = HashMap::new();
        if let Some(m) = t["spk_id"].as_object() {
            for (k, val) in m {
                if let Some(id) = val.as_u64() {
                    spk_id.insert(k.to_lowercase(), id as u32);
                }
            }
        }
        Ok(TtsSpecials {
            tts_bos: gu(v, "tts_bos_token_id", 151672),
            tts_eos: gu(v, "tts_eos_token_id", 151673),
            tts_pad: gu(v, "tts_pad_token_id", 151671),
            codec_nothink: gu(t, "codec_nothink_id", 2155),
            codec_think: gu(t, "codec_think_id", 2154),
            codec_think_bos: gu(t, "codec_think_bos_id", 2156),
            codec_think_eos: gu(t, "codec_think_eos_id", 2157),
            codec_pad: gu(t, "codec_pad_id", 2148),
            codec_bos: gu(t, "codec_bos_id", 2149),
            codec_eos: gu(t, "codec_eos_token_id", 2150),
            lang,
            spk_id,
            checkpoint: Checkpoint::from_config(v)?,
        })
    }

    /// Whether this checkpoint can steer the voice as `steering` asks: a
    /// reference only on Base, an instruction alone only on VoiceDesign, a
    /// preset speaker only on CustomVoice and an instruction beside it only on
    /// the larger CustomVoice (the reference accepts and then drops one on the
    /// 0.6B; brain refuses it instead of speaking as if it had been followed).
    /// The error is tagged [`capability::failure::FailureClass::UnsupportedCapability`]
    /// and names the checkpoint kind that would serve.
    pub fn check_steering(&self, steering: Steering) -> Result<(), String> {
        use capability::failure::{tag, FailureClass};
        let Some(ckpt) = &self.checkpoint else { return Ok(()) };
        let refuse = |why: &str| Err(tag(FailureClass::UnsupportedCapability, why));
        match (steering, &ckpt.kind) {
            (Steering::Reference, CheckpointKind::Base) => Ok(()),
            (Steering::Reference, _) => refuse("this checkpoint does not clone a recorded voice; a Base checkpoint does"),
            (Steering::Design { instruct: false, speaker: false }, _) => refuse("voice design needs an instruction or a preset speaker"),
            (Steering::Design { .. }, CheckpointKind::Base) => refuse("a Base checkpoint takes no instruction or preset speaker; a VoiceDesign checkpoint follows an instruction, a CustomVoice one has preset speakers"),
            (Steering::Design { speaker: true, .. }, CheckpointKind::VoiceDesign) => refuse("a VoiceDesign checkpoint has no preset speakers; a CustomVoice one does"),
            (Steering::Design { speaker: false, .. }, CheckpointKind::CustomVoice) => refuse("a CustomVoice checkpoint speaks as one of its preset speakers: name one"),
            (Steering::Design { instruct: true, .. }, CheckpointKind::CustomVoice) if ckpt.size == SMALL_SIZE => refuse("the 0.6B CustomVoice checkpoint does not follow an instruction; the 1.7B one does"),
            (Steering::Design { .. }, _) => Ok(()),
        }
    }

    /// Preset speaker codec-token id for a CustomVoice speaker name (case-insensitive).
    pub fn speaker_id(&self, name: &str) -> Option<u32> {
        self.spk_id.get(&name.to_lowercase()).copied()
    }

    /// Codec language id for a language name (case-insensitive). `"auto"`
    /// yields `None`, the no-language ("nothink") prefix, which lets the
    /// model infer the language from the text. A language the checkpoint
    /// does not name is refused with the ones it does, tagged
    /// `invalid-request`, rather than spoken as if `auto` had been asked.
    pub fn language_id(&self, name: &str) -> Result<Option<u32>, String> {
        let n = name.to_lowercase();
        if n == "auto" {
            return Ok(None);
        }
        self.lang.get(&n).copied().map(Some).ok_or_else(|| {
            let mut known: Vec<&str> = self.lang.keys().map(String::as_str).collect();
            known.sort_unstable();
            capability::failure::tag(
                capability::failure::FailureClass::InvalidRequest,
                format!("this checkpoint was not trained on language {name:?}; it names {known:?}, or \"auto\" to infer it from the text"),
            )
        })
    }
}

/// The checkpoint's tokenizer: [`QwenBpe::from_dir`], the one canonical Qwen
/// loader, so the split-file layout the Base model ships (`vocab.json` +
/// `merges.txt` + `tokenizer_config.json`) gets Qwen2's pre-tokenizer and its
/// special tokens exactly as every other Qwen model does. A parallel loader
/// that omitted the pre-tokenizer merged whitespace runs across word
/// boundaries and encoded 7.9% of the Sten-TTS training transcripts
/// differently from the reference.
pub fn load_tokenizer(dir: &str) -> Result<data::qwen_tokenizer::QwenBpe, String> {
    data::qwen_tokenizer::QwenBpe::from_dir(dir)
}

/// The assembled Talker generation prompt.
#[derive(Clone)]
pub struct Prompt {
    /// Prefix input embeddings, `[t_prefix, d_model]` row-major.
    pub embeds: Vec<f32>,
    /// Trailing per-frame text hidden states, `[t_trail, d_model]` (added to the
    /// codec feedback embedding during generation; `tts_pad` once exhausted).
    pub trailing: Vec<f32>,
    /// The `tts_pad` projected embedding (`[d_model]`).
    pub tts_pad: Vec<f32>,
}

/// `text_projection(text_embedding(ids))` -> `[ids.len(), d_model]`.
fn proj(gen: &(impl TalkerHost + ?Sized), ids: &[u32]) -> Vec<f32> {
    gen.project_text(ids)
}

/// Elementwise `a += b` over `[d]` slices.
fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// How the target text enters the Talker's input stream.
///
/// The reference generates in either layout (`non_streaming_mode`), and a
/// checkpoint speaks best in the one it was trained on: its CustomVoice and
/// VoiceDesign entry points default to [`TextLayout::NonStreaming`], voice
/// cloning to [`TextLayout::Streaming`] ([`TextLayout::default_for`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextLayout {
    /// The first text token rides on the `codec_bos` row; every later token,
    /// then `tts_eos`, is added to one generated frame's input row each, and
    /// `tts_pad` once they run out.
    Streaming,
    /// The whole text, then `tts_eos`, sits in the prefix over `codec_pad`
    /// rows, then `tts_pad + codec_bos`; every generated frame's row carries
    /// `tts_pad`.
    NonStreaming,
}

impl TextLayout {
    /// The names [`Self::parse`] takes, as a speech action's param lists them.
    pub const CHOICES: [&'static str; 2] = ["streaming", "non_streaming"];

    /// The layout named `name` (one of [`Self::CHOICES`]).
    pub fn parse(name: &str) -> Result<TextLayout, String> {
        match name {
            "streaming" => Ok(TextLayout::Streaming),
            "non_streaming" => Ok(TextLayout::NonStreaming),
            other => Err(capability::failure::tag(capability::failure::FailureClass::InvalidRequest, format!("text_layout {other:?}: expected one of {:?}", Self::CHOICES))),
        }
    }

    /// This layout's name in [`Self::CHOICES`].
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            TextLayout::Streaming => Self::CHOICES[0],
            TextLayout::NonStreaming => Self::CHOICES[1],
        }
    }

    /// The reference's default for a request steered as `steering`: preset
    /// speakers and designed voices speak non-streaming, a clone streaming.
    #[must_use]
    pub fn default_for(steering: Steering) -> TextLayout {
        match steering {
            Steering::Reference => TextLayout::Streaming,
            Steering::Design { .. } => TextLayout::NonStreaming,
        }
    }
}

/// One term of a prompt row: every prefix row is the sum of its parts, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// A text token through the frozen text path (`text_projection(text_embedding(id))`).
    Text(u32),
    /// A row of the Talker's `codec_embedding` (special, language and speaker
    /// rows; codebook-0 codes).
    Codec(u32),
    /// The request's speaker x-vector, in the speaker slot.
    XVector,
}

/// What fills the codec prefix's speaker slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeakerSlot {
    /// No speaker row at all (speaker-free synthesis, voice design).
    None,
    /// The request's x-vector (voice cloning).
    XVector,
    /// A named voice: its row of the Talker's `codec_embedding` (`spk_id`).
    Codec(u32),
}

/// Everything a prompt is planned from.
#[derive(Clone, Copy, Debug)]
pub struct PromptSpec<'a> {
    /// The role header (`input_id[..3]`).
    pub role_ids: &'a [u32],
    /// The target text (`input_id[3..len-5]`).
    pub text_ids: &'a [u32],
    /// The tokenised `<|im_start|>user\n{instruct}<|im_end|>\n` turn, or empty.
    pub instruct_ids: &'a [u32],
    pub speaker: SpeakerSlot,
    /// The language row, `None` for the no-language ("nothink") prefix.
    pub language_id: Option<u32>,
    pub layout: TextLayout,
}

/// A prompt as data: for every prefix row the parts it sums, and the text
/// token each generated frame's row adds (`tts_pad` once they run out).
///
/// Inference materialises it from the live tables ([`Self::materialise`]);
/// training materialises it from the tables it trains and routes each row's
/// gradient back to the codec rows its parts name, so the two always agree on
/// what a row is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptPlan {
    pub rows: Vec<Vec<Part>>,
    /// Text tokens added to generated frames `0, 1, ..`.
    pub trailing: Vec<u32>,
    /// The text token every later frame adds.
    pub tts_pad: u32,
}

impl PromptPlan {
    /// The prompt the reference's `generate` builds for `spec` (see the module
    /// documentation): `[instruct ; role ; codec prefix ; text body]`.
    pub fn build(sp: &TtsSpecials, spec: &PromptSpec<'_>) -> Result<PromptPlan, String> {
        if spec.role_ids.len() < 3 {
            return Err(format!("the role header holds {} tokens, expected 3", spec.role_ids.len()));
        }
        if spec.text_ids.is_empty() {
            return Err("the target text holds no tokens".to_string());
        }
        let mut rows = prefix_rows(sp, spec.role_ids, spec.instruct_ids, spec.speaker, spec.language_id);
        let trailing = match spec.layout {
            TextLayout::Streaming => {
                // tts_text_first_token rides on the codec_bos row; the rest trail.
                rows.push(vec![Part::Text(spec.text_ids[0]), Part::Codec(sp.codec_bos)]);
                spec.text_ids[1..].iter().copied().chain([sp.tts_eos]).collect()
            }
            TextLayout::NonStreaming => {
                for &t in spec.text_ids.iter().chain(&[sp.tts_eos]) {
                    rows.push(vec![Part::Text(t), Part::Codec(sp.codec_pad)]);
                }
                rows.push(vec![Part::Text(sp.tts_pad), Part::Codec(sp.codec_bos)]);
                vec![sp.tts_pad]
            }
        };
        Ok(PromptPlan { rows, trailing, tts_pad: sp.tts_pad })
    }

    /// Prefix rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the plan has no prefix row.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The text token frame `t`'s row adds.
    #[must_use]
    pub fn trailing_at(&self, t: usize) -> u32 {
        self.trailing.get(t).copied().unwrap_or(self.tts_pad)
    }

    /// The prompt over `gen`'s tables: each row the sum of its parts in
    /// order, `xvector` in the speaker slot of a plan that has one.
    ///
    /// # Panics
    /// The plan has an x-vector slot and `xvector` is not one `d_model`-wide row.
    pub fn materialise(&self, gen: &(impl TalkerHost + ?Sized), xvector: Option<&[f32]>) -> Prompt {
        let d = gen.d();
        let mut text: HashMap<u32, Vec<f32>> = HashMap::new();
        let mut project = |id: u32| text.entry(id).or_insert_with(|| proj(gen, &[id])).clone();
        let mut embeds = Vec::with_capacity(self.rows.len() * d);
        for parts in &self.rows {
            let mut row: Option<Vec<f32>> = None;
            for part in parts {
                let term = match *part {
                    Part::Text(id) => project(id),
                    Part::Codec(id) => gen.codec_embed(id).to_vec(),
                    Part::XVector => {
                        let x = xvector.expect("this prompt's speaker slot takes an x-vector");
                        assert_eq!(x.len(), d, "speaker x-vector must be d_model");
                        x.to_vec()
                    }
                };
                match &mut row {
                    None => row = Some(term),
                    Some(r) => add_into(r, &term),
                }
            }
            embeds.extend_from_slice(&row.expect("a prompt row has at least one part"));
        }
        let trailing = self.trailing.iter().flat_map(|&id| project(id)).collect();
        Prompt { embeds, trailing, tts_pad: project(self.tts_pad) }
    }
}

/// The rows every layout starts with: the instruction turn, the role header,
/// then `(tts_pad x (m-2), tts_bos) + codec_prefix[0..m-1]` where the codec
/// prefix (`codec_input_embedding` in the reference) is the think/language tag
/// tokens, the speaker slot, and `[codec_pad, codec_bos]`; its last row
/// (`codec_bos`) is the text body's to place.
fn prefix_rows(sp: &TtsSpecials, role_ids: &[u32], instruct_ids: &[u32], speaker: SpeakerSlot, language_id: Option<u32>) -> Vec<Vec<Part>> {
    let mut codec: Vec<Part> = match language_id {
        Some(l) => vec![sp.codec_think, sp.codec_think_bos, l, sp.codec_think_eos],
        None => vec![sp.codec_nothink, sp.codec_think_bos, sp.codec_think_eos],
    }
    .into_iter()
    .map(Part::Codec)
    .collect();
    match speaker {
        SpeakerSlot::None => {}
        SpeakerSlot::XVector => codec.push(Part::XVector),
        SpeakerSlot::Codec(id) => codec.push(Part::Codec(id)),
    }
    codec.push(Part::Codec(sp.codec_pad));
    let m = codec.len() + 1;
    let mut rows: Vec<Vec<Part>> = instruct_ids.iter().chain(&role_ids[..3]).map(|&t| vec![Part::Text(t)]).collect();
    for (r, c) in codec.into_iter().enumerate() {
        let base = if r < m - 2 { sp.tts_pad } else { sp.tts_bos };
        rows.push(vec![Part::Text(base), c]);
    }
    rows
}

/// [`PromptPlan::build`] then [`PromptPlan::materialise`], panicking on a
/// malformed request as the builders below always have.
fn planned(gen: &(impl TalkerHost + ?Sized), sp: &TtsSpecials, spec: &PromptSpec<'_>, xvector: Option<&[f32]>) -> Prompt {
    PromptPlan::build(sp, spec).unwrap_or_else(|e| panic!("{e}")).materialise(gen, xvector)
}

/// **x-vector-only** (and speaker-free "synth") voice prompt in the
/// streaming text layout.
///
/// `role_ids`/`text_ids` are the role header (`input_id[..3]`) and the target
/// text content (`input_id[3..len-5]`). `speaker` is the 1024-d x-vector, or
/// `None` for the speaker-free synth prompt.
pub fn build_xvector_prompt(
    gen: &(impl TalkerHost + ?Sized),
    sp: &TtsSpecials,
    role_ids: &[u32],
    text_ids: &[u32],
    speaker: Option<&[f32]>,
    language_id: Option<u32>,
) -> Prompt {
    build_xvector_prompt_in(gen, sp, role_ids, text_ids, speaker, language_id, TextLayout::Streaming)
}

/// [`build_xvector_prompt`] in the text layout `layout`.
pub fn build_xvector_prompt_in(
    gen: &(impl TalkerHost + ?Sized),
    sp: &TtsSpecials,
    role_ids: &[u32],
    text_ids: &[u32],
    speaker: Option<&[f32]>,
    language_id: Option<u32>,
    layout: TextLayout,
) -> Prompt {
    let slot = if speaker.is_some() { SpeakerSlot::XVector } else { SpeakerSlot::None };
    planned(gen, sp, &PromptSpec { role_ids, text_ids, instruct_ids: &[], speaker: slot, language_id, layout }, speaker)
}

/// **VoiceDesign / CustomVoice (instruct)** prompt in the streaming text
/// layout: [`build_xvector_prompt`] with the projected **instruct** text
/// (`instruct_ids` = the tokenized `<|im_start|>user\n{instruct}<|im_end|>\n`
/// turn) prepended - exactly as `Qwen3TTSModel.generate` prepends
/// `text_projection(text_embedding(instruct_id))`.
///
/// For **CustomVoice**, pass the preset speaker's codec-token id (from
/// [`TtsSpecials::speaker_id`]) as `speaker_id`: its row of the Talker's codec
/// embedding table takes the codec prefix's speaker slot (the same slot the
/// x-vector occupies). For **VoiceDesign**, pass `speaker_id = None`: the
/// model designs the voice purely from the instruction.
pub fn build_instruct_prompt(
    gen: &(impl TalkerHost + ?Sized),
    sp: &TtsSpecials,
    role_ids: &[u32],
    text_ids: &[u32],
    instruct_ids: &[u32],
    speaker_id: Option<u32>,
    language_id: Option<u32>,
) -> Prompt {
    build_instruct_prompt_in(gen, sp, role_ids, text_ids, instruct_ids, speaker_id, language_id, TextLayout::Streaming)
}

/// [`build_instruct_prompt`] in the text layout `layout`.
#[allow(clippy::too_many_arguments)]
pub fn build_instruct_prompt_in(
    gen: &(impl TalkerHost + ?Sized),
    sp: &TtsSpecials,
    role_ids: &[u32],
    text_ids: &[u32],
    instruct_ids: &[u32],
    speaker_id: Option<u32>,
    language_id: Option<u32>,
    layout: TextLayout,
) -> Prompt {
    let slot = speaker_id.map_or(SpeakerSlot::None, SpeakerSlot::Codec);
    planned(gen, sp, &PromptSpec { role_ids, text_ids, instruct_ids, speaker: slot, language_id, layout }, None)
}

/// **ICL** voice-clone prompt - streaming layout only. Conditions on the reference
/// transcript (`ref_ids` = `ref_id[3..len-2]`) and the reference audio codec
/// codes (`ref_code`, `[T_ref*16]` row-major, codebooks 0..15 per frame). The
/// `mtp` provides the residual-codebook embedding tables.
#[allow(clippy::too_many_arguments)]
pub fn build_icl_prompt(
    gen: &(impl TalkerHost + ?Sized),
    mtp: &(impl MtpHost + ?Sized),
    sp: &TtsSpecials,
    role_ids: &[u32],
    text_ids: &[u32],
    ref_ids: &[u32],
    ref_code: &[u32],
    speaker: &[f32],
    language_id: Option<u32>,
) -> Prompt {
    let d = gen.d();
    let ncode = 16usize;
    assert_eq!(ref_code.len() % ncode, 0, "ref_code must be [T,16]");
    let t_ref = ref_code.len() / ncode;
    let tts2 = proj(gen, &[sp.tts_eos, sp.tts_pad]);
    let tts_eos = tts2[0..d].to_vec();
    let tts_pad = tts2[d..2 * d].to_vec();

    // The shared prefix, up to (and excluding) the codec_bos row the ICL body places.
    let pre = PromptPlan { rows: prefix_rows(sp, role_ids, &[], SpeakerSlot::XVector, language_id), trailing: Vec::new(), tts_pad: sp.tts_pad };
    let mut embeds = pre.materialise(gen, Some(speaker)).embeds;

    // text_embed = proj([ref_ids ; text_ids]) ; append tts_eos.
    let mut concat = ref_ids.to_vec();
    concat.extend_from_slice(text_ids);
    let mut text_embed = proj(gen, &concat);
    text_embed.extend_from_slice(&tts_eos);
    let text_lens = text_embed.len() / d;

    // codec_embed: per ref frame, sum over the 16 codebooks; prepend codec_bos.
    let mut codec_embed = gen.codec_embed(sp.codec_bos).to_vec(); // [1, d]
    for f in 0..t_ref {
        let frame = &ref_code[f * ncode..(f + 1) * ncode];
        let mut acc = gen.codec_embed(frame[0]).to_vec();
        for (j, &code) in frame.iter().enumerate().skip(1) {
            add_into(&mut acc, mtp.codec_embed(j - 1, code));
        }
        codec_embed.extend_from_slice(&acc);
    }
    let codec_lens = codec_embed.len() / d;

    // Align text and codec; trailing = leftover text (or a single tts_pad).
    let (icl, trailing): (Vec<f32>, Vec<f32>) = if text_lens > codec_lens {
        let mut icl = text_embed[..codec_lens * d].to_vec();
        add_into(&mut icl, &codec_embed);
        (icl, text_embed[codec_lens * d..].to_vec())
    } else {
        let mut padded = text_embed;
        for _ in 0..(codec_lens - text_lens) {
            padded.extend_from_slice(&tts_pad);
        }
        add_into(&mut padded, &codec_embed);
        (padded, tts_pad.clone())
    };
    embeds.extend_from_slice(&icl);

    Prompt {
        embeds,
        trailing,
        tts_pad,
    }
}

#[cfg(test)]
mod steering_tests {
    use super::*;

    fn specials(kind: Option<(&str, &str)>) -> TtsSpecials {
        let mut v = serde_json::json!({"talker_config": {}});
        if let Some((kind, size)) = kind {
            v["tts_model_type"] = serde_json::json!(kind);
            v["tts_model_size"] = serde_json::json!(size);
        }
        TtsSpecials { checkpoint: Checkpoint::from_config(&v).unwrap(), ..crate::testsupport::tiny_specials() }
    }
    const fn design(instruct: bool, speaker: bool) -> Steering {
        Steering::Design { instruct, speaker }
    }

    /// The reference package's rule per checkpoint kind (`generate_voice_clone`,
    /// `generate_voice_design`, `generate_custom_voice`), with the 0.6B
    /// CustomVoice's instruction refused rather than dropped.
    #[test]
    fn each_checkpoint_kind_serves_only_the_steering_it_was_trained_for() {
        let base = specials(Some(("base", "0b6")));
        let design_ckpt = specials(Some(("voice_design", "1b7")));
        let small_cv = specials(Some(("custom_voice", "0b6")));
        let large_cv = specials(Some(("custom_voice", "1b7")));
        assert!(base.check_steering(Steering::Reference).is_ok());
        for s in [&design_ckpt, &small_cv, &large_cv] {
            assert!(s.check_steering(Steering::Reference).is_err());
        }
        assert!(base.check_steering(design(true, false)).is_err(), "Base would ignore the instruction");
        assert!(design_ckpt.check_steering(design(true, false)).is_ok());
        assert!(design_ckpt.check_steering(design(true, true)).is_err());
        assert!(small_cv.check_steering(design(false, true)).is_ok());
        assert!(small_cv.check_steering(design(true, true)).is_err(), "the 0.6B drops the instruction");
        assert!(large_cv.check_steering(design(true, true)).is_ok());
        assert!(large_cv.check_steering(design(true, false)).is_err(), "CustomVoice needs a speaker");
        assert!(design_ckpt.check_steering(design(false, false)).is_err());
        let err = base.check_steering(design(true, false)).unwrap_err();
        assert_eq!(capability::failure::classify(&err).map(|(c, _)| c), Some(capability::failure::FailureClass::UnsupportedCapability), "{err}");
    }

    #[test]
    fn a_language_the_checkpoint_does_not_name_is_refused_not_spoken_as_auto() {
        let mut sp = crate::testsupport::tiny_specials();
        sp.lang.insert("english".into(), 2050);
        assert_eq!(sp.language_id("English").unwrap(), Some(2050));
        assert_eq!(sp.language_id("auto").unwrap(), None);
        let err = sp.language_id("swedish").unwrap_err();
        assert!(err.contains("swedish") && err.contains("english") && err.contains("auto"), "{err}");
        assert_eq!(capability::failure::classify(&err).map(|(c, _)| c), Some(capability::failure::FailureClass::InvalidRequest));
    }

    #[test]
    fn an_unstated_kind_is_held_to_nothing_and_an_unknown_one_is_refused() {
        assert!(specials(None).check_steering(design(true, true)).is_ok());
        assert!(Checkpoint::from_config(&serde_json::json!({"tts_model_type": "duet"})).is_err());
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    /// A host with tiny random tables: `d = 4`, 16 codec rows, 12 text tokens.
    struct TinyHost {
        text: TextProjection,
        codec: Vec<f32>,
    }

    impl TinyHost {
        fn new() -> TinyHost {
            let mut rng = data::rng::Lcg::new(7); // seed: fixed fixture
            let (in_dim, inter, out, vocab) = (3, 5, 4, 12);
            let text = TextProjection {
                text_embedding: Some(rng.vec(vocab * in_dim)),
                fc1_w: rng.vec(inter * in_dim),
                fc1_b: rng.vec(inter),
                fc2_w: rng.vec(out * inter),
                fc2_b: rng.vec(out),
                in_dim,
                inter,
                out,
                text_vocab: vocab,
            };
            TinyHost { text, codec: rng.vec(16 * out) }
        }
    }

    impl TalkerHost for TinyHost {
        fn d(&self) -> usize {
            4
        }
        fn text(&self) -> &TextProjection {
            &self.text
        }
        fn codec_embed(&self, id: u32) -> &[f32] {
            &self.codec[id as usize * 4..(id as usize + 1) * 4]
        }
    }

    const ROLE: [u32; 3] = [9, 10, 11];
    const TEXT: [u32; 3] = [3, 4, 5];
    const LANG: u32 = 12;

    fn spec(speaker: SpeakerSlot, layout: TextLayout) -> PromptSpec<'static> {
        PromptSpec { role_ids: &ROLE, text_ids: &TEXT, instruct_ids: &[], speaker, language_id: Some(LANG), layout }
    }

    /// The rows the reference's `generate` builds: role, `tts_pad`s then
    /// `tts_bos` over the codec prefix (think tag, language, speaker, pad), then
    /// the text body of the layout.
    fn expected_prefix(sp: &TtsSpecials, speaker: Part) -> Vec<Vec<Part>> {
        use Part::{Codec, Text};
        let mut rows: Vec<Vec<Part>> = ROLE.iter().map(|&t| vec![Text(t)]).collect();
        for c in [Codec(sp.codec_think), Codec(sp.codec_think_bos), Codec(LANG), Codec(sp.codec_think_eos), speaker] {
            rows.push(vec![Text(sp.tts_pad), c]);
        }
        rows.push(vec![Text(sp.tts_bos), Codec(sp.codec_pad)]);
        rows
    }

    #[test]
    fn the_streaming_layout_puts_the_first_token_on_codec_bos_and_trails_the_rest() {
        use Part::{Codec, Text};
        let sp = crate::testsupport::tiny_specials();
        let plan = PromptPlan::build(&sp, &spec(SpeakerSlot::XVector, TextLayout::Streaming)).unwrap();
        let mut rows = expected_prefix(&sp, Part::XVector);
        rows.push(vec![Text(TEXT[0]), Codec(sp.codec_bos)]);
        assert_eq!(plan.rows, rows);
        assert_eq!(plan.trailing, vec![TEXT[1], TEXT[2], sp.tts_eos]);
        assert_eq!(plan.trailing_at(3), sp.tts_pad, "the text runs out into tts_pad");
    }

    #[test]
    fn the_non_streaming_layout_holds_the_whole_text_in_the_prefix_over_codec_pad() {
        use Part::{Codec, Text};
        let sp = crate::testsupport::tiny_specials();
        let plan = PromptPlan::build(&sp, &spec(SpeakerSlot::Codec(13), TextLayout::NonStreaming)).unwrap();
        let mut rows = expected_prefix(&sp, Codec(13));
        for t in TEXT.iter().chain(&[sp.tts_eos]) {
            rows.push(vec![Text(*t), Codec(sp.codec_pad)]);
        }
        rows.push(vec![Text(sp.tts_pad), Codec(sp.codec_bos)]);
        assert_eq!(plan.rows, rows);
        assert!((0..5).all(|t| plan.trailing_at(t) == sp.tts_pad), "every frame carries tts_pad");
    }

    #[test]
    fn an_instruction_leads_and_no_language_takes_the_nothink_tag() {
        use Part::{Codec, Text};
        let sp = crate::testsupport::tiny_specials();
        let s = PromptSpec { instruct_ids: &[6, 7], language_id: None, ..spec(SpeakerSlot::None, TextLayout::Streaming) };
        let plan = PromptPlan::build(&sp, &s).unwrap();
        assert_eq!(&plan.rows[..5], &[vec![Text(6)], vec![Text(7)], vec![Text(9)], vec![Text(10)], vec![Text(11)]]);
        assert_eq!(&plan.rows[5..9], &[vec![Text(sp.tts_pad), Codec(sp.codec_nothink)], vec![Text(sp.tts_pad), Codec(sp.codec_think_bos)], vec![Text(sp.tts_pad), Codec(sp.codec_think_eos)], vec![Text(sp.tts_bos), Codec(sp.codec_pad)]]);
        assert!(PromptPlan::build(&sp, &PromptSpec { text_ids: &[], ..s }).is_err());
    }

    /// Each materialised row is the sum of its parts' rows, the trailing text
    /// is projected token by token, and the x-vector fills its slot.
    #[test]
    fn a_plan_materialises_as_the_sum_of_its_parts() {
        let sp = crate::testsupport::tiny_specials();
        let host = TinyHost::new();
        let xvec = [0.5f32, -0.25, 1.0, 2.0];
        for layout in [TextLayout::Streaming, TextLayout::NonStreaming] {
            let plan = PromptPlan::build(&sp, &spec(SpeakerSlot::XVector, layout)).unwrap();
            let prompt = plan.materialise(&host, Some(&xvec));
            assert_eq!(prompt.embeds.len(), plan.len() * 4);
            for (r, parts) in plan.rows.iter().enumerate() {
                let mut want = vec![0.0f32; 4];
                for part in parts {
                    let term = match *part {
                        Part::Text(id) => host.project_text(&[id]),
                        Part::Codec(id) => host.codec_embed(id).to_vec(),
                        Part::XVector => xvec.to_vec(),
                    };
                    for (w, t) in want.iter_mut().zip(term) {
                        *w += t;
                    }
                }
                for (got, want) in prompt.embeds[r * 4..(r + 1) * 4].iter().zip(&want) {
                    assert!((got - want).abs() < 1e-6, "{layout:?} row {r}: {got} vs {want}");
                }
            }
            assert_eq!(prompt.trailing, host.project_text(&plan.trailing));
            assert_eq!(prompt.tts_pad, host.project_text(&[sp.tts_pad]));
        }
    }

    #[test]
    fn the_layout_defaults_follow_the_reference_and_names_round_trip() {
        assert_eq!(TextLayout::default_for(Steering::Reference), TextLayout::Streaming);
        assert_eq!(TextLayout::default_for(Steering::Design { instruct: true, speaker: false }), TextLayout::NonStreaming);
        for name in TextLayout::CHOICES {
            assert_eq!(TextLayout::parse(name).unwrap().name(), name);
        }
        assert!(TextLayout::parse("chunked").is_err());
    }
}
