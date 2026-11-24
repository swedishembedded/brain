// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Voice-synthesis **prompt assembly** for the Qwen3-TTS Talker.
//!
//! Reproduces the input-embedding prefix that `Qwen3TTSForConditionalGeneration.
//! generate` builds in `qwen3_tts_model.py` for the *Base* (voice-clone) model.
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
            tts_bos: gu(&v, "tts_bos_token_id", 151672),
            tts_eos: gu(&v, "tts_eos_token_id", 151673),
            tts_pad: gu(&v, "tts_pad_token_id", 151671),
            codec_nothink: gu(t, "codec_nothink_id", 2155),
            codec_think: gu(t, "codec_think_id", 2154),
            codec_think_bos: gu(t, "codec_think_bos_id", 2156),
            codec_think_eos: gu(t, "codec_think_eos_id", 2157),
            codec_pad: gu(t, "codec_pad_id", 2148),
            codec_bos: gu(t, "codec_bos_id", 2149),
            codec_eos: gu(t, "codec_eos_token_id", 2150),
            lang,
            spk_id,
            checkpoint: Checkpoint::from_config(&v)?,
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

/// Build a `tokenizer.json`-shaped value in memory from the checkpoint's
/// `vocab.json` + `merges.txt` + `tokenizer_config.json` (the Base model ships
/// these rather than a single `tokenizer.json`), then load [`QwenBpe`].
pub fn load_tokenizer(dir: &str) -> Result<data::qwen_tokenizer::QwenBpe, String> {
    let dir = std::path::Path::new(dir);
    let vocab: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("vocab.json")).map_err(|e| e.to_string())?)
            .map_err(|e| format!("vocab.json: {e}"))?;
    let merges_txt = std::fs::read_to_string(dir.join("merges.txt")).map_err(|e| e.to_string())?;
    let merges: Vec<serde_json::Value> = merges_txt
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with("#version"))
        .map(serde_json::Value::from)
        .collect();
    let tcfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("tokenizer_config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("tokenizer_config.json: {e}"))?;
    let mut added = Vec::new();
    if let Some(m) = tcfg["added_tokens_decoder"].as_object() {
        for (id, info) in m {
            if let (Ok(id), Some(content)) = (id.parse::<u64>(), info["content"].as_str()) {
                added.push(serde_json::json!({"content": content, "id": id}));
            }
        }
    }
    let tj = serde_json::json!({
        "model": {"vocab": vocab, "merges": merges},
        "added_tokens": added,
    });
    data::qwen_tokenizer::QwenBpe::from_json_bytes(tj.to_string().as_bytes())
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
fn proj(gen: &impl TalkerHost, ids: &[u32]) -> Vec<f32> {
    gen.text().project(&gen.text().embed_text(ids))
}

/// Elementwise `a += b` over `[d]` slices.
fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Build the codec prefix embedding (`codec_input_embedding` in the reference):
/// the think/language tag tokens, the optional speaker x-vector, and
/// `[codec_pad, codec_bos]`. Returns `[m, d]`.
fn codec_prefix(
    gen: &impl TalkerHost,
    sp: &TtsSpecials,
    language_id: Option<u32>,
    speaker: Option<&[f32]>,
) -> Vec<f32> {
    let d = gen.d();
    let tag: Vec<u32> = match language_id {
        Some(l) => vec![sp.codec_think, sp.codec_think_bos, l, sp.codec_think_eos],
        None => vec![sp.codec_nothink, sp.codec_think_bos, sp.codec_think_eos],
    };
    let mut out = Vec::new();
    for &id in &tag {
        out.extend_from_slice(gen.codec_embed(id));
    }
    if let Some(spk) = speaker {
        assert_eq!(spk.len(), d, "speaker x-vector must be d_model");
        out.extend_from_slice(spk);
    }
    out.extend_from_slice(gen.codec_embed(sp.codec_pad));
    out.extend_from_slice(gen.codec_embed(sp.codec_bos));
    out
}

/// Assemble the shared prefix up to (and excluding) the text/ICL body:
/// `[role(3) ; (pad×(m-2), tts_bos) + cie[0..m-1]]`. Returns `(pre, cie)` where
/// `cie` is the `[m, d]` codec prefix (so callers can reuse `cie[m-1]`).
fn build_pre(
    gen: &impl TalkerHost,
    sp: &TtsSpecials,
    role_ids: &[u32],
    language_id: Option<u32>,
    speaker: Option<&[f32]>,
    tts_bos: &[f32],
    tts_pad: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let d = gen.d();
    let cie = codec_prefix(gen, sp, language_id, speaker);
    let m = cie.len() / d;
    let role = proj(gen, &role_ids[..3]);
    let mut pre = role; // [3, d]
    // (tts_pad × (m-2), tts_bos) + cie[0..m-1]
    for r in 0..(m - 1) {
        let base = if r < m - 2 { tts_pad } else { tts_bos };
        let mut row = base.to_vec();
        add_into(&mut row, &cie[r * d..(r + 1) * d]);
        pre.extend_from_slice(&row);
    }
    (pre, cie)
}

/// **x-vector-only** (and speaker-free "synth") voice prompt — non-streaming.
///
/// `role_ids`/`text_ids` are the role header (`input_id[..3]`) and the target
/// text content (`input_id[3..len-5]`). `speaker` is the 1024-d x-vector, or
/// `None` for the speaker-free synth prompt.
pub fn build_xvector_prompt(
    gen: &impl TalkerHost,
    sp: &TtsSpecials,
    role_ids: &[u32],
    text_ids: &[u32],
    speaker: Option<&[f32]>,
    language_id: Option<u32>,
) -> Prompt {
    assert!(text_ids.len() >= 2, "target text too short");
    let d = gen.d();
    let tts3 = proj(gen, &[sp.tts_bos, sp.tts_eos, sp.tts_pad]);
    let tts_bos = tts3[0..d].to_vec();
    let tts_eos = tts3[d..2 * d].to_vec();
    let tts_pad = tts3[2 * d..3 * d].to_vec();

    let (mut embeds, cie) = build_pre(gen, sp, role_ids, language_id, speaker, &tts_bos, &tts_pad);
    let m = cie.len() / d;
    // tts_text_first_token: proj(text_ids[0]) + cie[m-1] (the codec_bos row).
    let first = proj(gen, &text_ids[..1]);
    let mut row = first;
    add_into(&mut row, &cie[(m - 1) * d..m * d]);
    embeds.extend_from_slice(&row);

    // trailing_text_hidden = [proj(text_ids[1..]) ; tts_eos]
    let mut trailing = proj(gen, &text_ids[1..]);
    trailing.extend_from_slice(&tts_eos);

    Prompt {
        embeds,
        trailing,
        tts_pad,
    }
}

/// **VoiceDesign / CustomVoice (instruct)** prompt — non-streaming. Built on top
/// of [`build_xvector_prompt`] by prepending the projected **instruct** text
/// (`instruct_ids` = the tokenized `<|im_start|>user\n{instruct}<|im_end|>\n`
/// turn) to the input-embedding prefix — exactly as `Qwen3TTSModel.generate`
/// prepends `text_projection(text_embedding(instruct_id))`.
///
/// For **CustomVoice**, pass the preset speaker's codec-token id (from
/// [`TtsSpecials::speaker_id`]) as `speaker_id` — it is looked up in the Talker's
/// codec embedding table and placed in the codec prefix's speaker slot (the same
/// slot the x-vector occupies). For **VoiceDesign**, pass `speaker_id = None`: the
/// model designs the voice purely from the instruction.
#[allow(clippy::too_many_arguments)]
pub fn build_instruct_prompt(
    gen: &impl TalkerHost,
    sp: &TtsSpecials,
    role_ids: &[u32],
    text_ids: &[u32],
    instruct_ids: &[u32],
    speaker_id: Option<u32>,
    language_id: Option<u32>,
) -> Prompt {
    let speaker_embed: Option<Vec<f32>> = speaker_id.map(|id| gen.codec_embed(id).to_vec());
    let mut base =
        build_xvector_prompt(gen, sp, role_ids, text_ids, speaker_embed.as_deref(), language_id);
    if !instruct_ids.is_empty() {
        let mut embeds = proj(gen, instruct_ids);
        embeds.extend_from_slice(&base.embeds);
        base.embeds = embeds;
    }
    base
}

/// **ICL** voice-clone prompt — non-streaming. Conditions on the reference
/// transcript (`ref_ids` = `ref_id[3..len-2]`) and the reference audio codec
/// codes (`ref_code`, `[T_ref*16]` row-major, codebooks 0..15 per frame). The
/// `mtp` provides the residual-codebook embedding tables.
#[allow(clippy::too_many_arguments)]
pub fn build_icl_prompt(
    gen: &impl TalkerHost,
    mtp: &impl MtpHost,
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
    let tts3 = proj(gen, &[sp.tts_bos, sp.tts_eos, sp.tts_pad]);
    let tts_bos = tts3[0..d].to_vec();
    let tts_eos = tts3[d..2 * d].to_vec();
    let tts_pad = tts3[2 * d..3 * d].to_vec();

    let (mut embeds, _cie) =
        build_pre(gen, sp, role_ids, language_id, Some(speaker), &tts_bos, &tts_pad);

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
