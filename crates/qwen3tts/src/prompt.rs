// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Voice-synthesis **prompt assembly** for the Qwen3-TTS Talker.
//!
//! Reproduces the input-embedding prefix that `Qwen3TTSForConditionalGeneration.
//! generate` builds in `qwen3_tts_model.py` for the *Base* (voice-clone) model.
//! Two conditioning modes are supported, matching the reference:
//!
//!   * **x-vector-only** ([`build_xvector_prompt`]) — the speaker timbre comes
//!     from a 1024-d x-vector (from [`speaker`]); the target text drives content.
//!     The whole path is pure brain (no reference audio codes required), so this
//!     is the one the end-to-end CLI demo runs.  Passing `speaker = None` yields
//!     the speaker-free "synth" prompt.
//!   * **ICL** ([`build_icl_prompt`]) — in-context cloning that additionally
//!     conditions on the reference transcript *and the reference audio's codec
//!     codes* (`ref_code`, `[T_ref, 16]`).  brain's codec is decode-only, so the
//!     `ref_code` must be supplied externally (there is no encoder in-tree); the
//!     prompt assembly itself is fully implemented here.
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
        })
    }

    /// Preset speaker codec-token id for a CustomVoice speaker name (case-insensitive).
    pub fn speaker_id(&self, name: &str) -> Option<u32> {
        self.spk_id.get(&name.to_lowercase()).copied()
    }

    /// Codec language id for a language name (case-insensitive). `"auto"` (or
    /// unknown) yields `None`, selecting the no-language ("nothink") prefix.
    pub fn language_id(&self, name: &str) -> Option<u32> {
        let n = name.to_lowercase();
        if n == "auto" {
            return None;
        }
        self.lang.get(&n).copied()
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
