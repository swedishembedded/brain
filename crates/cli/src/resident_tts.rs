// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting brain's Qwen3-TTS voice synthesis behind the
//! residency [`Executor`], mirroring the yolo adapter in [`crate::resident`].
//!
//! The Talker + MTP decode runs CPU-side in brain (the bit-exact `CpuTalker` /
//! `CpuMtp` KV-cache path that [`tts::pipeline::synth`]/[`tts::pipeline::clone`]
//! use by default), so the instance's Hot footprint is RAM, not VRAM, and
//! `activate` ignores the assigned `device`. One action, `speak`.
//!
//! Config is env-only (no hardcoded paths), mirroring `brain tts synth`'s
//! `--weights-dir` / `--ckpt` flags:
//!   * `BRAIN_TTS_WEIGHTS` — dir holding `talker.weights`, `mtp.weights`,
//!     `codec.weights`, `speaker.weights` (the primary gate; unset ⇒ not served).
//!   * `BRAIN_TTS_CKPT`     — HF checkpoint dir (for `config.json` + tokenizer).
//!   * `BRAIN_TTS_LANG`     — default synthesis language (default `english`).
//!   * `BRAIN_TTS_REF`      — optional reference `.wav`: when set, `speak` voice
//!     -clones this timbre instead of the speaker-free synth.
//!   * `BRAIN_TTS_REF_TEXT` — optional transcript of `BRAIN_TTS_REF`; enables the
//!     in-context (ICL) clone path (the reference wav is codec-encoded in-tree).

use capability::{
    ActionResult, ActionSpec, Blob, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType, Progress,
};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;
use tts::{GenOpts, TtsPaths};

/// Qwen3-TTS 24 kHz output sample rate (see `tts::pipeline` / `brain tts synth`).
const SAMPLE_RATE: u32 = 24_000;

/// Text-to-speech behind the scheduler. Loads brain-format Qwen3-TTS checkpoints
/// (`BRAIN_TTS_WEIGHTS`); the resident instance decodes on the CPU (brain's
/// `CpuTalker` default) — dropping it frees the RAM. One action, `speak`.
pub struct TtsResident {
    weights_dir: String,
    ckpt_dir: String,
    lang: String,
    ref_wav: Option<String>,
    ref_text: String,
}

impl TtsResident {
    /// Configure from the environment, mirroring `brain tts synth`'s flags. Returns
    /// `None` (not served) when `BRAIN_TTS_WEIGHTS` is unset/empty, like
    /// [`crate::resident::YoloResident::from_env`].
    pub fn from_env() -> Option<TtsResident> {
        let weights_dir = std::env::var("BRAIN_TTS_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let ckpt_dir = std::env::var("BRAIN_TTS_CKPT").ok().unwrap_or_default();
        let lang = std::env::var("BRAIN_TTS_LANG").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "english".to_string());
        let ref_wav = std::env::var("BRAIN_TTS_REF").ok().filter(|p| !p.is_empty());
        let ref_text = std::env::var("BRAIN_TTS_REF_TEXT").ok().unwrap_or_default();
        Some(TtsResident { weights_dir, ckpt_dir, lang, ref_wav, ref_text })
    }

    /// Brain checkpoint paths (same layout as `brain tts`'s `paths()` helper).
    fn paths(&self) -> TtsPaths {
        TtsPaths {
            talker: format!("{}/talker.weights", self.weights_dir),
            mtp: format!("{}/mtp.weights", self.weights_dir),
            codec: format!("{}/codec.weights", self.weights_dir),
            speaker: format!("{}/speaker.weights", self.weights_dir),
            ckpt_dir: self.ckpt_dir.clone(),
        }
    }

    fn speak_spec() -> ActionSpec {
        ActionSpec::new("speak", "synthesize speech from text (Qwen3-TTS, 24 kHz f32 PCM)")
            .param(ParamSpec::new("text", ParamType::Str, "the text to speak").required())
            .param(ParamSpec::new("lang", ParamType::Str, "synthesis language").default(json!("english")))
            .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature").default(json!(0.9)))
            .param(ParamSpec::new("top_k", ParamType::Int, "top-k sampling cutoff").default(json!(50)))
            .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (reproducible run)").default(json!(0)))
            .param(ParamSpec::new("max_frames", ParamType::Int, "max codec frames (length cap)").default(json!(256)))
    }
}

impl ResidentModel for TtsResident {
    fn manifest(&self) -> Manifest {
        Manifest::new("tts", "text-to-speech (Qwen3-TTS Talker + MTP + codec)", vec![Self::speak_spec()])
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new("tts", "default")
    }

    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // CPU-side decode (CpuTalker/CpuMtp) → the footprint is RAM. Budget ≈ 1.3×
        // the sum of the weight files it loads (talker + mtp + codec + speaker);
        // fall back to a conservative 4 GiB if the files aren't stat-able yet.
        let p = self.paths();
        let sum: u64 = [&p.talker, &p.mtp, &p.codec, &p.speaker]
            .iter()
            .filter_map(|f| std::fs::metadata(f).ok().map(|m| m.len()))
            .sum();
        let ram = if sum > 0 { sum + sum / 3 } else { 4u64 << 30 };
        MemCost::new(0, ram)
    }

    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        // The public TTS entry points (`pipeline::synth`/`clone`) load the weights
        // internally per call (the load-once seam, `serve::TtsEngine`, is
        // OpenVINO/NPU-only and the CPU decode helpers are `pub(crate)`), so the
        // resident owns the resolved config and fails fast here if the primary
        // Talker checkpoint is missing.
        let paths = self.paths();
        if !std::path::Path::new(&paths.talker).exists() {
            return Err(format!("tts: talker weights not found at {} (set BRAIN_TTS_WEIGHTS)", paths.talker));
        }
        Ok(Box::new(TtsInstance {
            paths,
            lang: self.lang.clone(),
            ref_wav: self.ref_wav.clone(),
            ref_text: self.ref_text.clone(),
        }))
    }
}

struct TtsInstance {
    paths: TtsPaths,
    lang: String,
    ref_wav: Option<String>,
    ref_text: String,
}

impl Instance for TtsInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let text = inv.get_str("text").ok_or("tts speak: missing required param 'text'")?;
        if text.trim().is_empty() {
            return Err("tts speak: 'text' must be non-empty".to_string());
        }
        let lang = inv.get_str("lang").unwrap_or_else(|| self.lang.clone());

        // Generation knobs → GenOpts (defaults from the reference sampling recipe).
        let mut opts = GenOpts::default();
        if let Some(t) = inv.get_f64("temp") {
            opts.temperature = t as f32;
        }
        if let Some(k) = inv.get_i64("top_k") {
            opts.top_k = k.max(0) as usize;
        }
        if let Some(s) = inv.get_i64("seed") {
            opts.seed = s.max(0) as u64;
        }
        if let Some(f) = inv.get_i64("max_frames") {
            opts.max_frames = f.max(1) as usize;
        }

        // Speaker-free synth by default; voice-clone the configured reference when
        // `BRAIN_TTS_REF` is set (ICL when a transcript is also given).
        let wav = match &self.ref_wav {
            Some(refw) => tts::pipeline::clone(&self.paths, &opts, &text, refw, &self.ref_text, &lang, None)?,
            None => tts::pipeline::synth(&self.paths, &opts, &text, &lang)?,
        };

        // Emit raw little-endian f32 PCM (mono, 24 kHz), as the tts_serve protocol
        // streams it — no external WAV encoder dependency.
        let bytes: Vec<u8> = wav.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(Outcome::new()
            .set("samples", json!(wav.len()))
            .set("sample_rate", json!(SAMPLE_RATE))
            .blob(
                "audio",
                Blob::new(Media::Audio, bytes)
                    .with_meta(json!({"sample_rate": SAMPLE_RATE, "format": "pcm_f32", "channels": 1})),
            ))
    }
}
