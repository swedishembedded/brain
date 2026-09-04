// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident-model adapter putting brain's Qwen3-TTS voice synthesis behind the
//! residency [`Executor`], mirroring the yolo adapter in [`crate::resident`].
//!
//! The instance owns a hot [`qwen3tts::ResidentEngine`]: the Talker, MTP,
//! codec and tokenizer are loaded ONCE in `activate` and reused by every
//! subsequent call, and each call's waveform is streamed back chunk-by-chunk
//! as the codec decodes it (`capability::Progress::chunk`, which the D-Bus
//! `Subscribe` transport turns into real mid-stream `blob` frames). Those two
//! properties - resident weights and progressive audio - are exactly what the
//! private `brain tts serve` Unix-socket protocol existed to provide; they now
//! live on the standard capability surface, so a D-Bus/HTTP client gets them
//! with no TTS-specific protocol.
//!
//! Decode is CPU-side by construction (`CpuTalker`/`CpuMtp` + the pure-CPU
//! streaming codec back), so the instance's Hot footprint is RAM, not VRAM,
//! and `activate` ignores the assigned `device` - which is what makes
//! [`TtsResident::estimate`]'s zero-VRAM cost honest.
//!
//! Config is env-only (no hardcoded paths), mirroring `brain tts synth`'s
//! `--weights-dir` / `--ckpt` flags:
//!   * `BRAIN_QWEN3TTS_WEIGHTS` - dir holding `talker.safetensors`, `mtp.safetensors`,
//!     `codec.safetensors`, `speaker.safetensors` (the primary gate; unset ⇒ not served).
//!   * `BRAIN_QWEN3TTS_CKPT`     - HF checkpoint dir (for `config.json` + tokenizer).
//!   * `BRAIN_QWEN3TTS_LANG`     - default synthesis language (default `english`).
//!   * `BRAIN_QWEN3TTS_REF`      - optional reference `.wav`: when set, `speak` voice
//!     -clones this timbre instead of the speaker-free synth.
//!   * `BRAIN_QWEN3TTS_REF_TEXT` - optional transcript of `BRAIN_QWEN3TTS_REF`; enables the
//!     in-context (ICL) clone path (the reference wav is codec-encoded in-tree).

use capability::{
    ActionResult, Blob, Invocation, Manifest, Media, Outcome, Progress,
};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;
use qwen3tts::{GenOpts, TtsPaths};

/// Qwen3-TTS 24 kHz output sample rate (see `qwen3tts::pipeline` / `brain tts synth`).
const SAMPLE_RATE: u32 = 24_000;

/// Text-to-speech behind the scheduler. Loads brain-format Qwen3-TTS checkpoints
/// (`BRAIN_QWEN3TTS_WEIGHTS`); the resident instance decodes on the CPU (brain's
/// `CpuTalker` default) - dropping it frees the RAM. One action, `speak`.
pub struct TtsResident {
    weights_dir: String,
    ckpt_dir: String,
    lang: String,
    ref_wav: Option<String>,
    ref_text: String,
}

impl TtsResident {
    /// Configure from the environment, mirroring `brain tts synth`'s flags. Returns
    /// `None` (not served) when `BRAIN_QWEN3TTS_WEIGHTS` is unset/empty, like
    /// [`crate::resident::YoloResident::from_env`].
    pub fn from_env() -> Option<TtsResident> {
        let weights_dir = std::env::var("BRAIN_QWEN3TTS_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let ckpt_dir = std::env::var("BRAIN_QWEN3TTS_CKPT").ok().unwrap_or_default();
        let lang = std::env::var("BRAIN_QWEN3TTS_LANG").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "english".to_string());
        let ref_wav = std::env::var("BRAIN_QWEN3TTS_REF").ok().filter(|p| !p.is_empty());
        let ref_text = std::env::var("BRAIN_QWEN3TTS_REF_TEXT").ok().unwrap_or_default();
        Some(TtsResident { weights_dir, ckpt_dir, lang, ref_wav, ref_text })
    }

    /// Direct constructor for callers that already hold the two paths (e.g.
    /// `brain perf`'s `tts:<weights-dir>:<hf-ckpt-dir>` target) - no env
    /// round-trip; see `crate::resident_scrfd::ScrfdResident::new`.
    /// Default English, no reference voice - the synthesis-tuning knobs
    /// (`BRAIN_QWEN3TTS_LANG`/`_REF`/`_REF_TEXT`) remain `from_env`-only.
    pub fn new(weights_dir: impl Into<String>, ckpt_dir: impl Into<String>) -> TtsResident {
        TtsResident {
            weights_dir: weights_dir.into(),
            ckpt_dir: ckpt_dir.into(),
            lang: "english".to_string(),
            ref_wav: None,
            ref_text: String::new(),
        }
    }

    /// Brain checkpoint paths (same layout as `brain tts`'s `paths()` helper).
    fn paths(&self) -> TtsPaths {
        TtsPaths {
            talker: format!("{}/talker.safetensors", self.weights_dir),
            mtp: format!("{}/mtp.safetensors", self.weights_dir),
            codec: format!("{}/codec.safetensors", self.weights_dir),
            speaker: format!("{}/speaker.safetensors", self.weights_dir),
            ckpt_dir: self.ckpt_dir.clone(),
        }
    }

}

impl ResidentModel for TtsResident {
    fn manifest(&self) -> Manifest {
        // The spec lives in qwen3tts::caps, next to the catalog's `synth` spec,
        // so the two surfaces cannot silently diverge.
        qwen3tts::caps::resident_manifest()
    }

    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(qwen3tts::caps::MODEL, "default")
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
        // Activation IS the load: `ResidentEngine::load` reads the Talker, MTP,
        // codec and tokenizer once and the instance keeps them hot for every
        // subsequent action, so a served request never re-reads 3 GB of Talker
        // weights from disk. A missing/unreadable checkpoint fails here, at
        // activation, rather than on the first request.
        let paths = self.paths();
        if !std::path::Path::new(&paths.talker).exists() {
            return Err(format!("tts: talker weights not found at {} (set BRAIN_QWEN3TTS_WEIGHTS)", paths.talker));
        }
        Ok(Box::new(TtsInstance {
            engine: qwen3tts::ResidentEngine::load(&paths)?,
            lang: self.lang.clone(),
            ref_wav: self.ref_wav.clone(),
            ref_text: self.ref_text.clone(),
        }))
    }
}

struct TtsInstance {
    engine: qwen3tts::ResidentEngine,
    lang: String,
    ref_wav: Option<String>,
    ref_text: String,
}

impl TtsInstance {
    /// Generation knobs shared by every action → `GenOpts`.
    ///
    /// A sampling knob the invocation does not carry stays `None`: the
    /// invocation is only the "explicit caller override" layer, and the
    /// checkpoint's `generation_config.json` (then the reference's defaults)
    /// answers for the rest - see `qwen3tts::genconfig`.
    fn opts_from(inv: &Invocation) -> GenOpts {
        let mut opts = GenOpts::default();
        opts.sampling.temperature = inv.get_f64("temp").map(|t| t as f32);
        opts.sampling.top_k = inv.get_i64("top_k").map(|k| k.max(0) as usize);
        opts.sampling.top_p = inv.get_f64("top_p").map(|p| p as f32);
        opts.sampling.repetition_penalty = inv.get_f64("repetition_penalty").map(|r| r as f32);
        if let Some(s) = inv.get_i64("seed") {
            opts.seed = s.max(0) as u64;
        }
        if let Some(f) = inv.get_i64("max_frames") {
            opts.max_frames = f.max(1) as usize;
        }
        opts
    }

    /// Emit raw little-endian f32 PCM (mono, 24 kHz) - the same wire shape the
    /// mid-run chunks below carry, so a client can append chunks and get
    /// exactly this. No external WAV encoder dependency.
    fn pcm_outcome(wav: Vec<f32>) -> ActionResult {
        Ok(Outcome::new()
            .set("samples", json!(wav.len()))
            .set("sample_rate", json!(SAMPLE_RATE))
            .blob("audio", pcm_blob(&wav)))
    }
}

/// Raw mono f32 little-endian PCM as a capability [`Blob`] - one encoder for
/// both the mid-run chunks and the terminal artifact.
fn pcm_blob(pcm: &[f32]) -> Blob {
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    Blob::new(Media::Audio, bytes).with_meta(json!({"sample_rate": SAMPLE_RATE, "format": "pcm_f32", "channels": 1}))
}

impl Instance for TtsInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let text = inv.get_str("text").ok_or_else(|| format!("tts {action}: missing required param 'text'"))?;
        if text.trim().is_empty() {
            return Err(format!("tts {action}: 'text' must be non-empty"));
        }
        let lang = inv.get_str("lang").unwrap_or_else(|| self.lang.clone());
        let opts = Self::opts_from(inv);

        // Every action streams its audio the same way: each decoded codec
        // chunk becomes a `Progress::chunk` named "audio", which `Subscribe`
        // forwards as a real out-of-band blob frame. `index` counts chunks
        // from 1, matching the omni resident's convention. A plain `Run`
        // caller sees none of this and still gets the whole waveform below.
        let mut n = 0u32;
        let mut on_audio = |pcm: &[f32], _seq: u32| {
            n += 1;
            progress(Progress::chunk(n, 0, format!("audio chunk {n}"), "audio", pcm_blob(pcm).with_meta(json!({"sample_rate": SAMPLE_RATE, "format": "pcm_f32", "channels": 1, "index": n}))));
        };

        let wav = match action {
            // Speaker-free synth by default; voice-clone the configured reference
            // when `BRAIN_QWEN3TTS_REF` is set (ICL when a transcript is also given).
            "speak" => match self.ref_wav.clone() {
                Some(refw) => self.engine.clone_voice(&text, &refw, &self.ref_text.clone(), &lang, &opts, &inv.cancel, &mut on_audio)?,
                None => self.engine.speak(&text, &lang, &opts, &inv.cancel, &mut on_audio)?,
            },
            // VoiceDesign/CustomVoice: instruct/speaker are per-call params, not
            // fixed at instance configuration (unlike `speak`'s ref voice).
            "design" => {
                let instruct = inv.get_str("instruct").unwrap_or_default();
                let speaker = inv.get_str("speaker").filter(|s| !s.is_empty());
                self.engine.design(&text, &lang, &instruct, speaker.as_deref(), &opts, &inv.cancel, &mut on_audio)?
            }
            other => return Err(format!("tts: unsupported action '{other}' (this resident declares: speak, design)")),
        };
        Self::pcm_outcome(wav)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Activation IS the load now, so a missing checkpoint must surface as a
    /// clean activation error rather than a panic inside a loader or a
    /// first-request failure.
    #[test]
    fn activating_without_weights_is_a_clean_error() {
        let r = TtsResident::new("/nonexistent/tts", "/nonexistent/ckpt");
        let Err(err) = r.activate(&InstanceKey::new(qwen3tts::caps::MODEL, "default"), Device::Cpu) else {
            panic!("activating without weights must fail");
        };
        assert!(err.contains("not found"), "got: {err}");
    }

    /// The consolidation contract at the spec level: both served actions must
    /// declare themselves streaming, because they now emit real mid-run audio
    /// chunks. A client discovers that from the manifest - which is the whole
    /// difference between this and the private socket protocol, where the
    /// streaming shape was documented only in a Rust comment.
    #[test]
    fn the_resident_manifest_declares_streaming_actions() {
        let m = qwen3tts::caps::resident_manifest();
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["speak", "design"]);
        for a in &m.actions {
            assert!(a.streaming, "{}: must declare streaming - it emits Progress::chunk audio", a.name);
            assert!(a.outputs.iter().any(|o| o.media == Media::Audio), "{}: must declare an audio output", a.name);
        }
    }

    /// End-to-end against real weights: one activation, TWO requests, and the
    /// mid-run chunks must reassemble byte-for-byte into the terminal `audio`
    /// blob. This is the property that makes the D-Bus surface a genuine
    /// replacement for the socket server's `audio_chunk` stream - a subscriber
    /// that concatenates the chunks holds exactly the clip a `Run` caller gets.
    /// Skips cleanly without `BRAIN_QWEN3TTS_WEIGHTS`/`_CKPT`.
    #[test]
    fn speak_streams_chunks_that_reassemble_into_the_terminal_blob() {
        let (Ok(weights), Ok(ckpt)) = (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT")) else {
            brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT not set");
            return;
        };
        if !std::path::Path::new(&format!("{weights}/talker.safetensors")).exists() {
            brain_testutil::skip("weights not found at BRAIN_QWEN3TTS_WEIGHTS");
            return;
        }
        let r = TtsResident::new(weights, ckpt);
        let t_load = std::time::Instant::now();
        let mut inst = r.activate(&InstanceKey::new(qwen3tts::caps::MODEL, "default"), Device::Cpu).expect("activate");
        let load_s = t_load.elapsed().as_secs_f64();

        let inv = |t: &str| Invocation::new().set("text", json!(t)).set("max_frames", json!(24)).set("seed", json!(3));
        let mut streamed: Vec<u8> = Vec::new();
        let mut n_chunks = 0usize;
        let t0 = std::time::Instant::now();
        let out = inst
            .run("speak", &inv("Chunks reassemble into the clip."), &mut |p: Progress| {
                if let Some((name, blob)) = &p.chunk {
                    assert_eq!(name, "audio", "chunk blobs must be named 'audio'");
                    assert_eq!(blob.media, Media::Audio);
                    n_chunks += 1;
                    streamed.extend_from_slice(&blob.bytes);
                }
            })
            .expect("speak");
        let first_s = t0.elapsed().as_secs_f64();
        assert!(n_chunks >= 1, "speak emitted no audio chunks");
        assert_eq!(streamed, out.blobs["audio"].bytes, "streamed chunks must equal the terminal audio blob");

        // Same request again on the same hot instance: bit-identical, or
        // resident state (the Talker KV cache, the RNG) leaked between calls.
        let t1 = std::time::Instant::now();
        let out2 = inst.run("speak", &inv("Chunks reassemble into the clip."), &mut |_| {}).expect("second speak");
        let second_s = t1.elapsed().as_secs_f64();
        assert_eq!(out2.blobs["audio"].bytes, out.blobs["audio"].bytes, "a repeated request must be bit-identical - resident state leaked");
        eprintln!("resident tts: activate={load_s:.1}s first={first_s:.1}s repeat={second_s:.1}s ({n_chunks} chunks)");
    }

    /// Cheap guard that survives without weights: the dispatcher's own
    /// argument validation. Uses the raw `Invocation` path (no engine needed
    /// for the rejection, which happens before any generation).
    #[test]
    fn empty_text_is_rejected_before_any_generation() {
        let spec = qwen3tts::caps::speak_spec();
        let err = spec.validate(Invocation::new()).unwrap_err();
        assert!(err.contains("text"), "speak must require 'text': {err}");
    }
}
