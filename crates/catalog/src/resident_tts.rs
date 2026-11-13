// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS voice synthesis behind the residency [`Executor`].
//!
//! The speech actions (`synth`, `clone`, `design`) run on a hot
//! [`qwen3tts::ResidentEngine`]: the Talker, MTP, codec and tokenizer load ONCE
//! in `activate` and are reused by every subsequent call, and each call's
//! waveform is streamed back chunk by chunk as the codec decodes it
//! (`capability::Progress::chunk`, which the D-Bus `Subscribe` transport turns
//! into real mid-stream `blob` frames). A reference voice's conditioning is
//! computed once per recording and kept, so repeat clones of one timbre skip
//! the speaker encoder and the codec encode.
//!
//! The other actions (`batch`, `import`, `finetune`, `sim`) are the provider's
//! own ([`qwen3tts::caps::provider`]'s), run on the device the scheduler
//! assigned.
//!
//! Decode is CPU-side by construction (`CpuTalker`/`CpuMtp` + the pure-CPU
//! streaming codec back), so the hot instance's footprint is RAM, not VRAM,
//! and `activate` ignores the assigned `device` - which is what makes
//! [`TtsResident::estimate`]'s zero-VRAM cost honest.
//!
//! The checkpoint is the model store's (`qwen3tts::spec::Qwen3TtsSpec`), under
//! the same `weights_dir` and `ckpt` roles the command line names; nothing is
//! read from the environment.
//!
//! Swedish Embedded AB implements low-latency, load-once speech-synthesis
//! services for its clients. If your team needs expertise in serving neural
//! speech with the weights kept hot and audio streamed as it is made then you
//! can procure our services by sending an email to info@swedishembedded.com.

use capability::{ActionResult, Invocation, Manifest, Progress, Provider};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};
use serde_json::json;
use qwen3tts::TtsPaths;

/// Instance key of the hot engine: the speech actions.
const ENGINE_KEY: &str = "default";
/// Instance key of `finetune`, which trains the Talker on the device.
const TRAIN_KEY: &str = "train";
/// Instance key of the provider's host-side actions (`batch`, `import`, `sim`).
const HOST_KEY: &str = "host";

/// Text-to-speech behind the scheduler, over one Qwen3-TTS checkpoint.
pub struct TtsResident {
    paths: TtsPaths,
}

impl TtsResident {
    /// The checkpoint the model store resolves for Qwen3-TTS, or `None` (not
    /// served) when it holds none or several that cannot be told apart.
    pub fn from_store(models_dir: Option<&std::path::Path>) -> Option<TtsResident> {
        let assembly = loader::served::served_assembly(models_dir, "qwen3tts", &qwen3tts::spec::Qwen3TtsSpec, &[])?;
        Some(TtsResident { paths: TtsPaths::from_assembly(&assembly).ok()? })
    }

    /// Direct constructor for callers that already hold the two paths (e.g.
    /// `brain perf`'s `tts:<weights-dir>:<hf-ckpt-dir>` target).
    pub fn new(weights_dir: impl Into<String>, ckpt_dir: impl Into<String>) -> TtsResident {
        TtsResident { paths: TtsPaths::new(weights_dir.into(), ckpt_dir) }
    }

    /// Host RAM the f32 weights of every component occupy, with a third on top
    /// for the working set; a conservative 4 GiB when they cannot be read yet.
    fn ram(&self) -> u64 {
        let sum = self.paths.f32_weight_bytes();
        if sum > 0 { sum + sum / 3 } else { 4 << 30 }
    }
}

impl ResidentModel for TtsResident {
    fn manifest(&self) -> Manifest {
        // The spec lives in qwen3tts::caps, next to the provider's, so the two
        // surfaces cannot silently diverge.
        qwen3tts::caps::resident_manifest()
    }

    fn instance_key(&self, action: &str, _inv: &Invocation) -> InstanceKey {
        let config = match action {
            a if qwen3tts::caps::RESIDENT_ACTIONS.contains(&a) => ENGINE_KEY,
            "finetune" => TRAIN_KEY,
            _ => HOST_KEY,
        };
        InstanceKey::new(qwen3tts::caps::MODEL, config)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        match key.config.as_str() {
            // Fine-tuning holds the Talker's weights, their gradients and
            // AdamW's two moments: four copies, the dominant term at any batch.
            TRAIN_KEY => MemCost::new(self.paths.f32_weight_bytes().max(1 << 30) * 4, self.ram()),
            // `batch` gives each request its own host decoder copy.
            HOST_KEY => MemCost::new(0, self.ram() * 2),
            _ => MemCost::new(0, self.ram()),
        }
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if key.config != ENGINE_KEY {
            let provider = qwen3tts::caps::TtsProvider::new(Some(self.paths.clone()));
            return Ok(Box::new(ProviderInstance { provider, device }));
        }
        // Activation IS the load: `ResidentEngine::load` reads the Talker, MTP,
        // codec and tokenizer once and the instance keeps them hot for every
        // subsequent action, so a served request never re-reads 3 GB of Talker
        // weights from disk. A missing/unreadable checkpoint fails here, at
        // activation, rather than on the first request.
        self.paths.require(false).map_err(|e| format!("tts: {e} (pull a Qwen3-TTS checkpoint, or name it with --weights-dir and --ckpt)"))?;
        Ok(Box::new(TtsInstance { engine: qwen3tts::ResidentEngine::load(&self.paths)? }))
    }
}

/// The provider's own actions, run on the device the scheduler assigned.
struct ProviderInstance {
    provider: qwen3tts::caps::TtsProvider,
    device: Device,
}

impl Instance for ProviderInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let act = self.provider.action(action).ok_or_else(|| format!("tts: unknown action '{action}'"))?;
        crate::resident_llm::on_device(self.device, || act.run(inv, progress))?
    }
}

struct TtsInstance {
    engine: qwen3tts::ResidentEngine,
}

/// Raw mono f32 little-endian PCM as a capability [`Blob`]: what a mid-run
/// chunk carries (the terminal artifact is the whole clip as a WAV).
fn pcm_blob(pcm: &[f32], index: u32) -> capability::Blob {
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    capability::Blob::new(capability::Media::Audio, bytes).with_meta(json!({"sample_rate": qwen3tts::engine::SAMPLE_RATE, "format": "pcm_f32", "channels": 1, "index": index}))
}

impl Instance for TtsInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let text = inv.get_str("text").ok_or_else(|| format!("tts {action}: missing required param 'text'"))?;
        if text.trim().is_empty() {
            return Err(format!("tts {action}: 'text' must be non-empty"));
        }
        let lang = inv.get_str("lang").unwrap_or_else(|| "english".to_string());
        let opts = qwen3tts::caps::gen_opts_from(inv).map_err(|e| format!("tts {action}: {e}"))?;
        // The chunk size is per request: one request's must not carry into the next.
        self.engine.set_chunk_frames(inv.get_i64("stream_chunk").map_or(qwen3tts::engine::DEFAULT_CHUNK_FRAMES, |n| n.max(1) as usize));

        // Every action streams its audio the same way: each decoded codec
        // chunk becomes a `Progress::chunk` named "audio", which `Subscribe`
        // forwards as a real out-of-band blob frame. `index` counts chunks
        // from 1, matching the omni resident's convention. A plain `Run`
        // caller sees none of this and still gets the whole waveform below.
        let mut n = 0u32;
        let mut on_audio = |pcm: &[f32], _seq: u32| {
            n += 1;
            progress(Progress::chunk(n, 0, format!("audio chunk {n}"), "audio", pcm_blob(pcm, n)));
        };

        let wav = match action {
            "synth" => self.engine.speak(&text, &lang, &opts, &inv.cancel, &mut on_audio)?,
            "clone" => {
                let (reference, ref_codes) = qwen3tts::caps::reference_from(inv, action)?;
                let ref_text = inv.get_str("ref_text").unwrap_or_default();
                self.engine.clone_voice(&text, &reference, &ref_text, ref_codes.as_deref(), &lang, &opts, &inv.cancel, &mut on_audio)?
            }
            // VoiceDesign/CustomVoice: instruct/speaker are per-call params.
            "design" => {
                let instruct = inv.get_str("instruct").unwrap_or_default();
                let speaker = inv.get_str("speaker").filter(|s| !s.is_empty());
                self.engine.design(&text, &lang, &instruct, speaker.as_deref(), &opts, &inv.cancel, &mut on_audio)?
            }
            other => return Err(format!("tts: unsupported action '{other}' (the hot engine runs: {})", qwen3tts::caps::RESIDENT_ACTIONS.join(", "))),
        };
        qwen3tts::caps::audio_outcome(wav, opts.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::Media;

    /// Activation IS the load now, so a missing checkpoint must surface as a
    /// clean activation error rather than a panic inside a loader or a
    /// first-request failure.
    #[test]
    fn activating_without_weights_is_a_clean_error() {
        let r = TtsResident::new("/nonexistent/tts", "/nonexistent/ckpt");
        let Err(err) = r.activate(&InstanceKey::new(qwen3tts::caps::MODEL, ENGINE_KEY), Device::Cpu) else {
            panic!("activating without weights must fail");
        };
        assert!(err.contains("not found"), "got: {err}");
    }

    /// The speech actions share one hot engine; the provider's own actions
    /// (training above all, which holds the weights four times over on a
    /// card) are costed and keyed apart.
    #[test]
    fn the_speech_actions_share_one_hot_engine_and_the_rest_are_costed_apart() {
        let r = TtsResident::new("/nonexistent/tts", "/nonexistent/ckpt");
        let key = |a: &str| r.instance_key(a, &Invocation::new());
        assert_eq!(key("synth"), key("clone"));
        assert_eq!(key("synth"), key("design"));
        assert_ne!(key("synth"), key("finetune"));
        assert_ne!(key("synth"), key("batch"));
        let cost = |a: &str| r.estimate(&key(a));
        assert!(cost("finetune").vram > 0 && cost("synth").vram == 0, "only training needs card memory");
        assert!(cost("batch").ram > cost("synth").ram, "a batch holds one host decoder per request");
    }

    /// The resident manifest is the provider's, with the hot actions streaming.
    #[test]
    fn the_resident_manifest_declares_streaming_speech_actions() {
        let r = TtsResident::new("/nonexistent/tts", "/nonexistent/ckpt");
        let m = r.manifest();
        assert_eq!(m.model, qwen3tts::caps::MODEL);
        for a in m.actions.iter().filter(|a| qwen3tts::caps::RESIDENT_ACTIONS.contains(&a.name.as_str())) {
            assert!(a.streaming, "{}: must declare streaming - it emits Progress::chunk audio", a.name);
            assert!(a.outputs.iter().any(|o| o.media == Media::Audio), "{}: must declare an audio output", a.name);
        }
    }

    /// End-to-end against real weights: one activation, TWO requests, and the
    /// mid-run chunks must reassemble into the terminal `audio` blob. This is
    /// the property that makes the D-Bus surface a genuine streaming surface -
    /// a subscriber that concatenates the chunks holds exactly the clip a
    /// `Run` caller gets (the same samples, the terminal one as a WAV).
    /// Skips cleanly when the model store holds no Qwen3-TTS checkpoint.
    #[test]
    fn synth_streams_chunks_that_reassemble_into_the_terminal_blob() {
        let Some(r) = TtsResident::from_store(brain_modelstore::default_root().as_deref()) else {
            brain_testutil::skip("no Qwen3-TTS checkpoint in the model store");
            return;
        };
        let key = InstanceKey::new(qwen3tts::caps::MODEL, ENGINE_KEY);
        let t_load = std::time::Instant::now();
        let mut inst = r.activate(&key, Device::Cpu).expect("activate");
        let load_s = t_load.elapsed().as_secs_f64();

        let inv = |t: &str| Invocation::new().set("text", json!(t)).set("max_frames", json!(24)).set("seed", json!(3));
        let mut streamed: Vec<f32> = Vec::new();
        let mut n_chunks = 0usize;
        let t0 = std::time::Instant::now();
        let out = inst
            .run("synth", &inv("Chunks reassemble into the clip."), &mut |p: Progress| {
                if let Some((name, blob)) = &p.chunk {
                    assert_eq!(name, "audio", "chunk blobs must be named 'audio'");
                    assert_eq!(blob.media, Media::Audio);
                    n_chunks += 1;
                    streamed.extend(blob.bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                }
            })
            .expect("synth");
        let first_s = t0.elapsed().as_secs_f64();
        assert!(n_chunks >= 1, "synth emitted no audio chunks");
        assert_eq!(audio::wav::encode(&streamed, 24_000), out.blobs["audio"].bytes, "streamed chunks must equal the terminal audio blob");

        // Same request again on the same hot instance: bit-identical, or
        // resident state (the Talker KV cache, the RNG) leaked between calls.
        let t1 = std::time::Instant::now();
        let out2 = inst.run("synth", &inv("Chunks reassemble into the clip."), &mut |_| {}).expect("second synth");
        let second_s = t1.elapsed().as_secs_f64();
        assert_eq!(out2.blobs["audio"].bytes, out.blobs["audio"].bytes, "a repeated request must be bit-identical - resident state leaked");
        eprintln!("resident tts: activate={load_s:.1}s first={first_s:.1}s repeat={second_s:.1}s ({n_chunks} chunks)");
    }

    /// Cheap guard that survives without weights: the dispatcher's own
    /// argument validation, which happens before any generation.
    #[test]
    fn empty_text_is_rejected_before_any_generation() {
        let m = qwen3tts::caps::resident_manifest();
        let spec = m.actions.iter().find(|a| a.name == "synth").unwrap();
        let err = spec.validate(Invocation::new()).unwrap_err();
        assert!(err.contains("text"), "synth must require 'text': {err}");
    }
}
