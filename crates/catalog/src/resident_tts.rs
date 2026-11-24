// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS voice synthesis behind the residency `Executor`.
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
//! The hot engine runs where the scheduler places it: on a GPU the Talker,
//! MTP and codec are the KV-cached device engine
//! ([`qwen3tts::engine::Placement::Device`]), on the host tier the host
//! decoder (`CpuTalker`/`CpuMtp`), which beats the CPU JIT. Its cost is
//! declared as device memory, so a GPU is chosen when one is budgeted and the
//! host otherwise.
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
use residency::{Device, Instance, InstanceKey, JobFeed, MemCost, ResidentModel};
use serde_json::json;
use qwen3tts::TtsPaths;

/// Instance key of the hot engine: the speech actions.
const ENGINE_KEY: &str = "default";
/// Instance key of `finetune`, which trains the Talker on the device.
const TRAIN_KEY: &str = "train";
/// Instance key of the provider's host-side actions (`batch`, `import`, `sim`).
const HOST_KEY: &str = "host";
/// Instance key of `encode`, which keeps its encoders loaded.
const ENCODE_KEY: &str = "encode";

/// Text-to-speech behind the scheduler, over one Qwen3-TTS checkpoint.
pub struct TtsResident {
    /// The served model id: the checkpoint's own `<vendor>/<repo>` when the
    /// store resolved it, [`qwen3tts::caps::MODEL`] when it was named outright.
    id: String,
    paths: TtsPaths,
}

impl TtsResident {
    /// One resident per Qwen3-TTS checkpoint the model store holds, each under
    /// its own `<vendor>/<repo>` id (`qwen3tts::spec::Qwen3TtsSpec::instance_role`):
    /// the upstream sizes and every trained release are models of their own,
    /// each with its own languages and voices. Pinned roles serve exactly that
    /// checkpoint, as [`qwen3tts::caps::MODEL`].
    pub fn all_from_store(store: &loader::served::ServedStore) -> Vec<TtsResident> {
        loader::served::served_assemblies(store, "qwen3tts", &qwen3tts::spec::Qwen3TtsSpec, qwen3tts::caps::MODEL)
            .into_iter()
            .filter_map(|assembly| Some(TtsResident { paths: TtsPaths::from_assembly(&assembly).ok()?, id: assembly.id }))
            .collect()
    }

    /// Direct constructor for callers that already hold the two paths (e.g.
    /// `brain perf`'s `tts:<weights-dir>:<hf-ckpt-dir>` target).
    pub fn new(weights_dir: impl Into<String>, ckpt_dir: impl Into<String>) -> TtsResident {
        TtsResident { id: qwen3tts::caps::MODEL.to_string(), paths: TtsPaths::new(weights_dir.into(), ckpt_dir) }
    }

    /// The bytes every component's weights occupy as f32, the size both the
    /// host engine and fine-tuning hold them at; 0 when they cannot be read yet.
    fn f32_weights(&self) -> u64 {
        self.paths.f32_weight_bytes()
    }

    /// Host RAM the f32 weights occupy, with a third on top for the working
    /// set; a conservative 4 GiB when they cannot be read yet.
    fn ram(&self) -> u64 {
        let sum = self.f32_weights();
        if sum > 0 { sum + sum / 3 } else { 4 << 30 }
    }
}

impl ResidentModel for TtsResident {
    fn manifest(&self) -> Manifest {
        // The spec lives in qwen3tts::caps, next to the provider's, so the two
        // surfaces cannot silently diverge.
        Manifest { model: self.id.clone(), ..qwen3tts::caps::resident_manifest() }
    }

    fn instance_key(&self, action: &str, _inv: &Invocation) -> InstanceKey {
        let config = match action {
            a if qwen3tts::caps::RESIDENT_ACTIONS.contains(&a) => ENGINE_KEY,
            "finetune" => TRAIN_KEY,
            "encode" => ENCODE_KEY,
            _ => HOST_KEY,
        };
        InstanceKey::new(&self.id, config)
    }

    fn estimate(&self, key: &InstanceKey) -> MemCost {
        match key.config.as_str() {
            // Fine-tuning holds the Talker's weights, their gradients and
            // AdamW's two moments: four copies, the dominant term at any batch.
            TRAIN_KEY => MemCost::new(self.f32_weights().max(1 << 30) * 4, self.ram()),
            // `batch` gives each request its own host decoder copy.
            HOST_KEY => MemCost::new(0, self.ram() * 2),
            // Each encoder holds a codec and a speaker encoder, with a third on top for the working set.
            ENCODE_KEY => MemCost::new(0, (self.paths.encoder_weight_bytes().max(1 << 30) * 4 / 3) * qwen3tts::encode::STREAMS as u64),
            // The hot engine's weights and working set, on whichever device holds it, and on a
            // card the KV pool its batched decoder serves every stream from.
            _ => MemCost::new(self.ram() + qwen3tts::batched::kv_pool_bytes(&self.paths, qwen3tts::engine::DEVICE_STREAMS).unwrap_or(0), 0),
        }
    }

    /// A card decodes this many speech requests at once, so the dispatcher hands the engine
    /// everything queued up to that.
    fn batch_limit(&self) -> Option<usize> {
        Some(qwen3tts::engine::DEVICE_STREAMS)
    }

    fn activate(&self, key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        if key.config == ENCODE_KEY {
            self.paths.require(true).map_err(|e| format!("tts encode: {e} (encoding needs a Base checkpoint, which ships the speaker encoder)"))?;
            let encoders = residency::on_device(device, || (0..qwen3tts::encode::STREAMS).map(|_| qwen3tts::encode::Encoder::load(&self.paths)).collect::<Result<Vec<_>, _>>())??;
            return Ok(Box::new(EncodeInstance { encoders, device }));
        }
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
        let placement = match device {
            Device::Gpu(_) => qwen3tts::engine::Placement::Device,
            Device::Cpu | Device::Npu(_) => qwen3tts::engine::Placement::Host,
        };
        let engine = residency::on_device(device, || qwen3tts::ResidentEngine::load_with(&self.paths, placement))??;
        Ok(Box::new(TtsInstance { engine, device }))
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
        residency::on_device(self.device, || act.run(inv, progress))?
    }
}

/// `encode` with its encoders loaded once: a batch of recordings is shared
/// out over them, each encoder working through its share on its own thread.
/// The codec encode is not yet one pass over several clips; side-by-side
/// encoders are what lifts throughput on the CPU (`qwen3tts::encode::STREAMS`).
struct EncodeInstance {
    encoders: Vec<qwen3tts::encode::Encoder>,
    device: Device,
}

impl EncodeInstance {
    fn one(encoder: &qwen3tts::encode::Encoder, inv: &Invocation) -> ActionResult {
        inv.cancel.check()?;
        let req = qwen3tts::caps::encode_request(inv)?;
        qwen3tts::caps::run_encode(encoder, inv, &req)
    }
}

impl Instance for EncodeInstance {
    fn run(&mut self, action: &str, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        if action != "encode" {
            return Err(format!("tts: the encode instance runs encode, not '{action}'"));
        }
        residency::on_device(self.device, || Self::one(&self.encoders[0], inv))?
    }

    fn run_batch(&mut self, action: &str, invs: &[Invocation], _progress: &mut dyn FnMut(usize, Progress)) -> Vec<ActionResult> {
        if action != "encode" {
            return invs.iter().map(|_| Err(format!("tts: the encode instance runs encode, not '{action}'"))).collect();
        }
        let device = self.device;
        let share = invs.len().div_ceil(self.encoders.len()).max(1);
        std::thread::scope(|s| {
            let workers: Vec<_> = invs
                .chunks(share)
                .zip(&self.encoders)
                .map(|(chunk, encoder)| s.spawn(move || chunk.iter().map(|inv| residency::on_device(device, || Self::one(encoder, inv)).and_then(|r| r)).collect::<Vec<_>>()))
                .collect();
            workers.into_iter().flat_map(|w| w.join().unwrap_or_else(|_| vec![Err("tts encode: an encoder thread panicked".to_string())])).collect()
        })
    }
}

struct TtsInstance {
    engine: qwen3tts::ResidentEngine,
    /// Where the engine was built: every request runs scoped to it, so what a
    /// request builds (a new reference's speaker encoder) lands there too.
    device: Device,
}

/// Raw mono f32 little-endian PCM as a capability [`Blob`]: what a mid-run
/// chunk carries (the terminal artifact is the whole clip as a WAV).
fn pcm_blob(pcm: &[f32], index: u32) -> capability::Blob {
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    capability::Blob::new(capability::Media::Audio, bytes).with_meta(json!({"sample_rate": qwen3tts::engine::SAMPLE_RATE, "format": "pcm_f32", "channels": 1, "index": index}))
}

impl Instance for TtsInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let device = self.device;
        residency::on_device(device, || {
            let req = request_from(action, inv)?;
            let seed = req.opts.seed;
            let mut chunks = AudioChunks::default();
            let wav = self.engine.run_request(&req, &mut |pcm, _| progress(chunks.next(pcm)))?;
            qwen3tts::caps::audio_outcome(wav, seed)
        })?
    }

    /// Every speech request queued for this engine joins one open group: on a
    /// card they are decoded together, a new one admitted as soon as one ends.
    fn run_open(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(usize, Progress), feed: &mut dyn JobFeed) -> bool {
        if !qwen3tts::caps::RESIDENT_ACTIONS.contains(&action) {
            return false;
        }
        let device = self.device;
        let engine = &mut self.engine;
        residency::on_device(device, || {
            let feed = std::cell::RefCell::new(feed);
            let mut first: std::collections::VecDeque<(usize, Invocation)> = invs.iter().cloned().enumerate().collect();
            let mut joined = invs.len();
            // Each running request's seed (its outcome reports it) and streamed-chunk count.
            let running: std::cell::RefCell<std::collections::HashMap<usize, (u64, AudioChunks)>> = std::cell::RefCell::default();
            loop {
                engine.serve(
                    &mut |free| {
                        let mut taken: Vec<(usize, Invocation)> = first.drain(..free.min(first.len())).collect();
                        for inv in feed.borrow_mut().take(free - taken.len()) {
                            taken.push((joined, inv));
                            joined += 1;
                        }
                        let mut reqs = Vec::new();
                        for (i, inv) in taken {
                            match request_from(action, &inv) {
                                Ok(req) => {
                                    running.borrow_mut().insert(i, (req.opts.seed, AudioChunks::default()));
                                    reqs.push((i, req));
                                }
                                Err(e) => feed.borrow_mut().finish(i, Err(e)),
                            }
                        }
                        reqs
                    },
                    &mut |i, pcm, _| {
                        let chunk = running.borrow_mut().entry(i).or_default().1.next(pcm);
                        progress(i, chunk);
                    },
                    &mut |i, result| {
                        let seed = running.borrow_mut().remove(&i).map_or(0, |(seed, _)| seed);
                        feed.borrow_mut().finish(i, result.and_then(|wav| qwen3tts::caps::audio_outcome(wav, seed)));
                    },
                );
                if first.is_empty() && feed.borrow_mut().close() {
                    return;
                }
            }
        })
        .is_ok()
    }
}

/// Numbers a request's streamed audio: each decoded codec chunk becomes a
/// `Progress::chunk` named "audio", which `Subscribe` forwards as a real
/// out-of-band blob frame. `index` counts chunks from 1, matching the omni
/// resident's convention. A plain `Run` caller sees none of this and still
/// gets the whole waveform.
#[derive(Default)]
struct AudioChunks {
    n: u32,
}

impl AudioChunks {
    fn next(&mut self, pcm: &[f32]) -> Progress {
        self.n += 1;
        let n = self.n;
        Progress::chunk(n, 0, format!("audio chunk {n}"), "audio", pcm_blob(pcm, n))
    }
}

/// The speech request `inv` asks `action` for.
fn request_from(action: &str, inv: &Invocation) -> Result<qwen3tts::engine::Request, String> {
    use qwen3tts::engine::Voice;
    let text = inv.get_str("text").ok_or_else(|| format!("tts {action}: missing required param 'text'"))?;
    if text.trim().is_empty() {
        return Err(format!("tts {action}: 'text' must be non-empty"));
    }
    let voice = match action {
        "synth" => Voice::Default,
        "clone" => {
            let (reference, ref_codes) = qwen3tts::caps::reference_from(inv, action)?;
            Voice::Clone { reference, ref_text: inv.get_str("ref_text").unwrap_or_default(), ref_codes }
        }
        // VoiceDesign/CustomVoice: instruct/speaker are per-call params.
        "design" => Voice::Design { instruct: inv.get_str("instruct").unwrap_or_default(), speaker: inv.get_str("speaker").filter(|s| !s.is_empty()) },
        other => return Err(format!("tts: unsupported action '{other}' (the hot engine runs: {})", qwen3tts::caps::RESIDENT_ACTIONS.join(", "))),
    };
    Ok(qwen3tts::engine::Request {
        text,
        lang: inv.get_str("lang").unwrap_or_else(|| "english".to_string()),
        voice,
        opts: qwen3tts::caps::gen_opts_from(inv).map_err(|e| format!("tts {action}: {e}"))?,
        cancel: inv.cancel.clone(),
        // The chunk size is per request: one request's must not carry into the next.
        chunk_frames: inv.get_i64("stream_chunk").map_or(qwen3tts::engine::DEFAULT_CHUNK_FRAMES, |n| n.max(1) as usize),
    })
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
        assert!(cost("synth").vram > 0, "the hot engine goes on a card when one is budgeted");
        assert!(cost("finetune").vram >= cost("synth").vram, "training holds the weights four times over");
        assert!(cost("batch").ram > 0 && cost("batch").vram == 0, "a batch holds host decoders, one per request");
        assert!(key("encode") != key("synth") && key("encode") != key("batch"), "encode keeps its own encoders loaded");
        assert!(cost("encode").ram > 0 && cost("encode").vram == 0, "the encoders run on the host");
    }

    /// Two Qwen3-TTS checkpoints in one store (an upstream release and a
    /// trained one) are two served models, each under its own id and over its
    /// own files, with instances of their own.
    #[test]
    fn every_qwen3tts_checkpoint_in_the_store_is_its_own_served_model() {
        let store = brain_testutil::Scratch::dir("brain-resident-tts-two");
        for (id, weights_dir) in [("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "."), ("local/sv-voices", ".")] {
            let repo = store.path().join(id);
            std::fs::create_dir_all(repo.join("speech_tokenizer")).unwrap();
            std::fs::write(repo.join("config.json"), br#"{"architectures": ["Qwen3TTSForConditionalGeneration"]}"#).unwrap();
            let manifest = brain_modelstore::CompoundManifest { id: id.to_string(), family: "qwen3tts".to_string(), roles: [("ckpt", "."), ("weights_dir", weights_dir)].into_iter().map(|(r, p)| (r.to_string(), p.to_string())).collect() };
            std::fs::write(repo.join(brain_modelstore::MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
        }
        let mut served: Vec<(String, String)> = TtsResident::all_from_store(&loader::served::ServedStore::new(Some(store.path().to_path_buf()))).into_iter().map(|r| (r.manifest().model, r.paths.ckpt_dir.clone())).collect();
        served.sort();
        let ids: Vec<&str> = served.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["Qwen/Qwen3-TTS-12Hz-0.6B-Base", "local/sv-voices"]);
        for (id, ckpt) in &served {
            assert!(std::path::Path::new(ckpt).starts_with(store.path().join(id)), "{id} is served from its own files: {ckpt}");
        }
        let both = TtsResident::all_from_store(&loader::served::ServedStore::new(Some(store.path().to_path_buf())));
        let keys: Vec<InstanceKey> = both.iter().map(|r| r.instance_key("synth", &Invocation::new())).collect();
        assert_ne!(keys[0], keys[1], "each model has its own engine");
    }

    /// The encode instance refuses a checkpoint without a speaker encoder at
    /// activation, not on the first clip.
    #[test]
    fn encode_needs_a_base_checkpoint_at_activation() {
        let r = TtsResident::new("/nonexistent/tts", "/nonexistent/ckpt");
        let e = r.activate(&r.instance_key("encode", &Invocation::new()), Device::Cpu).err().expect("no checkpoint, no instance");
        assert!(e.contains("tts encode"), "{e}");
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
        let Some(r) = TtsResident::all_from_store(&loader::served::ServedStore::new(brain_modelstore::default_root())).into_iter().next() else {
            brain_testutil::skip("no Qwen3-TTS checkpoint in the model store");
            return;
        };
        let key = r.instance_key("synth", &Invocation::new());
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
                    streamed.extend(blob.bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
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
