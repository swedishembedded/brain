// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A resident text-to-speech server for the **OpenVINO/NPU** Talker graphs,
//! speaking line-delimited JSON over a Unix socket: the compiled graphs load
//! once, then back-to-back requests skip the per-process graph load, and each
//! request's audio streams back as it is decoded.
//!
//! The ordinary host path needs none of this: `brain serve` already offers
//! load-once weights and chunk-streamed audio for every action (see
//! `catalog::resident_tts`). What lives here is the compiled-graph
//! configuration a generic residency surface has no vocabulary for - the
//! Talker's weight precision, the MTP engine, the codec mode, the compile
//! cache - and the socket protocol around it.
//!
//! Architecture:
//!  * a single **executor thread** owns every resident [`TtsEngine`] (OpenVINO
//!    infer requests are not shared across threads) and pulls queued jobs;
//!  * one **connection thread** per client reads a request line, enqueues a
//!    job and streams the generated PCM back as `audio_chunk` lines.
//!
//! Request (client to server, one JSON object per line):
//! `{"engine":"clone|design|customvoice|synth","text":"...","instruct":"...",
//! "speaker":"ryan","lang":"english","temp":0.9,"top_k":50,"top_p":1.0,
//! "repetition_penalty":1.05,"residual_temp":0.9,"residual_top_k":50,
//! "residual_top_p":1.0,"seed":0,"max_frames":256}`
//!
//! Response (server to client, one JSON object per line):
//! `{"event":"audio_chunk","pcm_b64":"<f32le base64>","sample_rate":24000,"seq":N,"done":false}`
//! ... terminated by `{"event":"audio_chunk","pcm_b64":"","seq":M,"done":true,"seed":S}`
//! where S is the seed the request generated under: the one it named, else
//! one drawn for it - sending it back as `"seed"` reproduces the audio. A
//! failure is `{"event":"error","message":"..."}`.
//!
//! Swedish Embedded AB implements low-latency speech synthesis on NPUs for
//! its clients. If your team needs expertise in serving neural speech from an
//! accelerator with the weights kept hot then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;

use data::base64;

use crate::npu_gen::Target;
use crate::placement::{CodecMode, MtpMode};
use crate::serve::{EngineCfg, Kind, Req, TtsEngine};
use crate::{GenOpts, TtsPaths};

const SAMPLE_RATE: u32 = 24_000;

/// Compiled KV context (prefix plus frames) when the caller names none: a
/// clone's long reference prefix needs more than a short text-only prompt.
const CLONE_CAP: usize = 384;
const SHORT_PROMPT_CAP: usize = 192;

/// The engines a server can host. Each is one resident compiled model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EngineName {
    /// In-context clone of one fixed reference voice.
    Clone,
    /// VoiceDesign: a per-request instruct.
    Design,
    /// CustomVoice: a per-request instruct and preset speaker.
    CustomVoice,
    /// Speaker-free synthesis.
    Synth,
}

impl EngineName {
    pub const NAMES: [&'static str; 4] = ["clone", "design", "customvoice", "synth"];

    pub fn parse(name: &str) -> Result<EngineName, String> {
        match name {
            "clone" => Ok(EngineName::Clone),
            "design" => Ok(EngineName::Design),
            "customvoice" => Ok(EngineName::CustomVoice),
            "synth" => Ok(EngineName::Synth),
            other => Err(format!("unknown engine {other:?} (one of {})", Self::NAMES.join(", "))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            EngineName::Clone => "clone",
            EngineName::Design => "design",
            EngineName::CustomVoice => "customvoice",
            EngineName::Synth => "synth",
        }
    }
}

/// The Talker's weight precision in the compiled decode graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TalkerQuant {
    /// INT8 weights (the default).
    Int8,
    /// INT4 weight-compression: faster on the bandwidth-bound Talker and half
    /// the graph RAM, though not native 4-bit compute on the Intel NPU.
    Int4,
    Fp32,
}

impl TalkerQuant {
    pub const NAMES: [&'static str; 3] = ["int8", "int4", "fp32"];

    pub fn parse(name: &str) -> Result<TalkerQuant, String> {
        match name {
            "int8" => Ok(TalkerQuant::Int8),
            "int4" => Ok(TalkerQuant::Int4),
            "fp32" => Ok(TalkerQuant::Fp32),
            other => Err(format!("unknown talker quant {other:?} (one of {})", Self::NAMES.join(", "))),
        }
    }

    /// `(weight-quantized, int4)` as [`EngineCfg`] spells them.
    fn flags(self) -> (bool, bool) {
        match self {
            TalkerQuant::Int8 => (true, false),
            TalkerQuant::Int4 => (true, true),
            TalkerQuant::Fp32 => (false, false),
        }
    }
}

/// One engine's checkpoint: where its weights and its HF config/tokenizer are.
#[derive(Clone, Debug)]
pub struct ModelDirs {
    pub weights_dir: String,
    pub ckpt_dir: String,
}

/// The clone engine's fixed reference voice.
#[derive(Clone, Debug)]
pub struct CloneVoice {
    pub model: ModelDirs,
    /// The reference WAV.
    pub ref_wav: String,
    /// The reference's transcript; empty runs x-vector-only cloning.
    pub ref_text: String,
}

/// Everything a server needs.
#[derive(Clone, Debug)]
pub struct ServerCfg {
    /// The Unix socket to listen on; a stale file there is replaced.
    pub socket: PathBuf,
    pub target: Target,
    /// The compiled KV context length of every engine; `None` picks a size per engine.
    pub cap: Option<usize>,
    pub talker_quant: TalkerQuant,
    pub mtp: Option<MtpMode>,
    pub codec: CodecMode,
    pub stream_chunk: usize,
    pub stream_window: usize,
    pub profile: bool,
    pub clone: Option<CloneVoice>,
    pub design: Option<ModelDirs>,
    pub custom_voice: Option<ModelDirs>,
    /// Speaker-free synthesis: it needs only a checkpoint.
    pub synth: Option<ModelDirs>,
}

impl ServerCfg {
    /// The engine configurations this server hosts, each named. An engine
    /// with no checkpoint configured is not hosted.
    fn engines(&self) -> Result<HashMap<EngineName, EngineCfg>, String> {
        let (quant, int4) = self.talker_quant.flags();
        let cfg = |kind: Kind, model: &ModelDirs, default_cap: usize, voice: Option<&CloneVoice>| EngineCfg {
            kind,
            weights_dir: model.weights_dir.clone(),
            ckpt_dir: model.ckpt_dir.clone(),
            // An engine always has a cache dir: with no resolvable cache
            // directory it falls back to one under the temp dir.
            npu_cache: TtsPaths::new(&model.weights_dir, String::new())
                .npu_cache_dir()
                .unwrap_or_else(|| std::env::temp_dir().join("brain-qwen3tts-npu-cache"))
                .to_string_lossy()
                .into_owned(),
            device: self.target,
            cap: self.cap.unwrap_or(default_cap),
            quant,
            int4,
            ref_wav: voice.map(|v| v.ref_wav.clone()),
            ref_text: voice.map(|v| v.ref_text.clone()),
            mtp: self.mtp,
            codec: self.codec,
            stream_chunk: self.stream_chunk,
            stream_window: self.stream_window,
            profile: self.profile,
        };
        let mut engines = HashMap::new();
        if let Some(v) = &self.clone {
            engines.insert(EngineName::Clone, cfg(Kind::Clone, &v.model, CLONE_CAP, Some(v)));
        }
        if let Some(m) = &self.design {
            engines.insert(EngineName::Design, cfg(Kind::Design, m, SHORT_PROMPT_CAP, None));
        }
        if let Some(m) = &self.custom_voice {
            engines.insert(EngineName::CustomVoice, cfg(Kind::Design, m, SHORT_PROMPT_CAP, None));
        }
        if let Some(m) = &self.synth {
            engines.insert(EngineName::Synth, cfg(Kind::Synth, m, SHORT_PROMPT_CAP, None));
        }
        if engines.is_empty() {
            return Err("no engine is configured to serve".to_string());
        }
        Ok(engines)
    }
}

/// Messages streamed from the executor back to a connection thread.
enum Msg {
    Audio { pcm_b64: String, seq: u32 },
    Done { samples: usize, ms: f64, seed: u64 },
    Error(String),
}

/// A queued request: which engine, the parsed request, where to stream results.
struct Job {
    engine: EngineName,
    req: Req,
    reply: Sender<Msg>,
}

/// Serve until the process ends. Engines load lazily on their first request
/// (the one-time graph compile), so the socket is accepting at once.
pub fn serve(cfg: &ServerCfg) -> Result<(), String> {
    let engines = cfg.engines()?;
    let hosted: Vec<&str> = engines.keys().map(|e| e.name()).collect();

    let (job_tx, job_rx): (Sender<Job>, Receiver<Job>) = channel();
    let exec = thread::spawn(move || executor(engines, job_rx));

    let _ = std::fs::remove_file(&cfg.socket);
    let listener = UnixListener::bind(&cfg.socket).map_err(|e| format!("bind {}: {e}", cfg.socket.display()))?;
    eprintln!("qwen3tts serve: listening on {} (engines: {})", cfg.socket.display(), hosted.join(", "));
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let jobs = job_tx.clone();
                thread::spawn(move || handle_conn(s, jobs));
            }
            Err(e) => eprintln!("qwen3tts serve: accept: {e}"),
        }
    }
    drop(job_tx);
    let _ = exec.join();
    Ok(())
}

/// Compile and load one engine. A panic in a loader reports an error instead of
/// killing the executor, which would hang every future request.
fn load_engine(name: &str, cfg: &EngineCfg) -> Result<TtsEngine, String> {
    eprintln!("qwen3tts serve: loading engine {name:?} (one-time compile)...");
    let t = Instant::now();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| TtsEngine::load(cfg.clone()))) {
        Ok(Ok(e)) => {
            eprintln!("qwen3tts serve: engine {name:?} ready on {} in {:.1}s", e.device(), t.elapsed().as_secs_f64());
            Ok(e)
        }
        Ok(Err(e)) => Err(format!("load engine {name:?}: {e}")),
        Err(p) => Err(format!("load engine {name:?} panicked: {}", panic_msg(&p))),
    }
}

/// The executor: load each engine on its first request, then serve jobs FIFO.
fn executor(cfgs: HashMap<EngineName, EngineCfg>, jobs: Receiver<Job>) {
    use std::collections::hash_map::Entry;
    let mut loaded: HashMap<EngineName, TtsEngine> = HashMap::new();
    for job in jobs {
        let name = job.engine.name();
        let engine = match loaded.entry(job.engine) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(slot) => {
                let loading = cfgs.get(&job.engine).ok_or_else(|| format!("engine {name:?} is not served here")).and_then(|cfg| load_engine(name, cfg));
                match loading {
                    Ok(e) => slot.insert(e),
                    Err(msg) => {
                        let _ = job.reply.send(Msg::Error(msg));
                        continue;
                    }
                }
            }
        };
        let t = Instant::now();
        let reply = job.reply.clone();
        let req = &job.req;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut on_audio = |pcm: &[f32], seq: u32| {
                let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                let _ = reply.send(Msg::Audio { pcm_b64: base64::encode(&bytes), seq });
            };
            engine.run(req, &mut on_audio)
        }));
        let _ = job.reply.send(match res {
            Ok(Ok(samples)) => Msg::Done { samples, ms: t.elapsed().as_secs_f64() * 1e3, seed: req.opts.seed },
            Ok(Err(e)) => Msg::Error(e),
            Err(p) => Msg::Error(format!("generation panicked: {}", panic_msg(&p))),
        });
    }
}

/// Best-effort string from a panic payload.
fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// One client connection: read JSONL requests, enqueue jobs, stream responses.
fn handle_conn(stream: UnixStream, jobs: Sender<Job>) {
    let Ok(mut w) = stream.try_clone() else { return };
    for line in BufReader::new(stream).lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            _ => continue,
        };
        let (engine, req) = match parse_request(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(w, "{}", error_line(&e));
                continue;
            }
        };
        let (tx, rx) = channel();
        if jobs.send(Job { engine, req, reply: tx }).is_err() {
            let _ = writeln!(w, "{}", error_line("server shutting down"));
            return;
        }
        for msg in rx {
            let out = match msg {
                Msg::Audio { pcm_b64, seq } => audio_line(&pcm_b64, seq, None),
                Msg::Done { samples, ms, seed } => {
                    eprintln!("qwen3tts serve: served {samples} samples in {ms:.0}ms (seed {seed})");
                    audio_line("", u32::MAX, Some(seed))
                }
                Msg::Error(e) => error_line(&e),
            };
            if writeln!(w, "{out}").and_then(|()| w.flush()).is_err() {
                return;
            }
        }
    }
}

/// A request line as the engine it names and the [`Req`] it asks for. Every
/// sampling key left out stays unset and resolves from the checkpoint's own
/// `generation_config.json`, then the reference defaults.
fn parse_request(line: &str) -> Result<(EngineName, Req), String> {
    let v: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("bad JSON: {e}"))?;
    let engine = EngineName::parse(v["engine"].as_str().unwrap_or("clone"))?;
    let text = v["text"].as_str().unwrap_or_default().to_string();
    if text.trim().is_empty() {
        return Err("request missing non-empty \"text\"".into());
    }
    let float = |key: &str| v[key].as_f64().map(|x| x as f32);
    let count = |key: &str| v[key].as_u64().map(|x| x as usize);
    let mut opts = GenOpts::default();
    opts.sampling.temperature = float("temp");
    opts.sampling.top_k = count("top_k");
    opts.sampling.top_p = float("top_p");
    opts.sampling.repetition_penalty = float("repetition_penalty");
    opts.sampling.subtalker.temperature = float("residual_temp");
    opts.sampling.subtalker.top_k = count("residual_top_k");
    opts.sampling.subtalker.top_p = float("residual_top_p");
    if let Some(x) = v["seed"].as_u64() {
        opts.seed = x;
    }
    if let Some(x) = count("max_frames") {
        opts.max_frames = x;
    }
    let req = Req {
        text,
        instruct: v["instruct"].as_str().unwrap_or_default().to_string(),
        speaker: v["speaker"].as_str().map(str::to_string),
        lang: v["lang"].as_str().unwrap_or("english").to_string(),
        opts,
    };
    Ok((engine, req))
}

/// One `audio_chunk` line. The final chunk is the one that carries the seed the
/// request generated under (`done_seed`); the others carry none.
fn audio_line(pcm_b64: &str, seq: u32, done_seed: Option<u64>) -> String {
    let mut line = serde_json::json!({
        "event": "audio_chunk", "pcm_b64": pcm_b64,
        "sample_rate": SAMPLE_RATE, "seq": seq, "done": done_seed.is_some(),
    });
    if let Some(seed) = done_seed {
        line["seed"] = serde_json::json!(seed);
    }
    line.to_string()
}

fn error_line(msg: &str) -> String {
    serde_json::json!({"event": "error", "message": msg}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_of(request: &str) -> u64 {
        parse_request(request).expect("valid request").1.opts.seed
    }

    /// A request that names no seed gets a drawn one, and the final chunk
    /// reports it, so naming it in a second request reproduces the first.
    #[test]
    fn a_drawn_seed_is_reported_on_the_final_chunk_and_replays() {
        let unseeded = r#"{"text":"hello"}"#;
        let drawn = seed_of(unseeded);
        assert_ne!(drawn, seed_of(unseeded), "an unseeded request draws a fresh seed each time");

        let done: serde_json::Value = serde_json::from_str(&audio_line("", u32::MAX, Some(drawn))).expect("json");
        assert_eq!(done["done"], true);
        let reported = done["seed"].as_u64().expect("the final chunk carries the seed");
        assert_eq!(seed_of(&format!(r#"{{"text":"hello","seed":{reported}}}"#)), drawn);

        let mid: serde_json::Value = serde_json::from_str(&audio_line("AAAA", 3, None)).expect("json");
        assert_eq!(mid["done"], false);
        assert!(mid.get("seed").is_none());
    }

    #[test]
    fn a_request_names_its_engine_and_every_sampling_knob() {
        let (engine, req) = parse_request(
            r#"{"engine":"design","text":"hi","instruct":"calm","speaker":"ryan","lang":"german","temp":0.7,"top_k":20,"top_p":0.9,"repetition_penalty":1.1,"residual_temp":0.5,"residual_top_k":10,"residual_top_p":0.8,"max_frames":64}"#,
        )
        .unwrap();
        assert_eq!(engine, EngineName::Design);
        assert_eq!((req.instruct.as_str(), req.speaker.as_deref(), req.lang.as_str(), req.opts.max_frames), ("calm", Some("ryan"), "german", 64));
        let s = &req.opts.sampling;
        assert_eq!((s.temperature, s.top_k, s.top_p, s.repetition_penalty), (Some(0.7), Some(20), Some(0.9), Some(1.1)));
        assert_eq!((s.subtalker.temperature, s.subtalker.top_k, s.subtalker.top_p), (Some(0.5), Some(10), Some(0.8)));
        // Unnamed engine: the clone engine, as the protocol always defaulted.
        assert_eq!(parse_request(r#"{"text":"hi"}"#).unwrap().0, EngineName::Clone);
    }

    #[test]
    fn a_bad_request_is_refused_with_a_reason() {
        assert!(parse_request("not json").unwrap_err().contains("bad JSON"));
        assert!(parse_request(r#"{"text":"  "}"#).unwrap_err().contains("text"));
        assert!(parse_request(r#"{"engine":"tenor","text":"hi"}"#).unwrap_err().contains("customvoice"));
    }

    #[test]
    fn only_configured_engines_are_hosted() {
        let model = ModelDirs { weights_dir: "w".into(), ckpt_dir: "c".into() };
        let cfg = ServerCfg {
            socket: std::env::temp_dir().join("s"),
            target: Target::new(npu::openvino::NpuDevice::Cpu, false),
            cap: None,
            talker_quant: TalkerQuant::Int4,
            mtp: None,
            codec: CodecMode::NpuStream,
            stream_chunk: 16,
            stream_window: 32,
            profile: false,
            clone: None,
            design: None,
            custom_voice: None,
            synth: Some(model),
        };
        let engines = cfg.engines().unwrap();
        assert_eq!(engines.len(), 1);
        let synth = &engines[&EngineName::Synth];
        assert_eq!((synth.cap, synth.quant, synth.int4), (SHORT_PROMPT_CAP, true, true));
        assert!(ServerCfg { synth: None, ..cfg }.engines().unwrap_err().contains("no engine"));
    }
}
