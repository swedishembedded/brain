// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain tts serve` — resident-engine TTS server.
//!
//! Loads the compiled NPU graphs ONCE and serves many requests over a Unix socket
//! using a line-delimited JSON (JSONL) protocol, so back-to-back generations skip
//! the per-process graph-load. Architecture:
//!
//!  * a single **executor thread** owns the resident [`tts::serve::TtsEngine`]s
//!    (OpenVINO infer requests are not shared across threads) and pulls **jobs**
//!    from a channel — work to the accelerator is a queued effect, scheduled here;
//!  * one **connection handler thread** per client reads a request line, enqueues
//!    a job, and streams the generated PCM back as `audio_chunk` lines.
//!
//! Request  (client -> server, one JSON object per line):
//!   {"engine":"clone|design|customvoice|synth","text":"...","instruct":"...",
//!    "speaker":"ryan","lang":"english","temp":0.9,"top_k":50,"seed":0,
//!    "max_frames":256}
//! Response (server -> client, JSONL):
//!   {"event":"audio_chunk","pcm_b64":"<f32le base64>","sample_rate":24000,"seq":N,"done":false}
//!   ... terminated by {"event":"audio_chunk","pcm_b64":"","seq":M,"done":true}
//!   or {"event":"error","message":"..."}

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;

use events::base64;
use tts::serve::{EngineCfg, Kind, Req, TtsEngine};
use tts::GenOpts;

const SAMPLE_RATE: u32 = 24000;
const CHUNK_SAMPLES: usize = 4800; // 0.2 s @ 24 kHz per streamed chunk

/// Messages streamed from the executor back to a connection handler.
enum Msg {
    Audio { pcm_b64: String, seq: u32 },
    Done { samples: usize, ms: f64 },
    Error(String),
}

/// A queued request ("effect"): which engine, the parsed request, and where to
/// stream results.
struct Job {
    engine: String,
    req: Req,
    reply: Sender<Msg>,
}

pub fn run_serve(args: &[String]) {
    let mut socket = "/tmp/brain-tts.sock".to_string();
    let mut cap_override: Option<usize> = None;
    let mut engines: HashMap<String, EngineCfg> = HashMap::new();

    // Per-engine flag triplets/quads. Defaults below cover the known local setup.
    let res = "/data/workspace/resources/tts/qwen3-tts";
    let mut clone_w = "out/tts-1b7".to_string();
    let mut clone_c = format!("{res}/ckpt/Qwen3-TTS-12Hz-1.7B-Base");
    let mut clone_ref = format!("{res}/voice-clone-example-voice.wav");
    let mut clone_ref_text = format!("{res}/voice-clone-example-voice.txt");
    let mut design_w = "out/tts-vd".to_string();
    let mut design_c = format!("{res}/ckpt/Qwen3-TTS-12Hz-1.7B-VoiceDesign");
    let mut cv_w = "out/tts-cv".to_string();
    let mut cv_c = format!("{res}/ckpt/Qwen3-TTS-12Hz-1.7B-CustomVoice");
    let mut enable: Vec<String> = vec![];

    let val = |a: &[String], i: &mut usize| {
        *i += 1;
        a.get(*i).cloned().unwrap_or_default()
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" => socket = val(args, &mut i),
            "--cap" => cap_override = val(args, &mut i).parse().ok(),
            "--enable" => enable = val(args, &mut i).split(',').map(|s| s.trim().to_string()).collect(),
            "--clone-weights" => clone_w = val(args, &mut i),
            "--clone-ckpt" => clone_c = val(args, &mut i),
            "--clone-ref" => clone_ref = val(args, &mut i),
            "--clone-ref-text" => clone_ref_text = val(args, &mut i),
            "--design-weights" => design_w = val(args, &mut i),
            "--design-ckpt" => design_c = val(args, &mut i),
            "--cv-weights" => cv_w = val(args, &mut i),
            "--cv-ckpt" => cv_c = val(args, &mut i),
            other => eprintln!("tts serve: ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if enable.is_empty() {
        enable = vec!["clone".into(), "design".into(), "customvoice".into(), "synth".into()];
    }

    let device = npu::openvino::NpuDevice::Npu;
    let cache = |w: &str| format!("{w}/npu-cache");
    let ref_text = std::fs::read_to_string(&clone_ref_text).unwrap_or_default().trim().to_string();
    for name in &enable {
        let cfg = match name.as_str() {
            "clone" => EngineCfg {
                kind: Kind::Clone,
                weights_dir: clone_w.clone(),
                ckpt_dir: clone_c.clone(),
                npu_cache: cache(&clone_w),
                device,
                cap: cap_override.unwrap_or(384),
                quant: true,
                ref_wav: Some(clone_ref.clone()),
                ref_text: Some(ref_text.clone()),
            },
            "design" => EngineCfg {
                kind: Kind::Design,
                weights_dir: design_w.clone(),
                ckpt_dir: design_c.clone(),
                npu_cache: cache(&design_w),
                device,
                cap: cap_override.unwrap_or(256),
                quant: true,
                ref_wav: None,
                ref_text: None,
            },
            "customvoice" => EngineCfg {
                kind: Kind::Design,
                weights_dir: cv_w.clone(),
                ckpt_dir: cv_c.clone(),
                npu_cache: cache(&cv_w),
                device,
                cap: cap_override.unwrap_or(256),
                quant: true,
                ref_wav: None,
                ref_text: None,
            },
            "synth" => EngineCfg {
                kind: Kind::Synth,
                weights_dir: design_w.clone(),
                ckpt_dir: design_c.clone(),
                npu_cache: cache(&design_w),
                device,
                cap: cap_override.unwrap_or(256),
                quant: true,
                ref_wav: None,
                ref_text: None,
            },
            other => {
                eprintln!("tts serve: unknown engine {other:?}");
                continue;
            }
        };
        engines.insert(name.clone(), cfg);
    }

    // Executor thread: owns the resident engines, services jobs FIFO.
    let (job_tx, job_rx): (Sender<Job>, Receiver<Job>) = channel();
    let exec = thread::spawn(move || executor(engines, job_rx));

    let _ = std::fs::remove_file(&socket);
    let listener = match UnixListener::bind(&socket) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("tts serve: bind {socket}: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("tts serve: listening on {socket} (engines: {})", enable.join(", "));
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let jt = job_tx.clone();
                thread::spawn(move || handle_conn(s, jt));
            }
            Err(e) => eprintln!("tts serve: accept: {e}"),
        }
    }
    drop(job_tx);
    let _ = exec.join();
}

/// The executor: lazily load engines (the one-time graph compile), then serve.
fn executor(cfgs: HashMap<String, EngineCfg>, jobs: Receiver<Job>) {
    let mut loaded: HashMap<String, TtsEngine> = HashMap::new();
    for job in jobs {
        if !loaded.contains_key(&job.engine) {
            let Some(cfg) = cfgs.remove_or_clone(&job.engine) else {
                let _ = job.reply.send(Msg::Error(format!("unknown engine {:?}", job.engine)));
                continue;
            };
            eprintln!("tts serve: loading engine {:?} (one-time compile)…", job.engine);
            let t = Instant::now();
            match TtsEngine::load(cfg) {
                Ok(e) => {
                    eprintln!("tts serve: engine {:?} ready on {} in {:.1}s", job.engine, e.device(), t.elapsed().as_secs_f64());
                    loaded.insert(job.engine.clone(), e);
                }
                Err(e) => {
                    let _ = job.reply.send(Msg::Error(format!("load engine {:?}: {e}", job.engine)));
                    continue;
                }
            }
        }
        let engine = loaded.get_mut(&job.engine).unwrap();
        let t = Instant::now();
        let reply = job.reply.clone();
        let mut on_audio = |pcm: &[f32], seq: u32| {
            let mut bytes = Vec::with_capacity(pcm.len() * 4);
            for &s in pcm {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            let _ = reply.send(Msg::Audio { pcm_b64: base64::encode(&bytes), seq });
        };
        match engine.run(&job.req, CHUNK_SAMPLES, &mut on_audio) {
            Ok(samples) => {
                let _ = job.reply.send(Msg::Done { samples, ms: t.elapsed().as_secs_f64() * 1e3 });
            }
            Err(e) => {
                let _ = job.reply.send(Msg::Error(e));
            }
        }
    }
}

/// Tiny helper: clone a cfg out of the registry (engines are re-loadable, but we
/// only load each once and keep it resident, so cloning the cfg is fine).
trait CfgRegistry {
    fn remove_or_clone(&self, name: &str) -> Option<EngineCfg>;
}
impl CfgRegistry for HashMap<String, EngineCfg> {
    fn remove_or_clone(&self, name: &str) -> Option<EngineCfg> {
        self.get(name).map(|c| EngineCfg {
            kind: c.kind,
            weights_dir: c.weights_dir.clone(),
            ckpt_dir: c.ckpt_dir.clone(),
            npu_cache: c.npu_cache.clone(),
            device: c.device,
            cap: c.cap,
            quant: c.quant,
            ref_wav: c.ref_wav.clone(),
            ref_text: c.ref_text.clone(),
        })
    }
}

/// One client connection: read JSONL requests, enqueue jobs, stream responses.
fn handle_conn(stream: UnixStream, jobs: Sender<Job>) {
    let mut w = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            _ => continue,
        };
        let (engine, req) = match parse_request(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(w, "{}", err_line(&e));
                continue;
            }
        };
        let (tx, rx) = channel();
        if jobs.send(Job { engine, req, reply: tx }).is_err() {
            let _ = writeln!(w, "{}", err_line("server shutting down"));
            return;
        }
        for msg in rx {
            let out = match msg {
                Msg::Audio { pcm_b64, seq } => audio_line(&pcm_b64, seq, false),
                Msg::Done { samples, ms } => {
                    eprintln!("tts serve: served {samples} samples in {ms:.0}ms");
                    audio_line("", u32::MAX, true)
                }
                Msg::Error(e) => err_line(&e),
            };
            if writeln!(w, "{out}").is_err() {
                return;
            }
            let _ = w.flush();
        }
    }
}

fn parse_request(line: &str) -> Result<(String, Req), String> {
    let v: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("bad JSON: {e}"))?;
    let engine = v["engine"].as_str().unwrap_or("clone").to_string();
    let text = v["text"].as_str().unwrap_or_default().to_string();
    if text.trim().is_empty() {
        return Err("request missing non-empty \"text\"".into());
    }
    let mut opts = GenOpts::default();
    if let Some(x) = v["temp"].as_f64() {
        opts.temperature = x as f32;
    }
    if let Some(x) = v["top_k"].as_u64() {
        opts.top_k = x as usize;
    }
    if let Some(x) = v["seed"].as_u64() {
        opts.seed = x;
    }
    if let Some(x) = v["max_frames"].as_u64() {
        opts.max_frames = x as usize;
    }
    let req = Req {
        text,
        instruct: v["instruct"].as_str().unwrap_or_default().to_string(),
        speaker: v["speaker"].as_str().map(|s| s.to_string()),
        lang: v["lang"].as_str().unwrap_or("english").to_string(),
        opts,
    };
    Ok((engine, req))
}

fn audio_line(pcm_b64: &str, seq: u32, done: bool) -> String {
    serde_json::json!({
        "event": "audio_chunk", "pcm_b64": pcm_b64,
        "sample_rate": SAMPLE_RATE, "seq": seq, "done": done,
    })
    .to_string()
}

fn err_line(msg: &str) -> String {
    serde_json::json!({"event": "error", "message": msg}).to_string()
}
