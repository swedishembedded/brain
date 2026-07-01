// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident TTS **engine** for server mode: loads the model weights + compiles
//! the NPU KV-cache graphs ONCE and serves many requests, so the ~tens-of-seconds
//! graph compile/load is paid a single time instead of per request. Holds the
//! Talker KV decoder (`KvTalker`), the CPU MTP, the host tables, and codec decode
//! sessions (cached per code-length bucket) all resident; each request resets the
//! cache, builds its prompt, generates, and streams PCM via a callback.
//!
//! Used by `brain tts serve` (one engine per voice/mode, owned by a dedicated
//! executor thread — OpenVINO infer requests are not shared across threads).

use std::collections::HashMap;
use std::path::Path;

use codec::decode_stream::StreamingCodecDecoder;
use data::tokenizer::Tokenizer;
use npu::openvino::{CodecSession, NpuDevice};

use crate::gen_kv_mtp::CpuMtp;
use crate::npu_gen::{
    codec_bucket, decode_with_session, generate_codes_kv, generate_codes_kv_streaming, open_codec_session, KvTalker,
    NpuStreamCodec, TalkerTables,
};
use crate::pipeline::{self, GenOpts};
use crate::prompt::{self, TtsSpecials};

/// What kind of synthesis an engine serves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// In-context voice clone of a fixed reference voice (text varies per request).
    Clone,
    /// VoiceDesign / CustomVoice: per-request `instruct` (+ optional preset speaker).
    Design,
    /// Speaker-free text-to-speech.
    Synth,
}

/// Engine configuration (resident model + NPU placement).
pub struct EngineCfg {
    pub kind: Kind,
    pub weights_dir: String,
    pub ckpt_dir: String,
    pub npu_cache: String,
    pub device: NpuDevice,
    /// Fixed compiled KV context length (prefix + frames). Requests are clamped
    /// to fit; pick generously for the mode (e.g. clone ~384, design ~256).
    pub cap: usize,
    pub quant: bool,
    /// Use the INT4 weight-compressed Talker decode graph (opt-in). On a device
    /// whose native max is INT8 (the Intel NPU) this is weight-compression, not
    /// native 4-bit — still ~20% faster on the bandwidth-bound Talker, half the
    /// graph RAM. Forces the prefill graph off (compiling both i4 graphs OOMs), so
    /// a long clone prefix seeds token-by-token (one-time cost).
    pub int4: bool,
    /// Clone only: the reference voice and its transcript (encoded once at load).
    pub ref_wav: Option<String>,
    pub ref_text: Option<String>,
}

/// A single synthesis request handed to a resident engine.
pub struct Req {
    pub text: String,
    pub instruct: String,
    pub speaker: Option<String>,
    pub lang: String,
    pub opts: GenOpts,
}

/// A loaded, resident engine. One per voice/mode; owned by its executor thread.
pub struct TtsEngine {
    cfg: EngineCfg,
    sp: TtsSpecials,
    tok: data::qwen_tokenizer::QwenBpe,
    tables: TalkerTables,
    mtp: CpuMtp,
    // Resident NPU MTP (INT8 KV-cache decode graph). When present it replaces the
    // host `CpuMtp` in the generation loop — ~87ms/frame vs ~580ms on the large
    // 1.7B (the host MTP re-streams its ~300MB fp32 weights 16x/frame and is
    // memory-bandwidth bound). Loaded for d_model>=2048 unless `BRAIN_TTS_MTP=cpu`.
    mtp_npu: Option<crate::npu_gen::KvMtp>,
    kv: KvTalker,
    codec_sessions: HashMap<usize, CodecSession>,
    // Pure-CPU stateful streaming codec (BRAIN_TTS_CODEC=cpu-stream); lazily loaded.
    cpu_codec: Option<StreamingCodecDecoder>,
    // NPU stateful streaming codec (BRAIN_TTS_CODEC=npu-stream); lazily loaded.
    npu_codec: Option<NpuStreamCodec>,
    // Clone-mode, encoded once at load:
    ref_code: Option<Vec<u32>>,
    ref_ids: Option<Vec<u32>>,
    xvec: Option<Vec<f32>>,
}

impl TtsEngine {
    /// Load the model + compile the resident KV graphs (the slow one-time step).
    pub fn load(cfg: EngineCfg) -> Result<TtsEngine, String> {
        let sp = TtsSpecials::from_config_dir(&cfg.ckpt_dir)?;
        let tok = prompt::load_tokenizer(&cfg.ckpt_dir)?;
        let talker = format!("{}/talker.weights", cfg.weights_dir);
        let mtp_path = format!("{}/mtp.weights", cfg.weights_dir);
        let tables = TalkerTables::load(&talker);
        let mtp = CpuMtp::load(&mtp_path);
        let cache = Path::new(&cfg.npu_cache);
        // Print the resolved hardware path (device + weight precision + native vs
        // weight-compression) so `brain tts serve` startup logs what it actually uses.
        eprintln!("{}", crate::npu_gen::describe_talker_path(cfg.device, true, cfg.quant, cfg.int4));
        // Only clone (long reference prefix) benefits from the prefill graph;
        // design/cv/synth have short prefixes, so skip its ~1.4 GB compile. INT4 also
        // skips prefill (compiling both i4 graphs OOMs) — prefix seeds token-by-token.
        let with_prefill = cfg.kind == Kind::Clone && !cfg.int4;
        let kv = KvTalker::load(&talker, cfg.cap, cfg.device, true, Some(cache), &tables.cfg, cfg.quant, cfg.int4, with_prefill)?;

        // MTP placement (mirrors `pipeline::run_npu`): the resident INT8 NPU decode
        // graph beats the host CpuMtp on the large model. Default on for d_model>=2048;
        // `BRAIN_TTS_MTP=cpu` forces the host path, `=npu` forces it on.
        let want_npu_mtp = match std::env::var("BRAIN_TTS_MTP").ok().as_deref() {
            Some("cpu") => false,
            Some("npu") => true,
            _ => tables.cfg.d_model >= 2048,
        };
        let mtp_npu = if want_npu_mtp {
            match crate::npu_gen::KvMtp::load(&mtp_path, cfg.device, true, Some(cache), tables.cfg.d_model >= 2048) {
                Ok(k) => Some(k),
                Err(e) => {
                    eprintln!("tts serve: NPU MTP unavailable ({e}); falling back to host CpuMtp");
                    None
                }
            }
        } else {
            None
        };

        let (ref_code, ref_ids, xvec) = if cfg.kind == Kind::Clone {
            let rw = cfg.ref_wav.as_ref().ok_or("clone engine needs a reference wav")?;
            let rt = cfg.ref_text.clone().unwrap_or_default();
            let wav = audio::wav::read(rw).map_err(|e| format!("read {rw}: {e}"))?;
            let speaker = speaker::SpeakerEncoder::load_inference(&format!("{}/speaker.weights", cfg.weights_dir));
            let xvec = speaker.embed_wav(&wav.samples, wav.sample_rate);
            let codec_path = format!("{}/codec.weights", cfg.weights_dir);
            let ref_code = pipeline::ref_codes_cached(&codec_path, &wav, rw, Some(&cfg.npu_cache));
            let ref_ids_full = tok.encode(&format!("<|im_start|>assistant\n{rt}<|im_end|>\n"));
            if ref_ids_full.len() < 6 {
                return Err("clone reference transcript tokenized too short".into());
            }
            let ref_ids = ref_ids_full[3..ref_ids_full.len() - 2].to_vec();
            (Some(ref_code), Some(ref_ids), Some(xvec))
        } else {
            (None, None, None)
        };

        Ok(TtsEngine {
            cfg,
            sp,
            tok,
            tables,
            mtp,
            mtp_npu,
            kv,
            codec_sessions: HashMap::new(),
            cpu_codec: None,
            npu_codec: None,
            ref_code,
            ref_ids,
            xvec,
        })
    }

    pub fn kind(&self) -> Kind {
        self.cfg.kind
    }
    pub fn device(&self) -> &str {
        self.kv.device()
    }

    /// Serve one request: build the prompt, then generate + decode + stream
    /// progressively — a sliding `win`-frame codec window is decoded every `chunk`
    /// frames and only the newest frames' audio is emitted via `on_audio`, so
    /// playback can start after the first chunk instead of after the whole clip.
    pub fn run(&mut self, req: &Req, on_audio: &mut dyn FnMut(&[f32], u32)) -> Result<usize, String> {
        let lang_id = self.sp.language_id(&req.lang);
        let input_ids = self.tok.encode(&pipeline::assistant_text(&req.text));
        let (role_ids, text_ids) = pipeline::split_input_ids(&input_ids)?;

        let prompt = match self.cfg.kind {
            Kind::Clone => {
                let ref_code = self.ref_code.as_ref().ok_or("clone engine missing ref codes")?;
                let ref_ids = self.ref_ids.as_ref().unwrap();
                let xvec = self.xvec.as_ref().unwrap();
                prompt::build_icl_prompt(
                    &self.tables, &self.mtp, &self.sp, &role_ids, &text_ids, ref_ids, ref_code, xvec, lang_id,
                )
            }
            Kind::Design => {
                let instruct_ids = if req.instruct.trim().is_empty() {
                    Vec::new()
                } else {
                    self.tok.encode(&pipeline::instruct_text(&req.instruct))
                };
                let speaker_id = pipeline::resolve_speaker(&self.sp, req.speaker.as_deref())?;
                prompt::build_instruct_prompt(
                    &self.tables, &self.sp, &role_ids, &text_ids, &instruct_ids, speaker_id, lang_id,
                )
            }
            Kind::Synth => prompt::build_xvector_prompt(&self.tables, &self.sp, &role_ids, &text_ids, None, lang_id),
        };

        // Clamp the request to the resident graph capacity (cap = prefix + frames).
        let prefix = prompt.embeds.len() / self.tables.d();
        let fit = self.kv.cap().saturating_sub(prefix + 2);
        if fit == 0 {
            return Err(format!(
                "prompt prefix {prefix} leaves no room in compiled cap {} (raise --cap)",
                self.kv.cap()
            ));
        }
        let mut opts = req.opts.clone();
        if opts.max_frames > fit {
            opts.max_frames = fit;
        }

        // Codec backend `cpu-stream`: the pure-CPU *stateful* streaming decoder —
        // each chunk decodes ONLY its new frames (no warmup re-decode), emitting
        // audio progressively. Generate the full codes, then stream-decode.
        if std::env::var("BRAIN_TTS_CODEC").map(|v| v == "cpu-stream").unwrap_or(false) {
            let mtp_eng: &mut dyn crate::npu_gen::MtpEngine =
                match self.mtp_npu.as_mut() { Some(k) => k, None => &mut self.mtp };
            let codes = generate_codes_kv(&mut self.kv, &self.tables, mtp_eng, &self.sp, &prompt, &opts)?;
            if codes.is_empty() {
                return Err("no codec frames were generated".into());
            }
            if self.cpu_codec.is_none() {
                let codec_path = format!("{}/codec.weights", self.cfg.weights_dir);
                self.cpu_codec = Some(StreamingCodecDecoder::load(&codec_path));
            }
            let chunk = std::env::var("BRAIN_TTS_STREAM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize).max(1);
            let dec = self.cpu_codec.as_ref().unwrap();
            let mut total = 0usize;
            dec.decode_streaming_cb(&codes, chunk, &mut |pcm, seq| {
                on_audio(pcm, seq);
                total += pcm.len();
            });
            if total == 0 {
                return Err("no audio produced".into());
            }
            return Ok(total);
        }

        // Codec backend `npu-stream` (DEFAULT): the NPU *stateful* streaming
        // decoder — front graph once, then the streaming-back graph per chunk
        // carrying per-conv state (no warmup re-decode). Exact and ~faster than the
        // windowed path. `BRAIN_TTS_CODEC=windowed` forces the old sliding-window
        // path below.
        if std::env::var("BRAIN_TTS_CODEC").map(|v| v != "windowed").unwrap_or(true) {
            let mtp_eng: &mut dyn crate::npu_gen::MtpEngine =
                match self.mtp_npu.as_mut() { Some(k) => k, None => &mut self.mtp };
            let codes = generate_codes_kv(&mut self.kv, &self.tables, mtp_eng, &self.sp, &prompt, &opts)?;
            if codes.is_empty() {
                return Err("no codec frames were generated".into());
            }
            if self.npu_codec.is_none() {
                let codec_path = format!("{}/codec.weights", self.cfg.weights_dir);
                let front_t = self.kv.cap();
                let chunk = std::env::var("BRAIN_TTS_STREAM_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize).max(1);
                self.npu_codec = Some(NpuStreamCodec::load(
                    &codec_path,
                    front_t,
                    chunk,
                    self.cfg.device,
                    true,
                    Some(Path::new(&self.cfg.npu_cache)),
                )?);
            }
            let total = self.npu_codec.as_mut().unwrap().decode(&codes, on_audio)?;
            if total == 0 {
                return Err("no audio produced".into());
            }
            return Ok(total);
        }

        // Sliding-window streaming codec. `win` is the decoded window (warmup +
        // chunk) — its leading frames give the causal codec context; only the
        // newest `chunk` frames' audio is emitted. Gaps between chunks are fine
        // (fast hardware keeps up). One resident codec session at length `win`.
        let envn = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let chunk = envn("BRAIN_TTS_STREAM_CHUNK", 16).max(1);
        let win = codec_bucket(envn("BRAIN_TTS_STREAM_WIN", 32).max(chunk));
        if !self.codec_sessions.contains_key(&win) {
            let codec_path = format!("{}/codec.weights", self.cfg.weights_dir);
            let s = open_codec_session(&codec_path, win, self.cfg.device, true, Some(Path::new(&self.cfg.npu_cache)))?;
            self.codec_sessions.insert(win, s);
        }
        let sess = self.codec_sessions.get_mut(&win).unwrap();

        let mut emitted = 0usize;
        let mut seq = 0u32;
        let mut total_samples = 0usize;
        let mut cb_err: Option<String> = None;
        {
            let mut on_chunk = |all: &[u32]| {
                if cb_err.is_some() {
                    return;
                }
                let total = all.len() / 16;
                let new = total - emitted;
                if new == 0 {
                    return;
                }
                // Build the `win`-frame window: the last `win` real frames (the
                // newest at the end), left-zero-padded when fewer are available.
                let start = total.saturating_sub(win);
                let pad = win - (total - start);
                let mut wc = vec![0u32; win * 16];
                for (i, f) in (start..total).enumerate() {
                    let dst = (pad + i) * 16;
                    wc[dst..dst + 16].copy_from_slice(&all[f * 16..f * 16 + 16]);
                }
                match decode_with_session(sess, &wc) {
                    Ok(wav) => {
                        let spf = wav.len() / win.max(1);
                        let take = new.min(win) * spf;
                        on_audio(&wav[wav.len() - take..], seq);
                        seq += 1;
                        total_samples += take;
                    }
                    Err(e) => cb_err = Some(e),
                }
                emitted = total;
            };
            let mtp_eng: &mut dyn crate::npu_gen::MtpEngine =
                match self.mtp_npu.as_mut() { Some(k) => k, None => &mut self.mtp };
            generate_codes_kv_streaming(
                &mut self.kv, &self.tables, mtp_eng, &self.sp, &prompt, &opts, chunk, &mut on_chunk,
            )?;
        }
        if let Some(e) = cb_err {
            return Err(e);
        }
        if total_samples == 0 {
            return Err("no audio produced".into());
        }
        Ok(total_samples)
    }
}
