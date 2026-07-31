// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Resident adapters that put the ASR models (Nemotron 3.5 ASR, Qwen3-ASR) behind
//! the residency [`Executor`], so `brain serve --dbus` schedules, batches and swaps
//! them exactly like every other model.
//!
//! Both build their model ONCE in `activate` (weights uploaded once) and hold it for
//! the instance's life (dropping frees the RAM). Nemotron — the streaming model —
//! implements a **true batched** `run_batch`: concurrent stream-windows with the
//! same language prompt encode in one FastConformer forward
//! ([`nemotron::encoder::Encoder::transcribe_batch`]). Qwen3-ASR is offline and
//! autoregressive, so its `run_batch` is sequential over a build-once, fixed-window
//! instance (the audio encoder still amortises across the batch).
//!
//! Config is env-only (each `from_env` returns `None`/skips when unset):
//!   * `BRAIN_NEMOTRON`      — Nemotron 3.5 ASR checkpoint dir.
//!   * `BRAIN_QWEN_ASR`      — Qwen3-ASR checkpoint dir.
//!   * `BRAIN_QWEN_ASR_WINDOW` — Qwen3-ASR audio window in seconds (default 30).
//!   * `BRAIN_QWEN_ASR_MAXNEW` — Qwen3-ASR max generated tokens (default 200).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};

use audio::asr_caps::{transcription_outcome, wav_from_blob};
use capability::{ActionResult, Invocation, Manifest, Progress};
use npu::openvino::{Feed, NpuConfig, NpuDevice, NpuGraph};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

/// OpenVINO config for the ASR encoders: prefer the NPU, fall back to CPU-OpenVINO
/// where no NPU is present (the graph is device-precision-independent).
fn asr_npu_cfg() -> NpuConfig {
    NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() }
}

// ---------------------------------------------------------------- Nemotron

/// Nemotron 3.5 ASR Streaming (0.6B) behind the scheduler. CPU-resident (the encoder
/// runs on a device-resolved `Gpu`; on this box that is CPU/Vulkan). One action,
/// `transcribe`, with a truly batched forward across concurrent same-prompt streams.
pub struct NemotronResident {
    dir: String,
}

impl NemotronResident {
    pub fn from_env() -> Option<NemotronResident> {
        std::env::var("BRAIN_NEMOTRON").ok().filter(|p| !p.is_empty()).map(|dir| NemotronResident { dir })
    }
}

impl ResidentModel for NemotronResident {
    fn manifest(&self) -> Manifest {
        nemotron::caps::manifest()
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(nemotron::caps::MODEL, "default")
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // 0.6B f32 weights (~2.4 GB) + activation scratch, held in RAM. On the NPU
        // path the raw weights are also held for the ONNX export, so advertise more
        // RAM there; `with_npu` lets the scheduler place the FastConformer encoder
        // on the Intel NPU (host RNN-T decode stays on the device backend).
        MemCost::new(0, 6u64 << 30).with_npu(2u64 << 30)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let cfg = nemotron::NemotronConfig::nemotron_3_5_asr_0_6b();
        let model = nemotron::model::NemotronAsr::from_hf(&self.dir, cfg.clone())?;
        let detok = nemotron::tokenizer::Detokenizer::from_hf(&self.dir)?;
        if let Device::Npu(_) = device {
            return Ok(Box::new(NemotronNpuInstance::new(&self.dir, cfg, model, detok)?));
        }
        Ok(Box::new(NemotronInstance { model, detok, sessions: nemotron::caps::StreamSessions::new() }))
    }
}

/// Nemotron on the NPU: the FastConformer encoder ONNX (mel → pooler) is compiled
/// on OpenVINO and run on the Intel NPU; the host frontend feeds it and the RNN-T
/// greedy decode stays on the device backend. Bit-identical to the device path
/// (the encoder graph is parity-gated to cosine 1.0 vs `nemotron::encoder::encode`).
struct NemotronNpuInstance {
    model: nemotron::model::NemotronAsr, // device backend: RNN-T decode
    detok: nemotron::tokenizer::Detokenizer,
    weights: HashMap<String, Vec<f32>>, // raw HF tensors, for the ONNX export
    topo: npu::NemotronTopo,
    /// One compiled encoder graph per fixed `(mel_t, mel_valid, prompt_id)`.
    graphs: RefCell<HashMap<(u32, u32, usize), NpuGraph>>,
    tmp: std::path::PathBuf,
    device: RefCell<String>,
}

impl NemotronNpuInstance {
    fn new(dir: &str, cfg: nemotron::NemotronConfig, model: nemotron::model::NemotronAsr, detok: nemotron::tokenizer::Detokenizer) -> Result<NemotronNpuInstance, String> {
        let tensors = checkpoint::safetensors::read_model_dir(std::path::Path::new(dir))?;
        let weights: HashMap<String, Vec<f32>> = tensors.into_iter().map(|t| (t.name, t.data)).collect();
        let topo = npu::NemotronTopo {
            num_mel_bins: cfg.num_mel_bins,
            hidden: cfg.hidden,
            subsampling_channels: cfg.subsampling_channels,
            subsampling_kernel: cfg.subsampling_kernel,
            subsampling_stride: cfg.subsampling_stride,
            subsampling_stages: cfg.subsampling_stages(),
            n_layers: cfg.n_layers,
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim(),
            intermediate: cfg.intermediate,
            conv_kernel: cfg.conv_kernel,
            left_ctx: cfg.sliding_window - 1,
            right_ctx: cfg.default_lookahead,
            ln_eps: cfg.ln_eps,
            num_prompts: cfg.num_prompts,
            prompt_intermediate: cfg.prompt_intermediate,
            decoder_hidden: cfg.decoder_hidden,
        };
        let tmp = std::env::temp_dir().join(format!("brain_nemotron_npu_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).map_err(|e| format!("tmp dir: {e}"))?;
        Ok(NemotronNpuInstance { model, detok, weights, topo, graphs: RefCell::new(HashMap::new()), tmp, device: RefCell::new("npu".into()) })
    }

    /// Run the FastConformer encoder ONNX on the NPU: `mel[t·num_mel]` → pooler
    /// `[T'·decoder_hidden]`; returns `(pooler, valid_T')`. Compiles (and caches)
    /// one static graph per `(mel_t, mel_valid, prompt_id)`.
    fn npu_encode(&self, mel: &[f32], t: u32, mel_valid: u32, prompt_id: usize) -> (Vec<f32>, u32) {
        let key = (t, mel_valid, prompt_id);
        if !self.graphs.borrow().contains_key(&key) {
            let path = self.tmp.join(format!("enc_{t}_{mel_valid}_{prompt_id}.onnx"));
            let ps = path.to_str().expect("utf8 tmp path");
            npu::nemotron_export::export(&self.weights, &self.topo, t, mel_valid, prompt_id as u32, ps).expect("export nemotron encoder ONNX");
            let graph = NpuGraph::compile_path(&path, &asr_npu_cfg()).expect("compile nemotron encoder on NPU");
            *self.device.borrow_mut() = graph.device().to_string();
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(format!("{ps}.data")); // external-data sidecar
            self.graphs.borrow_mut().insert(key, graph);
        }
        let mut g = self.graphs.borrow_mut();
        let graph = g.get_mut(&key).unwrap();
        let nmel = self.topo.num_mel_bins as i64;
        let out = graph.run(&[("mel", Feed::F32(mel, vec![1, 1, t as i64, nmel]))]).expect("nemotron encoder NPU infer");
        let pooler = out.into_iter().find(|(n, _, _)| n == "pooler").expect("pooler output").2;
        (pooler, self.topo.subsampled_len(mel_valid))
    }
}

impl Instance for NemotronNpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let blob = inv.get_blob("audio").ok_or("nemotron transcribe: missing 'audio' input")?;
        let wav = wav_from_blob(blob)?;
        let prompt_id = inv.get_i64("prompt_id").unwrap_or(0).max(0) as usize;
        progress(Progress { step: 0, total: 1, message: "transcribing (NPU encoder)".into() });
        let toks = self.model.encoder().transcribe_with_encoder(&wav, prompt_id, |mel, t, mv, pid| self.npu_encode(mel, t, mv, pid));
        let text = self.detok.decode(&toks);
        progress(Progress { step: 1, total: 1, message: text.clone() });
        let mut out = transcription_outcome(text, &toks);
        out = out.set("device", serde_json::json!(self.device.borrow().clone()));
        Ok(out)
    }
    // run_batch: the default sequential loop (the NPU encoder graph is per-utterance
    // fixed-shape, so batching happens across requests via the scheduler, not in-graph).
}

struct NemotronInstance {
    model: nemotron::model::NemotronAsr,
    detok: nemotron::tokenizer::Detokenizer,
    /// Live `transcribe_stream` sessions. They live on the instance, so evicting
    /// the model (residency swap) drops any in-flight streams — a restarted stream
    /// id simply begins a fresh session.
    sessions: nemotron::caps::StreamSessions,
}

impl Instance for NemotronInstance {
    fn run(&mut self, action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        self.run_batch(action, std::slice::from_ref(inv), progress).pop().unwrap()
    }

    /// TRUE batched forward, for both actions (the executor groups jobs per action).
    /// `transcribe`: group by `prompt_id`, one `transcribe_batch` per group (one
    /// FastConformer forward over all of that group's windows). `transcribe_stream`:
    /// one batched encoder step over every concurrent stream's window
    /// (`StreamSessions::step_batch`). Per-job decode errors stay per-job.
    fn run_batch(&mut self, action: &str, invs: &[Invocation], progress: &mut dyn FnMut(Progress)) -> Vec<ActionResult> {
        if action != "transcribe_stream" {
            return self.offline_batch(invs, progress);
        }
        let mut results: Vec<Option<ActionResult>> = vec![None; invs.len()];
        let mut jobs: Vec<(usize, (String, Vec<f32>, usize, bool))> = Vec::new();
        for (i, inv) in invs.iter().enumerate() {
            match nemotron::caps::stream_job_from_inv(inv) {
                Ok(j) => jobs.push((i, j)),
                Err(e) => results[i] = Some(Err(e)),
            }
        }
        progress(Progress { step: 0, total: 1, message: format!("stepping {} stream(s)", jobs.len()) });
        let refs: Vec<(&str, &[f32], usize, bool)> = jobs.iter().map(|(_, (id, w, p, e))| (id.as_str(), w.as_slice(), *p, *e)).collect();
        let outs = self.sessions.step_batch(self.model.encoder(), &self.detok, &refs);
        for ((i, _), o) in jobs.into_iter().zip(outs) {
            results[i] = Some(Ok(o));
        }
        results.into_iter().map(|r| r.unwrap()).collect()
    }
}

impl NemotronInstance {
    /// The offline `transcribe` batch path (whole utterances, prompt-grouped).
    fn offline_batch(&mut self, invs: &[Invocation], progress: &mut dyn FnMut(Progress)) -> Vec<ActionResult> {
        // 1. decode audio + prompt_id per job (errors recorded, don't abort the batch).
        let mut wavs: Vec<Vec<f32>> = Vec::with_capacity(invs.len());
        let mut pids: Vec<usize> = Vec::with_capacity(invs.len());
        let mut errs: Vec<Option<String>> = Vec::with_capacity(invs.len());
        for inv in invs {
            match inv.get_blob("audio").ok_or_else(|| "nemotron transcribe: missing 'audio' input".to_string()).and_then(wav_from_blob) {
                Ok(w) => {
                    pids.push(inv.get_i64("prompt_id").unwrap_or(0).max(0) as usize);
                    wavs.push(w);
                    errs.push(None);
                }
                Err(e) => {
                    pids.push(0);
                    wavs.push(Vec::new());
                    errs.push(Some(e));
                }
            }
        }
        progress(Progress { step: 0, total: 1, message: format!("transcribing {} stream(s)", invs.len()) });

        // 2. group valid jobs by prompt_id; batch-encode each group.
        let mut tokens_by_job: Vec<Option<Vec<u32>>> = vec![None; invs.len()];
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, e) in errs.iter().enumerate() {
            if e.is_none() {
                groups.entry(pids[i]).or_default().push(i);
            }
        }
        for (pid, idxs) in groups {
            let refs: Vec<&[f32]> = idxs.iter().map(|&i| wavs[i].as_slice()).collect();
            let batch = self.model.encoder().transcribe_batch(&refs, pid);
            for (slot, toks) in idxs.into_iter().zip(batch) {
                tokens_by_job[slot] = Some(toks);
            }
        }

        // 3. detokenize + pack per job, in order.
        invs.iter()
            .enumerate()
            .map(|(i, _)| match (&errs[i], tokens_by_job[i].take()) {
                (Some(e), _) => Err(e.clone()),
                (None, Some(toks)) => {
                    let text = self.detok.decode(&toks);
                    Ok(transcription_outcome(text, &toks))
                }
                (None, None) => Err("nemotron transcribe: internal batching error".to_string()),
            })
            .collect()
    }
}

// ---------------------------------------------------------------- Qwen3-ASR

/// Qwen3-ASR (1.7B) behind the scheduler — offline, fixed audio window. CPU-resident.
pub struct QwenAsrResident {
    dir: String,
    window_secs: f32,
    max_new: usize,
}

impl QwenAsrResident {
    pub fn from_env() -> Option<QwenAsrResident> {
        let dir = std::env::var("BRAIN_QWEN_ASR").ok().filter(|p| !p.is_empty())?;
        let window_secs = std::env::var("BRAIN_QWEN_ASR_WINDOW").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0f32);
        let max_new = std::env::var("BRAIN_QWEN_ASR_MAXNEW").ok().and_then(|s| s.parse().ok()).unwrap_or(200usize);
        Some(QwenAsrResident { dir, window_secs, max_new })
    }
}

impl ResidentModel for QwenAsrResident {
    fn manifest(&self) -> Manifest {
        qwen_asr::caps::manifest()
    }
    fn instance_key(&self, _action: &str, _inv: &Invocation) -> InstanceKey {
        InstanceKey::new(qwen_asr::caps::MODEL, format!("w{}", self.window_secs))
    }
    fn estimate(&self, _key: &InstanceKey) -> MemCost {
        // 1.7B decoder (~7 GB f32) + audio tower (~2 GB), held in RAM. `with_npu`
        // lets the scheduler place the audio-encoder head on the NPU (the conv stem
        // + the Qwen decoder stay on the device backend).
        MemCost::new(0, 10u64 << 30).with_npu(2u64 << 30)
    }
    fn activate(&self, _key: &InstanceKey, device: Device) -> Result<Box<dyn Instance>, String> {
        let cfg = qwen_asr::config::QwenAsrConfig::qwen3_asr_1_7b();
        let provider = qwen_asr::caps::QwenAsrProvider::load(&self.dir, cfg.clone(), self.window_secs, self.max_new)?;
        if let Device::Npu(_) = device {
            return Ok(Box::new(QwenAsrNpuInstance::new(&self.dir, cfg, provider)?));
        }
        Ok(Box::new(QwenAsrInstance { provider }))
    }
}

struct QwenAsrInstance {
    provider: qwen_asr::caps::QwenAsrProvider,
}

impl Instance for QwenAsrInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let blob = inv.get_blob("audio").ok_or("qwen-asr transcribe: missing 'audio' input")?;
        let wav = wav_from_blob(blob)?;
        progress(Progress { step: 0, total: 1, message: "transcribing".into() });
        let (text, tokens) = self.provider.transcribe(&wav)?;
        progress(Progress { step: 1, total: 1, message: text.clone() });
        Ok(transcription_outcome(text, &tokens))
    }
    // run_batch: default sequential loop (offline autoregressive decode; the audio
    // encoder still amortises within the fixed-window instance).
}

/// Qwen3-ASR on the NPU: the audio-tower transformer HEAD (24 windowed ViT blocks
/// + ln_post + projector) runs as an ONNX graph on the Intel NPU; the conv stem +
/// valid-position packing and the Qwen decoder stay on the device backend.
/// Bit-identical to the device path (the head graph is parity-gated to cosine 1.0
/// vs `qwen_asr::encoder::encode_packed`).
struct QwenAsrNpuInstance {
    provider: qwen_asr::caps::QwenAsrProvider,
    weights: HashMap<String, Vec<f32>>, // audio-encoder tensors, for the head ONNX
    topo: npu::QwenAsrTopo,
    /// One compiled head graph per `(n_audio, spans)` (constant for a full window).
    graphs: RefCell<HashMap<String, NpuGraph>>,
    device: RefCell<String>,
}

impl QwenAsrNpuInstance {
    fn new(dir: &str, cfg: qwen_asr::config::QwenAsrConfig, provider: qwen_asr::caps::QwenAsrProvider) -> Result<QwenAsrNpuInstance, String> {
        let weights = qwen_asr::import::load_audio_encoder(std::path::Path::new(dir), &cfg.audio)?;
        let a = &cfg.audio;
        let topo = npu::QwenAsrTopo {
            d_model: a.d_model,
            n_heads: a.n_heads,
            head_dim: a.head_dim(),
            ffn_dim: a.ffn_dim,
            n_layers: a.n_layers,
            output_dim: a.output_dim,
            eps: a.eps,
        };
        Ok(QwenAsrNpuInstance { provider, weights, topo, graphs: RefCell::new(HashMap::new()), device: RefCell::new("npu".into()) })
    }

    /// Run the audio-encoder head ONNX on the NPU: `packed[n_audio·d] + spans →
    /// audio_embeds[n_audio·output_dim]`. Compiles/caches one graph per `(n_audio,
    /// spans)`. Returns `(_, embeds)` — only the embeds are used downstream.
    fn npu_head(&self, packed: &[f32], n_audio: u32, spans: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
        let (d, out) = (self.topo.d_model, self.topo.output_dim);
        let key = format!("{n_audio}:{spans:?}");
        if !self.graphs.borrow().contains_key(&key) {
            let mut g = onnx::GraphBuilder::new("qwen_asr_head");
            g.input_f32("x", &[n_audio as i64, d as i64]);
            npu::build_qwen_asr_head(&mut g, &self.topo, &self.weights, n_audio, spans, "x", "embeds");
            g.output_f32("embeds", &[n_audio as i64, out as i64]);
            let graph = NpuGraph::compile_bytes(&g.finish(), &asr_npu_cfg()).expect("compile qwen-asr head on NPU");
            *self.device.borrow_mut() = graph.device().to_string();
            self.graphs.borrow_mut().insert(key.clone(), graph);
        }
        let mut gm = self.graphs.borrow_mut();
        let graph = gm.get_mut(&key).unwrap();
        let o = graph.run(&[("x", Feed::F32(packed, vec![n_audio as i64, d as i64]))]).expect("qwen-asr head NPU infer");
        let embeds = o.into_iter().find(|(n, _, _)| n == "embeds").expect("embeds output").2;
        (Vec::new(), embeds)
    }
}

impl Instance for QwenAsrNpuInstance {
    fn run(&mut self, _action: &str, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let blob = inv.get_blob("audio").ok_or("qwen-asr transcribe: missing 'audio' input")?;
        let wav = wav_from_blob(blob)?;
        progress(Progress { step: 0, total: 1, message: "transcribing (NPU audio encoder)".into() });
        let (text, tokens) = self.provider.transcribe_with_head(&wav, |packed, n, spans| self.npu_head(packed, n, spans))?;
        progress(Progress { step: 1, total: 1, message: text.clone() });
        Ok(transcription_outcome(text, &tokens).set("device", serde_json::json!(self.device.borrow().clone())))
    }
}
