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

use std::collections::BTreeMap;

use audio::asr_caps::{transcription_outcome, wav_from_blob};
use capability::{ActionResult, Invocation, Manifest, Progress};
use residency::{Device, Instance, InstanceKey, MemCost, ResidentModel};

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
        // 0.6B f32 weights (~2.4 GB) + activation scratch, held in RAM.
        MemCost::new(0, 4u64 << 30)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        let cfg = nemotron::NemotronConfig::nemotron_3_5_asr_0_6b();
        let model = nemotron::model::NemotronAsr::from_hf(&self.dir, cfg)?;
        let detok = nemotron::tokenizer::Detokenizer::from_hf(&self.dir)?;
        Ok(Box::new(NemotronInstance { model, detok, sessions: nemotron::caps::StreamSessions::new() }))
    }
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
        // 1.7B decoder (~7 GB f32) + audio tower (~2 GB), held in RAM.
        MemCost::new(0, 10u64 << 30)
    }
    fn activate(&self, _key: &InstanceKey, _device: Device) -> Result<Box<dyn Instance>, String> {
        let cfg = qwen_asr::config::QwenAsrConfig::qwen3_asr_1_7b();
        let provider = qwen_asr::caps::QwenAsrProvider::load(&self.dir, cfg, self.window_secs, self.max_new)?;
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
