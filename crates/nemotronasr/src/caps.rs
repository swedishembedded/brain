// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `capability` surface for Nemotron 3.5 ASR — the shared pieces every serving
//! path reuses so transcription is exposed the one brain way (`brain do`, the
//! event API, D-Bus, the residency scheduler).
//!
//! The heavy model + tokenizer are held by the caller (the resident adapter builds
//! them once — see `cli::resident_asr`); this module owns the *contract*: the
//! [`ActionSpec`], the audio-blob decoding, and a thin [`Provider`] for the direct
//! `brain do` path. Audio arrives as **raw mono f32 little-endian PCM at 16 kHz**
//! (meta `{"sample_rate":16000}`) — the same convention the D-Bus fd transport and
//! the Python example use.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use audio::asr_caps::{transcribe_spec, transcribe_stream_spec, transcription_outcome, wav_from_blob};
use capability::{Action, ActionResult, ActionSpec, Invocation, Manifest, Outcome, Progress, Provider};

use crate::config::NemotronConfig;
use crate::encoder::Encoder;
use crate::model::NemotronAsr;
use crate::stream::StreamState;
use crate::tokenizer::Detokenizer;

/// The model name advertised in the manifest.
pub const MODEL: &str = "brain/nemotronasr";

/// The manifest: offline `transcribe` plus frame-synchronous `transcribe_stream`
/// (schemas shared via [`audio::asr_caps`]).
pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "NVIDIA Nemotron 3.5 ASR Streaming 0.6B — speech-to-text.", vec![transcribe_spec(), transcribe_stream_spec()])
}

/// Sessions idle longer than this are dropped on the next step (a client that
/// died without sending `eos` must not leak its ~12 MB state forever).
const SESSION_IDLE_SECS: u64 = 600;

struct StreamSession {
    st: StreamState,
    /// Byte length of the transcript already returned to the client — the decoded
    /// text grows monotonically, so each step returns `full[decoded..]`.
    decoded: usize,
    /// Tokens already returned to the client.
    tok_seen: usize,
    last_used: Instant,
}

/// Live `transcribe_stream` sessions for one serving instance (the resident
/// adapter and the direct `Provider` each own one). Maps session id → the
/// per-stream [`StreamState`]; created on first use, dropped on `eos` (or idle
/// timeout). One implementation for every serving path.
#[derive(Default)]
pub struct StreamSessions {
    sessions: HashMap<String, StreamSession>,
}

impl StreamSessions {
    pub fn new() -> StreamSessions {
        StreamSessions::default()
    }

    /// Step one window of one session (see [`step_batch`](Self::step_batch)).
    pub fn step(&mut self, enc: &Encoder, detok: &Detokenizer, id: &str, wav: &[f32], prompt_id: usize, eos: bool) -> Outcome {
        self.step_batch(enc, detok, &[(id, wav, prompt_id, eos)]).pop().unwrap()
    }

    /// Step one window of each of several sessions **concurrently**: all pushes run
    /// through one batched encoder step ([`Encoder::stream_push_batch`] — per-frame
    /// ops batched across streams), then `eos` sessions are flushed and dropped.
    /// Jobs sharing a session id within one batch are stepped in order (the later
    /// ones in a follow-up round). Returns one outcome per job, in order: the
    /// session's newly emitted text/tokens, with `"final": true` on the flush.
    pub fn step_batch(&mut self, enc: &Encoder, detok: &Detokenizer, jobs: &[(&str, &[f32], usize, bool)]) -> Vec<Outcome> {
        self.sessions.retain(|_, s| s.last_used.elapsed().as_secs() < SESSION_IDLE_SECS);
        let mut out: Vec<Option<Outcome>> = vec![None; jobs.len()];
        // rounds of unique session ids, so state stepping stays ordered per session
        let mut remaining: Vec<usize> = (0..jobs.len()).collect();
        while !remaining.is_empty() {
            let mut round: Vec<usize> = Vec::new();
            let mut later: Vec<usize> = Vec::new();
            for &j in &remaining {
                if round.iter().any(|&r| jobs[r].0 == jobs[j].0) {
                    later.push(j);
                } else {
                    round.push(j);
                }
            }
            // take the round's sessions out of the map, batch-push, put them back
            let mut taken: Vec<(usize, StreamSession)> = round
                .iter()
                .map(|&j| {
                    let (id, _, prompt_id, _) = jobs[j];
                    let sess = self.sessions.remove(id).unwrap_or_else(|| StreamSession {
                        st: enc.stream_new(prompt_id),
                        decoded: 0,
                        tok_seen: 0,
                        last_used: Instant::now(),
                    });
                    (j, sess)
                })
                .collect();
            {
                let mut items: Vec<(&mut StreamState, &[f32])> = taken.iter_mut().map(|(j, s)| (&mut s.st, jobs[*j].1)).collect();
                enc.stream_push_batch(&mut items);
            }
            for (j, mut sess) in taken {
                let (id, _, _, eos) = jobs[j];
                if eos {
                    enc.stream_finish(&mut sess.st);
                }
                let full = detok.decode(sess.st.tokens());
                let new_text = full.get(sess.decoded..).unwrap_or("").to_string();
                let new_tokens: Vec<u32> = sess.st.tokens()[sess.tok_seen..].to_vec();
                sess.decoded = full.len();
                sess.tok_seen = sess.st.tokens().len();
                sess.last_used = Instant::now();
                let o = transcription_outcome(new_text, &new_tokens)
                    .set("stream", serde_json::json!(id))
                    .set("final", serde_json::json!(eos));
                out[j] = Some(o);
                if !eos {
                    self.sessions.insert(id.to_string(), sess);
                }
            }
            remaining = later;
        }
        out.into_iter().map(|o| o.unwrap()).collect()
    }
}

/// Decode a `transcribe_stream` invocation: `(session id, window, prompt_id, eos)`.
/// The audio blob may be absent (a final `eos`-only flush).
pub fn stream_job_from_inv(inv: &Invocation) -> Result<(String, Vec<f32>, usize, bool), String> {
    let id = inv.get_str("stream").filter(|s| !s.is_empty()).ok_or("transcribe_stream: missing 'stream' session id")?;
    let wav = match inv.get_blob("audio") {
        Some(b) => wav_from_blob(b)?,
        None => Vec::new(),
    };
    let prompt_id = inv.get_i64("prompt_id").unwrap_or(0).max(0) as usize;
    let eos = inv.get_bool("eos").unwrap_or(false);
    Ok((id, wav, prompt_id, eos))
}

// ---------------------------------------------------------------- direct Provider

/// A loaded Nemotron model behind the `capability` interface for the direct
/// `brain do nemotron transcribe` / `Registry` path (the resident adapter has its
/// own build-once, batched instance and does not go through this). Streaming
/// sessions live on the provider, so `transcribe_stream` state persists across
/// event-API invocations within one process.
pub struct NemotronProvider {
    model: Arc<NemotronAsr>,
    detok: Arc<Detokenizer>,
    sessions: Arc<Mutex<StreamSessions>>,
}

impl NemotronProvider {
    /// Load an HF checkpoint dir (weights + `tokenizer.json`) and build the model on
    /// a device-resolved `Gpu` (`BRAIN_DEVICE`).
    pub fn load(dir: &str, cfg: NemotronConfig) -> Result<NemotronProvider, String> {
        let model = Arc::new(NemotronAsr::from_hf(dir, cfg)?);
        let detok = Arc::new(Detokenizer::from_hf(dir)?);
        Ok(NemotronProvider { model, detok, sessions: Arc::new(Mutex::new(StreamSessions::new())) })
    }
}

impl Provider for NemotronProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "transcribe" => Some(Arc::new(TranscribeAction { model: self.model.clone(), detok: self.detok.clone() }) as Arc<dyn Action>),
            "transcribe_stream" => {
                Some(Arc::new(TranscribeStreamAction { model: self.model.clone(), detok: self.detok.clone(), sessions: self.sessions.clone() }) as Arc<dyn Action>)
            }
            _ => None,
        }
    }
}

struct TranscribeStreamAction {
    model: Arc<NemotronAsr>,
    detok: Arc<Detokenizer>,
    sessions: Arc<Mutex<StreamSessions>>,
}

impl Action for TranscribeStreamAction {
    fn spec(&self) -> ActionSpec {
        transcribe_stream_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let (id, wav, prompt_id, eos) = stream_job_from_inv(inv)?;
        progress(Progress::step(0, 1, format!("stream {id}: {} samples", wav.len())));
        let mut sessions = self.sessions.lock().map_err(|_| "nemotron stream sessions poisoned".to_string())?;
        Ok(sessions.step(self.model.encoder(), &self.detok, &id, &wav, prompt_id, eos))
    }
}

struct TranscribeAction {
    model: Arc<NemotronAsr>,
    detok: Arc<Detokenizer>,
}

impl Action for TranscribeAction {
    fn spec(&self) -> ActionSpec {
        transcribe_spec()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let blob = inv.get_blob("audio").ok_or("nemotron transcribe: missing 'audio' input")?;
        let wav = wav_from_blob(blob)?;
        let prompt_id = inv.get_i64("prompt_id").unwrap_or(0).max(0) as usize;
        progress(Progress::step(0, 1, "transcribing"));
        let tokens = self.model.transcribe(&wav, prompt_id);
        let text = self.detok.decode(&tokens);
        progress(Progress::step(1, 1, text.clone()));
        Ok(transcription_outcome(text, &tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Session semantics against the real checkpoint: windowed `step` calls emit
    /// text deltas whose concatenation equals the offline transcription, `eos`
    /// flushes the tail and closes the session. Heavy; run explicitly:
    /// `cargo test -p brain-nemotron --release stream_sessions -- --ignored`.
    #[test]
    #[ignore = "loads the 0.6B checkpoint (run explicitly)"]
    fn stream_sessions_deltas_match_offline() {
        use crate::encoder::encoder_pipelines;
        use std::path::Path;
        let ckpt = crate::model_dir("nvidia/nemotron-3.5-asr-streaming-0.6b").unwrap_or_default();
        let wav_path = crate::testdata("asr/audio/librispeech_mr_quilter.wav");
        if !Path::new(&wav_path).exists() || !Path::new(&format!("{ckpt}/model.safetensors")).exists() {
            eprintln!("skipping: assets absent (run `make fetch/testdata`)");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let wav = audio::wav::read(&wav_path).expect("wav");
        let w = crate::import::load_tensors(Path::new(&ckpt)).expect("load");
        let enc = Encoder::new(gpu_core::testgpu::dev(encoder_pipelines()), cfg, &w);
        let detok = Detokenizer::from_hf(&ckpt).expect("detok");

        let offline_text = detok.decode(&enc.transcribe(&wav.samples, 0));

        let mut sessions = StreamSessions::new();
        let mut text = String::new();
        let win = 16000usize; // 1 s windows, like the D-Bus reader default
        let mut i = 0;
        while i < wav.samples.len() {
            let end = (i + win).min(wav.samples.len());
            let eos = end == wav.samples.len();
            let o = sessions.step(&enc, &detok, "s1", &wav.samples[i..end], 0, eos);
            let seg = o.outputs.get("text").and_then(|v| v.as_str()).unwrap_or("");
            text.push_str(seg);
            if eos {
                assert_eq!(o.outputs.get("final").and_then(|v| v.as_bool()), Some(true));
            }
            i = end;
        }
        assert!(sessions.sessions.is_empty(), "eos closes the session");
        eprintln!("streamed: {text:?}");
        assert_eq!(text, offline_text, "concatenated deltas == offline transcription");
    }

    #[test]
    fn manifest_advertises_transcribe_and_stream() {
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 2);
        assert_eq!(m.actions[0].name, "transcribe");
        assert_eq!(m.actions[1].name, "transcribe_stream");
        assert!(m.actions[1].params.iter().any(|p| p.name == "stream"));
    }
}
