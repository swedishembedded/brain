// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A **resident** Qwen3-TTS engine for the ordinary (non-NPU) host path: load
//! the checkpoints ONCE, then serve many requests, streaming each request's
//! waveform out chunk-by-chunk as the codec decodes it.
//!
//! Swedish Embedded AB implements low-latency, load-once speech-synthesis
//! services for its clients. If your team needs expertise in resident
//! neural-audio serving - keeping the weights hot and getting the first audio
//! samples onto the wire before the clip has finished generating - you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # Why this exists
//!
//! [`crate::pipeline`]'s `synth`/`clone`/`design` are one-shot: each call
//! re-reads the Talker (3 GB fp32 for the 0.6B), the MTP, the codec and the
//! speaker encoder from disk, generates, and returns the finished waveform in
//! one piece. That is right for a CLI invocation and wrong for a server: a
//! served model pays the load on every request and cannot emit a single
//! sample until the last frame is generated.
//!
//! The only load-once seam this crate had was [`crate::serve::TtsEngine`],
//! and it is OpenVINO/NPU-only - which also forced a private Unix-socket
//! protocol in the CLI to reach it. This engine is the same two capabilities
//! (resident weights + progressive audio) on hardware that any box has:
//!
//!   * **Resident weights** - [`CpuTalker`], [`CpuMtp`], the streaming codec
//!     decoder and the tokenizer are loaded in [`ResidentEngine::load`] and
//!     reused by every subsequent call. A configured reference voice is
//!     encoded once too ([`ResidentEngine::clone_voice`]'s x-vector + ICL
//!     codes cache), so repeat clones of the same timbre skip the speaker
//!     encoder AND the codec encode entirely.
//!   * **Progressive audio** - the codec decode runs through
//!     [`mimi::decode_stream::StreamingCodecDecoder`], which carries real
//!     per-conv state across chunks (no warmup re-decode), so each
//!     `chunk_frames` block of codec frames yields its own audio segment via
//!     the `on_audio` callback while the rest is still decoding. The full
//!     waveform is still returned, so a caller that only wants the finished
//!     clip passes a no-op callback and notices nothing.
//!
//! Everything here is CPU-side by construction (host `CpuTalker`/`CpuMtp`
//! decode, pure-CPU streaming codec back), which is what makes the resident
//! adapter's declared cost - RAM, zero VRAM - honest. A GPU-resident variant
//! would be a different `MemCost` and a different engine.

use crate::gen_kv::CpuTalker;
use crate::gen_kv_mtp::CpuMtp;
use crate::pipeline::{self, GenOpts, TtsPaths};
use crate::prompt::{self, Prompt, TtsSpecials};
use capability::CancelToken;
use data::tokenizer::Tokenizer;
use mimi::decode_stream::StreamingCodecDecoder;

/// Qwen3-TTS output sample rate (see [`crate::pipeline`]).
pub const SAMPLE_RATE: u32 = 24_000;

/// Codec frames decoded per streamed audio chunk. 16 frames at 12.5 Hz is
/// ~1.28 s of audio - the same default `crate::serve`'s NPU server uses, and
/// overridable with the same `BRAIN_QWEN3TTS_STREAM_CHUNK` variable so the two
/// serving paths cannot drift apart on the one knob a listener actually hears.
pub const DEFAULT_CHUNK_FRAMES: usize = 16;

/// Read the shared streaming-chunk knob.
pub fn chunk_frames_from_env() -> usize {
    std::env::var("BRAIN_QWEN3TTS_STREAM_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(DEFAULT_CHUNK_FRAMES).max(1)
}

/// Everything a resident TTS instance holds hot between requests.
pub struct ResidentEngine {
    talker: CpuTalker,
    mtp: CpuMtp,
    codec: StreamingCodecDecoder,
    tok: data::qwen_tokenizer::QwenBpe,
    sp: TtsSpecials,
    codec_path: String,
    speaker_path: String,
    /// Lazily loaded ECAPA x-vector encoder (only the clone path needs it).
    speaker: Option<ecapatdnn::SpeakerEncoder>,
    /// The reference voice conditioning computed for one wav, kept across
    /// calls: `(wav path, ref_text, x-vector, ICL codes)`. Re-encoding a
    /// reference clip is the single most expensive thing a clone does that
    /// does not depend on the request, so a server must do it once.
    ref_cache: Option<(String, String, Vec<f32>, Option<Vec<u32>>)>,
    chunk_frames: usize,
}

impl ResidentEngine {
    /// Load every checkpoint the host path needs. Fails cleanly (a `Result`,
    /// not a panic inside a loader) when a file is missing - a served model
    /// must report a bad configuration, not abort the worker.
    pub fn load(paths: &TtsPaths) -> Result<ResidentEngine, String> {
        for p in [&paths.talker, &paths.mtp, &paths.codec] {
            if !std::path::Path::new(p).exists() {
                return Err(format!("tts: weights not found at '{p}' (run `brain tts import`)"));
            }
        }
        let sp = TtsSpecials::from_config_dir(&paths.ckpt_dir)?;
        let tok = prompt::load_tokenizer(&paths.ckpt_dir)?;
        Ok(ResidentEngine {
            talker: CpuTalker::load(&paths.talker),
            mtp: CpuMtp::load(&paths.mtp),
            codec: StreamingCodecDecoder::load(&paths.codec),
            tok,
            sp,
            codec_path: paths.codec.clone(),
            speaker_path: paths.speaker.clone(),
            speaker: None,
            ref_cache: None,
            chunk_frames: chunk_frames_from_env(),
        })
    }

    /// Codec frames per streamed chunk (see [`DEFAULT_CHUNK_FRAMES`]).
    pub fn chunk_frames(&self) -> usize {
        self.chunk_frames
    }

    /// Override the streamed chunk size (frames). `0` is clamped to 1.
    pub fn set_chunk_frames(&mut self, n: usize) {
        self.chunk_frames = n.max(1);
    }

    /// The checkpoint's special-token table (language ids, preset speakers) -
    /// a caller validating a request against the loaded model, not a copy.
    pub fn specials(&self) -> &TtsSpecials {
        &self.sp
    }

    /// Tokenize `text` through the assistant chat template and split it into
    /// the role header and the target-text content, exactly as every
    /// [`crate::pipeline`] entry point does.
    fn text_ids(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>), String> {
        pipeline::split_input_ids(&self.tok.encode(&pipeline::assistant_text(text)))
    }

    /// **Speaker-free synthesis** - the resident mirror of
    /// [`crate::pipeline::synth`]. `cancel` is polled once per generated
    /// frame, same convention as every other cancellable entry point in this
    /// crate; pass `&CancelToken::default()` to run uninterrupted.
    pub fn speak(
        &mut self,
        text: &str,
        lang: &str,
        opts: &GenOpts,
        cancel: &CancelToken,
        on_audio: &mut dyn FnMut(&[f32], u32),
    ) -> Result<Vec<f32>, String> {
        let language_id = self.sp.language_id(lang);
        let (role_ids, text_ids) = self.text_ids(text)?;
        let prompt = prompt::build_xvector_prompt(&self.talker, &self.sp, &role_ids, &text_ids, None, language_id);
        self.run_prompt(&prompt, opts, cancel, on_audio)
    }

    /// **VoiceDesign / CustomVoice** - the resident mirror of
    /// [`crate::pipeline::design`].
    #[allow(clippy::too_many_arguments)]
    pub fn design(
        &mut self,
        text: &str,
        lang: &str,
        instruct: &str,
        speaker: Option<&str>,
        opts: &GenOpts,
        cancel: &CancelToken,
        on_audio: &mut dyn FnMut(&[f32], u32),
    ) -> Result<Vec<f32>, String> {
        let language_id = self.sp.language_id(lang);
        let speaker_id = pipeline::resolve_speaker(&self.sp, speaker)?;
        let (role_ids, text_ids) = self.text_ids(text)?;
        let instruct_ids = if instruct.trim().is_empty() { Vec::new() } else { self.tok.encode(&pipeline::instruct_text(instruct)) };
        let prompt =
            prompt::build_instruct_prompt(&self.talker, &self.sp, &role_ids, &text_ids, &instruct_ids, speaker_id, language_id);
        self.run_prompt(&prompt, opts, cancel, on_audio)
    }

    /// **Voice cloning** - the resident mirror of [`crate::pipeline::clone`].
    /// The reference conditioning (x-vector, and the ICL codec codes when
    /// `ref_text` is non-empty) is computed once per `(ref_wav, ref_text)`
    /// pair and reused, which is the whole point of a resident clone server:
    /// only the target text changes between requests.
    #[allow(clippy::too_many_arguments)]
    pub fn clone_voice(
        &mut self,
        text: &str,
        ref_wav: &str,
        ref_text: &str,
        lang: &str,
        opts: &GenOpts,
        cancel: &CancelToken,
        on_audio: &mut dyn FnMut(&[f32], u32),
    ) -> Result<Vec<f32>, String> {
        let language_id = self.sp.language_id(lang);
        self.ensure_ref(ref_wav, ref_text)?;
        let (_, _, xvec, ref_code) = self.ref_cache.as_ref().expect("ensure_ref populated the cache");
        let xvec = xvec.clone();
        let ref_code = ref_code.clone();
        let (role_ids, text_ids) = self.text_ids(text)?;

        let prompt = match &ref_code {
            Some(codes) => {
                let full = self.tok.encode(&format!("<|im_start|>assistant\n{ref_text}<|im_end|>\n"));
                if full.len() < 3 + 2 + 1 {
                    return Err("ref_text tokenized too short".to_string());
                }
                let ref_ids = &full[3..full.len() - 2];
                prompt::build_icl_prompt(&self.talker, &self.mtp, &self.sp, &role_ids, &text_ids, ref_ids, codes, &xvec, language_id)
            }
            None => prompt::build_xvector_prompt(&self.talker, &self.sp, &role_ids, &text_ids, Some(&xvec), language_id),
        };
        self.run_prompt(&prompt, opts, cancel, on_audio)
    }

    /// Populate [`Self::ref_cache`] for `(ref_wav, ref_text)` if it does not
    /// already hold exactly that pair.
    fn ensure_ref(&mut self, ref_wav: &str, ref_text: &str) -> Result<(), String> {
        if matches!(&self.ref_cache, Some((w, t, _, _)) if w == ref_wav && t == ref_text) {
            return Ok(());
        }
        if !std::path::Path::new(&self.speaker_path).exists() {
            return Err(format!("tts clone: speaker encoder not found at '{}'", self.speaker_path));
        }
        let wav = audio::wav::read(ref_wav).map_err(|e| format!("read {ref_wav}: {e}"))?;
        if self.speaker.is_none() {
            self.speaker = Some(ecapatdnn::SpeakerEncoder::load_inference(&self.speaker_path));
        }
        let xvec = self.speaker.as_ref().expect("just loaded").embed_wav(&wav.samples, wav.sample_rate);
        let ref_code = if ref_text.trim().is_empty() {
            None
        } else {
            let codec = mimi::Codec::load_inference(&self.codec_path);
            let sr = codec.cfg.input_sample_rate;
            Some(codec.encode(&audio::resample_linear(&wav.samples, wav.sample_rate, sr)))
        };
        self.ref_cache = Some((ref_wav.to_string(), ref_text.to_string(), xvec, ref_code));
        Ok(())
    }

    /// Generate + progressively decode one assembled prompt: the shared tail
    /// of every action above. Returns the complete waveform; `on_audio` sees
    /// the same samples first, `chunk_frames` codec frames at a time. A
    /// cancel observed mid-generation reports `Err("cancelled")` rather than
    /// decoding the partial codes - a server-side hangup drops the request,
    /// it does not return a truncated clip as if it were complete.
    fn run_prompt(
        &mut self,
        prompt: &Prompt,
        opts: &GenOpts,
        cancel: &CancelToken,
        on_audio: &mut dyn FnMut(&[f32], u32),
    ) -> Result<Vec<f32>, String> {
        let codes = pipeline::generate_codes_cached(&mut self.talker, &mut self.mtp, &self.sp, prompt, opts, cancel)
            .map_err(|_| "cancelled".to_string())?;
        if codes.is_empty() {
            return Err("no codec frames were generated".to_string());
        }
        let mut full: Vec<f32> = Vec::new();
        self.codec.decode_streaming_cb(&codes, self.chunk_frames, &mut |pcm, seq| {
            full.extend_from_slice(pcm);
            on_audio(pcm, seq);
        });
        if full.is_empty() {
            return Err("codec produced no audio".to_string());
        }
        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_paths() -> Option<TtsPaths> {
        let (Ok(w), Ok(c)) = (std::env::var("BRAIN_QWEN3TTS_WEIGHTS"), std::env::var("BRAIN_QWEN3TTS_CKPT")) else {
            return None;
        };
        let paths = TtsPaths {
            talker: format!("{w}/talker.safetensors"),
            mtp: format!("{w}/mtp.safetensors"),
            codec: format!("{w}/codec.safetensors"),
            speaker: format!("{w}/speaker.safetensors"),
            ckpt_dir: c,
        };
        std::path::Path::new(&paths.talker).exists().then_some(paths)
    }

    /// A missing checkpoint must come back as a clean error, not a panic
    /// inside `CpuTalker::load` - a served model reports a bad config, it
    /// does not abort the worker thread that was asked to activate it.
    #[test]
    fn loading_absent_weights_is_a_clean_error() {
        let paths = TtsPaths {
            talker: "/nonexistent/talker.safetensors".into(),
            mtp: "/nonexistent/mtp.safetensors".into(),
            codec: "/nonexistent/codec.safetensors".into(),
            speaker: "/nonexistent/speaker.safetensors".into(),
            ckpt_dir: "/nonexistent".into(),
        };
        let Err(err) = ResidentEngine::load(&paths) else {
            panic!("loading absent weights must fail");
        };
        assert!(err.contains("not found"), "got: {err}");
    }

    /// The two properties that make this an engine rather than another
    /// one-shot wrapper, against REAL weights: (1) the audio arrives in
    /// several progressive chunks whose concatenation IS the returned
    /// waveform, and (2) a SECOND request on the same engine reuses the
    /// loaded weights - so it must be markedly faster than the first,
    /// load-inclusive call, and still produce audio.
    /// Gated on `BRAIN_QWEN3TTS_WEIGHTS`/`_CKPT` like every other
    /// real-checkpoint test in this crate.
    #[test]
    fn resident_engine_streams_chunks_and_stays_hot_between_requests() {
        let Some(paths) = real_paths() else {
            brain_testutil::skip("BRAIN_QWEN3TTS_WEIGHTS/BRAIN_QWEN3TTS_CKPT not set");
            return;
        };
        let t_load = std::time::Instant::now();
        let mut eng = ResidentEngine::load(&paths).expect("engine loads");
        let load_s = t_load.elapsed().as_secs_f64();
        eng.set_chunk_frames(8);
        // Long enough that the clip is past the model's leading silence, so
        // the audio assertion below is about real synthesized content rather
        // than a 0.6 s lead-in that is legitimately near-DC.
        let opts = GenOpts { max_frames: 40, seed: 7, ..GenOpts::default() };

        let mut chunks: Vec<usize> = Vec::new();
        let mut seqs: Vec<u32> = Vec::new();
        let t0 = std::time::Instant::now();
        let wav = eng
            .speak("Streaming the first request.", "english", &opts, &CancelToken::default(), &mut |pcm, seq| {
                chunks.push(pcm.len());
                seqs.push(seq);
            })
            .expect("first speak");
        let first_s = t0.elapsed().as_secs_f64();

        assert!(chunks.len() >= 2, "expected progressive chunks, got {}", chunks.len());
        assert_eq!(seqs, (0..seqs.len() as u32).collect::<Vec<_>>(), "chunk sequence numbers must be dense and ordered");
        assert_eq!(chunks.iter().sum::<usize>(), wav.len(), "streamed chunks must concatenate to the returned waveform");
        let rms = (wav.iter().map(|s| s * s).sum::<f32>() / wav.len() as f32).sqrt();
        assert!(rms > 1e-3, "resident engine decoded to near-silence (rms {rms:.3e}) - the collapse this crate has a named history of");

        // The property a RESIDENT engine has to get right that a load-per-call
        // one gets for free: no state leaks between requests. The same request
        // run again on the same hot engine must produce bit-identical audio -
        // a Talker KV cache (or an RNG) carried over from the previous call
        // would silently corrupt every request after the first.
        let t1 = std::time::Instant::now();
        let wav2 =
            eng.speak("Streaming the first request.", "english", &opts, &CancelToken::default(), &mut |_, _| {}).expect("second speak");
        let second_s = t1.elapsed().as_secs_f64();
        assert_eq!(wav, wav2, "a repeated request on a hot engine must be bit-identical - resident state leaked between calls");
        eprintln!("engine: load={load_s:.1}s first={first_s:.1}s repeat={second_s:.1}s ({} chunks, {} samples, rms {rms:.3e})", chunks.len(), wav.len());
    }
}
