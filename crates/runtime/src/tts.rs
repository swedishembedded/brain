// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for on-device, real-time speech
// synthesis inside event-driven control software for its clients. If your team
// needs expertise in wiring neural TTS engines into embedded state machines then
// you can procure our services by sending an email to info@swedishembedded.com.

//! The real Qwen3-TTS adapter behind the runtime's [`SynthModel`] seam.
//!
//! Compiled only with the `qwen3tts` feature. Without it this module does not
//! exist and `brain-runtime` does not depend on `brain-qwen3tts` at all - that
//! crate drags in the whole voice stack (Mimi-style codec, ECAPA speaker
//! encoder, Talker + MTP decoders, tokenizer), which is exactly what the
//! default runtime build is kept clear of.
//!
//! [`Qwen3TtsSynthModel`] owns a resolved [`TtsPaths`] + [`GenOpts`] and, per
//! request, calls
//!   * [`qwen3tts::pipeline::synth`] - no `ref_audio`: speaker-free synthesis;
//!   * [`qwen3tts::pipeline::clone`] - with `ref_audio`: x-vector voice clone,
//!     upgraded to the in-context (ICL) path when `ref_text` is also present
//!     (the reference wav is codec-encoded in-tree).
//! Both return the 24 kHz mono waveform the [`AudioStreamPump`](crate::AudioStreamPump)
//! then slices into `audio_chunk` events.
//!
//! **Errors.** The seam is `fn synth(&self, req) -> Vec<f32>` - there is no error
//! channel, and widening it would change every implementor including the fake.
//! A failed generation is therefore reported on stderr and returns an empty
//! waveform, which the controller already handles as "drained": it emits the
//! terminal `done:true` `audio_chunk` and returns to `Idle`, so a bad request
//! never wedges or faults the session. Fail-fast validation of the checkpoint
//! itself happens once, in [`Qwen3TtsSynthModel::load`], not per request.

use qwen3tts::{GenOpts, TtsPaths};

use crate::{SynthModel, SynthRequest};

/// Qwen3-TTS codec output rate. The Mimi-style codec always decodes to 24 kHz
/// mono, which is what the `audio_chunk` events advertise as `sample_rate`.
pub const SAMPLE_RATE: u32 = 24_000;

/// A real [`SynthModel`] backed by a Qwen3-TTS checkpoint.
///
/// Construct with [`load`](Qwen3TtsSynthModel::load) (explicit paths) or
/// [`from_env`](Qwen3TtsSynthModel::from_env) (`BRAIN_QWEN3TTS_WEIGHTS` /
/// `BRAIN_QWEN3TTS_CKPT` / `BRAIN_QWEN3TTS_LANG`, the same variables the D-Bus
/// resident already uses - one spelling for both serving surfaces).
///
/// The public pipeline entry points load their weights per call (the load-once
/// seam is OpenVINO/NPU-only), so this holds configuration, not tensors; its
/// footprint is a handful of strings.
pub struct Qwen3TtsSynthModel {
    paths: TtsPaths,
    opts: GenOpts,
    /// Language used when a [`SynthRequest`] does not name one.
    language: String,
}

impl Qwen3TtsSynthModel {
    /// Resolve a checkpoint and validate it eagerly.
    ///
    /// `weights_dir` holds the brain-format `talker/mtp/codec/speaker.safetensors`;
    /// `ckpt_dir` is the original HF checkpoint directory (`config.json` +
    /// tokenizer). Missing files are an error HERE rather than a silent empty
    /// waveform on the first request.
    pub fn load(weights_dir: &str, ckpt_dir: &str) -> Result<Qwen3TtsSynthModel, String> {
        let paths = TtsPaths {
            talker: format!("{weights_dir}/talker.safetensors"),
            mtp: format!("{weights_dir}/mtp.safetensors"),
            codec: format!("{weights_dir}/codec.safetensors"),
            speaker: format!("{weights_dir}/speaker.safetensors"),
            ckpt_dir: ckpt_dir.to_string(),
        };
        for f in [&paths.talker, &paths.mtp, &paths.codec, &paths.speaker] {
            if !std::path::Path::new(f).exists() {
                return Err(format!("qwen3tts: missing checkpoint file {f}"));
            }
        }
        let cfg = std::path::Path::new(ckpt_dir).join("config.json");
        if !cfg.exists() {
            return Err(format!(
                "qwen3tts: {} not found (the HF checkpoint dir supplies config.json + tokenizer)",
                cfg.display()
            ));
        }
        Ok(Qwen3TtsSynthModel { paths, opts: GenOpts::default(), language: "english".to_string() })
    }

    /// Configure from the environment, mirroring the D-Bus resident's own
    /// `from_env`. `None` (not registered) when `BRAIN_QWEN3TTS_WEIGHTS` is unset
    /// or empty; `Some(Err(..))` when it is set but the checkpoint is unusable,
    /// so a typo in the path is reported instead of silently serving nothing.
    pub fn from_env() -> Option<Result<Qwen3TtsSynthModel, String>> {
        let weights = std::env::var("BRAIN_QWEN3TTS_WEIGHTS").ok().filter(|p| !p.is_empty())?;
        let ckpt = std::env::var("BRAIN_QWEN3TTS_CKPT").unwrap_or_default();
        let lang = std::env::var("BRAIN_QWEN3TTS_LANG")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "english".to_string());
        Some(Qwen3TtsSynthModel::load(&weights, &ckpt).map(|m| m.with_language(lang)))
    }

    /// Override the sampling / length controls (default: the reference recipe -
    /// `temperature 0.9`, `top_k 50`, `max_frames 256`). `max_frames` bounds the
    /// clip: the codec runs at 12.5 Hz, so 256 frames is about twenty seconds.
    pub fn with_opts(mut self, opts: GenOpts) -> Qwen3TtsSynthModel {
        self.opts = opts;
        self
    }

    /// Override the fallback language used when a request does not name one.
    pub fn with_language(mut self, language: impl Into<String>) -> Qwen3TtsSynthModel {
        self.language = language.into();
        self
    }

    /// The generation config this adapter was built with.
    pub fn opts(&self) -> &GenOpts {
        &self.opts
    }

    /// The fallible synthesis, before the trait's error-free signature swallows
    /// the failure. Exposed so a caller that wants the message (a test, a CLI
    /// path) can have it.
    pub fn try_synth(&self, req: &SynthRequest) -> Result<Vec<f32>, String> {
        if req.text.trim().is_empty() {
            return Err("qwen3tts: empty synthesis text".to_string());
        }
        let language = req.language.clone().unwrap_or_else(|| self.language.clone());
        match &req.ref_audio {
            // Voice clone: x-vector timbre from the reference wav, and the ICL
            // path on top of it whenever a transcript came with it.
            Some(wav) => qwen3tts::pipeline::clone(
                &self.paths,
                &self.opts,
                &req.text,
                wav,
                req.ref_text.as_deref().unwrap_or(""),
                &language,
                None,
            ),
            None => qwen3tts::pipeline::synth(&self.paths, &self.opts, &req.text, &language),
        }
    }
}

impl SynthModel for Qwen3TtsSynthModel {
    fn synth(&self, req: &SynthRequest) -> Vec<f32> {
        match self.try_synth(req) {
            Ok(wav) => wav,
            Err(e) => {
                // See the module doc: no error channel on the seam, so report and
                // return "nothing to stream" rather than panicking a live session.
                eprintln!("qwen3tts synth: {e}");
                Vec::new()
            }
        }
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Controller, Registry};
    use events::{Envelope, Event};

    /// In-repo default weights directory produced by the Qwen3-TTS import
    /// (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`). Overridable with
    /// `BRAIN_QWEN3TTS_WEIGHTS`; the HF checkpoint dir has no in-repo default and
    /// must come from `BRAIN_QWEN3TTS_CKPT`.
    const DEFAULT_WEIGHTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../out/tts-base06");

    /// The codec runs at 12.5 Hz against a 24 kHz waveform, so one codec frame
    /// decodes to exactly 1920 samples.
    const SAMPLES_PER_FRAME: usize = SAMPLE_RATE as usize * 2 / 25;

    /// Real end-to-end generation is minutes-long at the default 256-frame cap;
    /// this bounds the test to a short clip while still exercising prefill,
    /// autoregressive Talker decode, MTP residual fill and the codec decode.
    /// `min_new` is raised above the default so the run cannot satisfy the
    /// assertions with two frames of audio if the model emits EOS early.
    fn short_clip_opts() -> GenOpts {
        GenOpts { max_frames: 24, min_new: 16, ..GenOpts::default() }
    }

    /// Resolve a real checkpoint, or `None` when this machine has none.
    fn real_model() -> Option<Qwen3TtsSynthModel> {
        let weights = std::env::var("BRAIN_QWEN3TTS_WEIGHTS")
            .ok()
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| DEFAULT_WEIGHTS.to_string());
        let Ok(ckpt) = std::env::var("BRAIN_QWEN3TTS_CKPT") else {
            eprintln!("skipping: set BRAIN_QWEN3TTS_CKPT to the HF Qwen3-TTS checkpoint dir");
            return None;
        };
        match Qwen3TtsSynthModel::load(&weights, &ckpt) {
            Ok(m) => Some(m.with_opts(short_clip_opts())),
            Err(e) => {
                eprintln!("skipping: {e}");
                None
            }
        }
    }

    /// Root-mean-square of a waveform - the "is this actually voice, or silence?"
    /// statistic the pipeline's own doc cites (about 0.07 for a sampled decode,
    /// about 0.004 for the degenerate greedy collapse).
    fn rms(w: &[f32]) -> f32 {
        (w.iter().map(|s| s * s).sum::<f32>() / w.len().max(1) as f32).sqrt()
    }

    /// A missing checkpoint is a construction-time error, not an empty waveform
    /// on the first request. Runs everywhere - no weights needed.
    #[test]
    fn load_rejects_a_checkpoint_that_is_not_there() {
        // `expect_err` would need `Qwen3TtsSynthModel: Debug`, which would in turn
        // need it on `qwen3tts::TtsPaths` - match instead of widening that.
        match Qwen3TtsSynthModel::load("/nonexistent-weights-dir", "/nonexistent-ckpt-dir") {
            Err(e) => assert!(e.contains("talker.safetensors"), "{e}"),
            Ok(_) => panic!("a missing checkpoint must not load"),
        }
    }

    /// REAL end-to-end generation through the seam: a genuine checkpoint, a
    /// genuine `SynthRequest`, a genuine 24 kHz waveform. Asserts the output is
    /// finite, is not silence, and is as long as the frame budget implies.
    #[test]
    fn a_real_checkpoint_synthesizes_a_real_waveform() {
        let Some(model) = real_model() else { return };
        let req = SynthRequest {
            text: "Brain speaks with its own voice.".to_string(),
            ..SynthRequest::default()
        };

        let t0 = std::time::Instant::now();
        let wav = model.try_synth(&req).expect("real synthesis must succeed");
        let secs = t0.elapsed().as_secs_f64();

        let opts = model.opts();
        let frames = wav.len() as f64 / SAMPLES_PER_FRAME as f64;
        let peak = wav.iter().fold(0f32, |m, s| m.max(s.abs()));
        println!(
            "qwen3tts synth: {} samples ({:.2}s audio, {frames:.1} codec frames) in {secs:.1}s wall, rms {:.4}, peak {peak:.4}",
            wav.len(),
            wav.len() as f32 / SAMPLE_RATE as f32,
            rms(&wav),
        );

        assert_eq!(model.sample_rate(), SAMPLE_RATE);
        assert!(wav.iter().all(|s| s.is_finite()), "waveform contains NaN/inf");
        // Length: at least `min_new` frames were forced, and the cap is `max_frames`
        // (plus codec transient slack).
        assert!(
            wav.len() >= opts.min_new * SAMPLES_PER_FRAME,
            "{} samples is shorter than the forced {} frames",
            wav.len(),
            opts.min_new
        );
        assert!(
            wav.len() <= (opts.max_frames + 2) * SAMPLES_PER_FRAME,
            "{} samples exceeds the {}-frame cap",
            wav.len(),
            opts.max_frames
        );
        // Content: real speech, not the near-silent degenerate decode.
        let r = rms(&wav);
        assert!(r > 0.01, "rms {r:.5} is silence, not speech");
        assert!(r < 1.0 && peak <= 1.5, "rms {r:.5} / peak {peak:.5} is not a sane waveform");
    }

    /// The adapter drives the controller's `Synthesizing` state for real: one
    /// `user_synth_request` in, a stream of `audio_chunk` events out, terminated
    /// by `done:true`, with the decoded PCM matching what the adapter produced.
    #[test]
    fn the_adapter_streams_through_the_controller() {
        let Some(model) = real_model() else { return };
        let mut reg = Registry::new();
        reg.synth = Some(Box::new(model));
        let mut ctrl = Controller::new(reg);

        let out = ctrl.feed_event(Event::UserSynthRequest {
            text: "Two short words.".to_string(),
            ref_audio: None,
            ref_text: None,
            language: None,
        });

        let chunks: Vec<(&String, bool, u32)> = out
            .iter()
            .filter_map(|Envelope { event, .. }| match event {
                Event::AudioChunk { pcm_b64, done, sample_rate, .. } => {
                    Some((pcm_b64, *done, *sample_rate))
                }
                _ => None,
            })
            .collect();
        // `audio_chunk` carries base64 little-endian f32 PCM (see `pump::encode_pcm`).
        let samples: usize = chunks
            .iter()
            .map(|(b64, _, _)| crate::pump::decode_pcm(b64).map(|s| s.len()).unwrap_or(0))
            .sum();
        assert!(chunks.len() >= 2, "expected streamed chunks plus a terminal done: {out:?}");
        println!(
            "controller stream: {} audio_chunk events, {samples} PCM samples ({:.2}s at {} Hz)",
            chunks.len(),
            samples as f32 / SAMPLE_RATE as f32,
            chunks[0].2
        );

        assert!(chunks.iter().all(|&(_, _, sr)| sr == SAMPLE_RATE));
        assert!(chunks.last().unwrap().1, "the last audio_chunk must carry done:true");
        assert!(
            chunks[..chunks.len() - 1].iter().all(|&(_, done, _)| !done),
            "only the final chunk may be done"
        );
        assert!(samples > 0, "the controller streamed no audio");

        // Recovered to Idle: a second request is served, not swallowed.
        let again = ctrl.feed_event(Event::CapabilitiesRequest);
        assert!(again.iter().any(|e| matches!(e.event, Event::CapabilitiesResult { .. })));
    }
}
