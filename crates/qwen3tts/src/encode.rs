// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning recorded speech corpora
// into training data for speech synthesizers for its clients. If your team
// needs expertise in speech data preparation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Encode a recording the way a training shard records it: the codec's
//! `[frames, 16]` codes, the speaker encoder's x-vector and, given its
//! transcript, the token ids generation tokenises that text into. This is the
//! `encode` action's work ([`crate::caps`]); an [`Encoder`] holds the codec,
//! the speaker encoder and the tokenizer loaded once, so a resident instance
//! serves a corpus without reloading them per clip.
//!
//! A recording arrives as WAV bytes from a caller and is parsed as hostile
//! input: its size is capped before it is decoded ([`MAX_WAV_BYTES`]) and its
//! duration after ([`MAX_SECONDS`]); a sample rate outside the shared bounds
//! is refused by the WAV reader. The codec reads 24 kHz, so the clip is
//! resampled with the shared resampler, exactly as a clone's reference is.

use capability::failure::{tag, FailureClass};
use data::tokenizer::Tokenizer;

use crate::TtsPaths;

/// Largest WAV a call may hand over: ten minutes of 48 kHz 16-bit stereo fits.
pub const MAX_WAV_BYTES: usize = 128 << 20;
/// Longest recording, in seconds (the shared served ceiling on audio length).
pub const MAX_SECONDS: f64 = capability::bounds::SECONDS;
/// Longest transcript, in bytes.
pub const MAX_TEXT_BYTES: usize = 16 << 10;

/// Encoders a resident instance runs side by side, each on its own share of
/// a batch. Measured on the CPU backend (a 10 s clip, a shared 288-core host
/// under load): one stream 3.2 audio seconds per wall second, two 5.2, four
/// 6.7. Each holds its own codec and speaker encoder.
pub const STREAMS: usize = 4;

/// A validated encode request.
pub struct Request {
    pub wav: audio::wav::Wav,
    /// The transcript, when its token ids are asked for.
    pub text: Option<String>,
}

impl Request {
    /// Parse a caller's WAV bytes and optional transcript, refusing what is
    /// oversized, malformed or too long before any model work. Errors are
    /// tagged `invalid-request` and name what is wrong.
    pub fn parse(wav: &[u8], text: Option<String>) -> Result<Request, String> {
        let invalid = |m: String| tag(FailureClass::InvalidRequest, m);
        if wav.len() > MAX_WAV_BYTES {
            return Err(invalid(format!("audio: {} bytes is more than the {MAX_WAV_BYTES} a WAV may hold", wav.len())));
        }
        let wav = audio::wav::parse(wav).map_err(|e| invalid(format!("audio: not a readable WAV: {e}")))?;
        let seconds = wav.samples.len() as f64 / f64::from(wav.sample_rate);
        if seconds > MAX_SECONDS {
            return Err(invalid(format!("audio: {seconds:.1} s is longer than the {MAX_SECONDS} s a recording may run")));
        }
        if wav.samples.is_empty() {
            return Err(invalid("audio: the WAV holds no samples".to_string()));
        }
        if let Some(t) = &text {
            if t.trim().is_empty() {
                return Err(invalid("text: empty; omit it to encode the audio alone".to_string()));
            }
            if t.len() > MAX_TEXT_BYTES {
                return Err(invalid(format!("text: {} bytes is more than {MAX_TEXT_BYTES}", t.len())));
            }
        }
        Ok(Request { wav, text })
    }

    /// The recording's duration in seconds, at its own rate.
    pub fn seconds(&self) -> f64 {
        self.wav.samples.len() as f64 / f64::from(self.wav.sample_rate)
    }
}

/// What one recording encodes to.
pub struct Encoded {
    /// `[frames, codebooks]` row-major.
    pub codes: Vec<u16>,
    pub frames: usize,
    pub xvector: Vec<f32>,
    /// The tokenised assistant turn of the transcript, when one was given.
    pub text_ids: Option<Vec<u32>>,
}

/// The codec, the speaker encoder and the tokenizer of one checkpoint, loaded
/// once.
pub struct Encoder {
    codec: mimi::Codec,
    speaker: ecapatdnn::SpeakerEncoder,
    tokenizer: data::qwen_tokenizer::QwenBpe,
}

impl Encoder {
    /// Load the components of `paths`. The checkpoint must ship a speaker
    /// encoder (a Base checkpoint does).
    pub fn load(paths: &TtsPaths) -> Result<Encoder, String> {
        paths.require(true)?;
        let tokenizer = crate::prompt::load_tokenizer(&paths.ckpt_dir)?;
        // The codec encodes on the CPU path, as a clone's reference is
        // encoded (`mimi::Codec::load_inference`), so the codes a corpus is
        // encoded to are the codes inference conditions on.
        let codec = mimi::Codec::load_inference(&paths.codec);
        if codec.cfg.semantic_codebook_size.max(codec.cfg.codebook_size) > u32::from(u16::MAX) + 1 {
            return Err("the codec's codebooks are wider than a u16 code".to_string());
        }
        let speaker = ecapatdnn::SpeakerEncoder::load_inference(&paths.speaker);
        Ok(Encoder { codec, speaker, tokenizer })
    }

    /// Codebooks per frame.
    pub fn codebooks(&self) -> u32 {
        self.codec.cfg.num_quantizers
    }

    /// Codec frames per second.
    pub fn frame_rate(&self) -> f64 {
        f64::from(self.codec.cfg.input_sample_rate) / f64::from(self.codec.cfg.decode_upsample_rate)
    }

    /// The token ids generation tokenises `text` into
    /// ([`crate::pipeline::assistant_text`]).
    pub fn text_ids(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(&crate::pipeline::assistant_text(text))
    }

    /// Encode one validated recording.
    pub fn encode(&self, req: &Request) -> Encoded {
        let wav = &req.wav;
        let at_codec_rate = audio::resample::rational(&wav.samples, self.codec.cfg.input_sample_rate, wav.sample_rate);
        let codes: Vec<u16> = self.codec.encode(&at_codec_rate).into_iter().map(|c| c as u16).collect();
        let frames = codes.len() / self.codebooks() as usize;
        let xvector = self.speaker.embed_wav(&wav.samples, wav.sample_rate);
        Encoded { codes, frames, xvector, text_ids: req.text.as_deref().map(|t| self.text_ids(t)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(seconds: f64, rate: u32) -> Vec<u8> {
        audio::wav::encode(&vec![0.1; (seconds * f64::from(rate)) as usize], rate)
    }

    #[test]
    fn a_hostile_recording_is_refused_naming_what_is_wrong_before_any_model_work() {
        let cases: Vec<(Vec<u8>, Option<String>, &str)> = vec![
            (b"RIFF\0\0\0\0WAVEjunk".to_vec(), None, "not a readable WAV"),
            (b"hello".to_vec(), None, "not a readable WAV"),
            (vec![0u8; MAX_WAV_BYTES + 1], None, "more than"),
            (wav(MAX_SECONDS + 1.0, 1_000), None, "longer than"),
            (wav(1.0, 16_000), Some("  ".to_string()), "empty"),
            (wav(1.0, 16_000), Some("a".repeat(MAX_TEXT_BYTES + 1)), "text"),
            (audio::wav::encode(&[], 16_000), None, "no samples"),
        ];
        for (bytes, text, why) in cases {
            let e = Request::parse(&bytes, text).err().unwrap_or_else(|| panic!("{why}: accepted"));
            assert!(e.contains(why), "{why}: {e}");
            assert!(e.contains("invalid-request"), "{e}");
        }
    }

    #[test]
    fn a_recording_at_any_rate_is_accepted_with_its_own_rate_and_duration() {
        for rate in [8_000, 16_000, 24_000, 44_100, 48_000] {
            let r = Request::parse(&wav(1.5, rate), Some("hej".to_string())).unwrap();
            assert_eq!(r.wav.sample_rate, rate);
            assert!((r.seconds() - 1.5).abs() < 1e-3);
        }
    }
}
