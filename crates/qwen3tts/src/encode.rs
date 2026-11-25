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
//! serves a corpus without reloading them per clip, and encodes a batch of
//! recordings as one device pass of each encoder ([`Encoder::encode_batch`]).
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

/// Recordings one many-clip request may carry.
pub const MAX_CLIPS: usize = 256;
/// Bytes a many-clip request's packed `clips` may hold (each recording also
/// within [`MAX_WAV_BYTES`]).
pub const MAX_CLIPS_BYTES: usize = 512 << 20;

/// Pack recordings (WAV bytes) as a many-clip request's `clips` blob: each one
/// as its byte length (u64 little-endian) followed by its bytes.
pub fn pack_clips(wavs: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(wavs.iter().map(|w| w.len() + 8).sum());
    for w in wavs {
        out.extend_from_slice(&(w.len() as u64).to_le_bytes());
        out.extend_from_slice(w);
    }
    out
}

/// Parse a many-clip request: the packed `clips` blob ([`pack_clips`]) and,
/// when given, `texts`, a JSON array with one transcript (or `null`) per clip.
/// Every recording is checked as [`Request::parse`] checks one; an error names
/// the clip and is tagged `invalid-request`.
pub fn parse_clips(packed: &[u8], texts: Option<&str>) -> Result<Vec<Request>, String> {
    let invalid = |m: String| tag(FailureClass::InvalidRequest, m);
    if packed.len() > MAX_CLIPS_BYTES {
        return Err(invalid(format!("clips: {} bytes is more than the {MAX_CLIPS_BYTES} a request may hold", packed.len())));
    }
    let mut wavs: Vec<&[u8]> = Vec::new();
    let mut rest = packed;
    while !rest.is_empty() {
        let i = wavs.len();
        if i == MAX_CLIPS {
            return Err(invalid(format!("clips: more than the {MAX_CLIPS} recordings a request may carry")));
        }
        let (len, body) = rest.split_at_checked(8).ok_or_else(|| invalid(format!("clips: clip {i}: its length is cut off")))?;
        let len = u64::from_le_bytes(len.try_into().expect("eight bytes"));
        let len = usize::try_from(len).ok().filter(|&n| n <= body.len()).ok_or_else(|| invalid(format!("clips: clip {i}: {len} bytes runs past the end of the blob")))?;
        wavs.push(&body[..len]);
        rest = &body[len..];
    }
    if wavs.is_empty() {
        return Err(invalid("clips: no clips; pack at least one recording".to_string()));
    }
    let texts: Vec<Option<String>> = match texts {
        None => vec![None; wavs.len()],
        Some(t) => {
            let texts: Vec<Option<String>> = serde_json::from_str(t).map_err(|e| invalid(format!("texts: not a JSON array of strings and nulls: {e}")))?;
            if texts.len() != wavs.len() {
                return Err(invalid(format!("texts: {} transcripts for {} clips", texts.len(), wavs.len())));
            }
            texts
        }
    };
    wavs.into_iter().zip(texts).enumerate().map(|(i, (wav, text))| Request::check(wav, text).map_err(|e| invalid(format!("clip {i}: {e}")))).collect()
}

/// Device bytes the widest batch pass holds beyond the weights: the codec's
/// SEANet head over `mimi::ENCODE_PASS_SAMPLES` padded samples dominates (64
/// channels of the input rate, a few such activations alive at once, and the
/// backend's pool of freed blocks); measured on a GH200 at that bound as
/// recorded in the roadmap ledger.
pub const BATCH_ACTIVATION_BYTES: u64 = 12 << 30;

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
        Self::check(wav, text).map_err(|m| tag(FailureClass::InvalidRequest, m))
    }

    /// [`Self::parse`]'s checks, the reason untagged.
    fn check(wav: &[u8], text: Option<String>) -> Result<Request, String> {
        let invalid = |m: String| m;
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
    producers: std::collections::BTreeMap<String, String>,
}

/// The content digest of the files a component is read from: `sha256:` and the SHA-256 of one line per file,
/// `<name>\t<sha256 of its bytes>\n`, in name order, so the same files give the same digest wherever they sit.
fn content_digest(files: &[std::path::PathBuf]) -> Result<String, String> {
    let mut lines: Vec<String> = Vec::with_capacity(files.len());
    for f in files {
        let name = f.file_name().and_then(|n| n.to_str()).ok_or_else(|| format!("{}: not a file name", f.display()))?;
        let digest = brain_modelstore::fetch::sha256_file(f).map_err(|e| format!("{}: {e}", f.display()))?;
        lines.push(format!("{name}\t{digest}\n"));
    }
    lines.sort();
    Ok(format!("sha256:{}", brain_modelstore::fetch::bytes_digest(lines.concat().as_bytes())))
}

/// The files a component at `path` (a checkpoint directory or one weights file) is read from.
fn component_files(path: &str, in_dir: &[&str]) -> Vec<std::path::PathBuf> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        in_dir.iter().map(|f| p.join(f)).collect()
    } else {
        vec![p.to_path_buf()]
    }
}

/// The content digests of what produces a recording's arrays, as a training shard records them
/// (`ShardMeta::producers`): the codec, the speaker encoder and the tokenizer.
pub fn producers(paths: &TtsPaths) -> Result<std::collections::BTreeMap<String, String>, String> {
    let ckpt = std::path::Path::new(&paths.ckpt_dir);
    Ok(std::collections::BTreeMap::from([
        ("codec".to_string(), content_digest(&component_files(&paths.codec, &["config.json", "model.safetensors"]))?),
        ("speaker_encoder".to_string(), content_digest(&component_files(&paths.speaker, &["config.json", "model.safetensors"]))?),
        ("tokenizer".to_string(), content_digest(&["vocab.json", "merges.txt", "tokenizer_config.json"].map(|f| ckpt.join(f)))?),
    ]))
}

impl Encoder {
    /// Load the components of `paths`. The checkpoint must ship a speaker
    /// encoder (a Base checkpoint does).
    pub fn load(paths: &TtsPaths) -> Result<Encoder, String> {
        paths.require(true)?;
        let tokenizer = crate::prompt::load_tokenizer(&paths.ckpt_dir)?;
        // Both encoders on the ambient device (the one the caller placed the
        // encoder on): on a card the codes equal the CPU backend's but for
        // near-ties of the residual argmin (`mimi` `tests/it/encode.rs`).
        let codec = mimi::Codec::load_inference_on(gpu_core::Gpu::new(mimi::PIPELINES), &paths.codec);
        if codec.cfg.semantic_codebook_size.max(codec.cfg.codebook_size) > u32::from(u16::MAX) + 1 {
            return Err("the codec's codebooks are wider than a u16 code".to_string());
        }
        let speaker = ecapatdnn::SpeakerEncoder::load_inference(&paths.speaker);
        Ok(Encoder { codec, speaker, tokenizer, producers: producers(paths)? })
    }

    /// The content digests of the codec, the speaker encoder and the tokenizer ([`producers`]).
    pub fn producers(&self) -> &std::collections::BTreeMap<String, String> {
        &self.producers
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
        self.encode_batch(&[req]).pop().expect("one recording in, one encoding out")
    }

    /// Encode validated recordings together: each resampled to the codec's
    /// rate on the host (in parallel), the codec and the speaker encoder each
    /// one batched device pass (`mimi::Codec::encode_batch`,
    /// `ecapatdnn::SpeakerEncoder::embed_batch`, its log-mel front end on the
    /// host during the codec's pass), reading back only the codes and the
    /// x-vectors. Each recording encodes as it does alone.
    pub fn encode_batch(&self, reqs: &[&Request]) -> Vec<Encoded> {
        let rate = self.codec.cfg.input_sample_rate;
        let at_codec_rate: Vec<Vec<f32>> = backend_cpu::par::map(reqs.len(), |i| audio::resample::rational(&reqs[i].wav.samples, rate, reqs[i].wav.sample_rate));
        // The speaker encoder's host front end runs while the codec's pass
        // occupies the device.
        let (codes, mels) = std::thread::scope(|s| {
            let mels = s.spawn(|| backend_cpu::par::map(reqs.len(), |i| self.speaker.log_mel(&reqs[i].wav.samples, reqs[i].wav.sample_rate)));
            let codes = self.codec.encode_batch(&at_codec_rate.iter().map(Vec::as_slice).collect::<Vec<_>>());
            (codes, mels.join().expect("the log-mel front end panicked"))
        });
        let xvectors = self.speaker.embed_batch(&mels.iter().map(Vec::as_slice).collect::<Vec<_>>());
        let nq = self.codebooks() as usize;
        reqs.iter()
            .zip(codes.into_iter().zip(xvectors))
            .map(|(req, (codes, xvector))| Encoded {
                frames: codes.len() / nq,
                codes: codes.into_iter().map(|c| c as u16).collect(),
                xvector,
                text_ids: req.text.as_deref().map(|t| self.text_ids(t)),
            })
            .collect()
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
