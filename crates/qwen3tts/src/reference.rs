// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reference voice a clone is conditioned on, and the codes file an
//! externally encoded reference is handed over as.
//!
//! A [`Reference`] carries the decoded audio plus a digest of the bytes it was
//! read from, so everything derived from it (the speaker x-vector, the
//! reference's own codec codes) is cached by what the audio IS, never by where
//! it happened to be read from: a clip that arrives over a socket as a blob
//! and the same clip read off disk are the same voice.
//!
//! Swedish Embedded AB implements voice-cloning speech services for its
//! clients. If your team needs expertise in conditioning speech synthesis on a
//! reference voice then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

/// Decoded reference audio and the digest of its source bytes.
pub struct Reference {
    pub wav: audio::wav::Wav,
    digest: String,
}

impl Reference {
    /// Decode a WAV held in memory (a served blob, or a file already read).
    pub fn from_bytes(bytes: &[u8]) -> Result<Reference, String> {
        let wav = audio::wav::parse(bytes).map_err(|e| format!("reference audio: {e}"))?;
        Ok(Reference { wav, digest: digest(bytes) })
    }

    /// Read and decode the WAV at `path`.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Reference, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// A stable name for this audio's content, safe in a file name.
    pub fn digest(&self) -> &str {
        &self.digest
    }
}

/// FNV-1a over `bytes`, as 16 hex digits: a cache key, not a security
/// boundary.
fn digest(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// Cosine similarity of two embeddings; 0 when either has no length, so a
/// degenerate (silent) utterance reads as "no similarity" rather than NaN.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let (na, nb) = (norm(a), norm(b));
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// Speaker-embedding cosine similarity of two utterances (ECAPA x-vectors from
/// `speaker`'s encoder weights): the timbre-preservation metric, e.g. between
/// a quantized Talker's clone and its reference. Each utterance is embedded at
/// its own sample rate; the encoder resamples internally.
pub fn speaker_similarity(speaker: &str, a: &Reference, b: &Reference) -> f32 {
    let encoder = ecapatdnn::SpeakerEncoder::load_inference(speaker);
    cosine(&encoder.embed_wav(&a.wav.samples, a.wav.sample_rate), &encoder.embed_wav(&b.wav.samples, b.wav.sample_rate))
}

/// Reference codec codes from a file: an 8-byte little-endian value count,
/// then that many little-endian `u32` codes (`[frames, 16]` row-major).
pub fn parse_codes(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let header: [u8; 8] = bytes.get(..8).and_then(|h| h.try_into().ok()).ok_or("codes file too short")?;
    let count = usize::try_from(u64::from_le_bytes(header)).map_err(|_| "codes file count does not fit this platform")?;
    let body = &bytes[8..];
    if count.checked_mul(4).is_none_or(|need| body.len() < need) {
        return Err("codes file truncated".to_string());
    }
    Ok(body.as_chunks::<4>().0.iter().take(count).map(|c| u32::from_le_bytes(*c)).collect())
}

/// The inverse of [`parse_codes`].
pub fn encode_codes(codes: &[u32]) -> Vec<u8> {
    let mut out = (codes.len() as u64).to_le_bytes().to_vec();
    out.extend(codes.iter().flat_map(|c| c.to_le_bytes()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_through_the_file_format() {
        let codes: Vec<u32> = (0..32).map(|i| i * 77).collect();
        assert_eq!(parse_codes(&encode_codes(&codes)).unwrap(), codes);
    }

    #[test]
    fn cosine_is_one_for_parallel_zero_for_orthogonal_and_defined_for_silence() {
        assert!((cosine(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 3.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 2.0], &[-1.0, -2.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn a_short_or_truncated_codes_file_is_refused_not_misread() {
        assert!(parse_codes(&[1, 2, 3]).unwrap_err().contains("too short"));
        let mut bytes = encode_codes(&[1, 2, 3, 4]);
        bytes.truncate(bytes.len() - 1);
        assert!(parse_codes(&bytes).unwrap_err().contains("truncated"));
        // A count that overflows the byte size is a truncation, never a panic.
        assert!(parse_codes(&u64::MAX.to_le_bytes()).is_err());
    }

    #[test]
    fn the_digest_names_the_audio_content() {
        let a = audio::wav::encode(&[0.0, 0.25, -0.25, 0.5], 24_000);
        let b = audio::wav::encode(&[0.0, 0.25, -0.25, 0.75], 24_000);
        let (ra, rb) = (Reference::from_bytes(&a).unwrap(), Reference::from_bytes(&b).unwrap());
        assert_eq!(ra.digest(), Reference::from_bytes(&a).unwrap().digest());
        assert_ne!(ra.digest(), rb.digest());
        assert_eq!(ra.wav.sample_rate, 24_000);
        assert!(Reference::from_bytes(b"not a wav").is_err());
    }
}
