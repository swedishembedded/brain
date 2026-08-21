// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal dependency-light WAV I/O (canonical PCM RIFF) for the TTS stack.
//! Reads 16/24/32-bit integer PCM and 32-bit float; writes 16-bit PCM. Samples
//! are normalized f32 in roughly [-1, 1]. Multi-channel input is downmixed to
//! mono (mean) since the codec/speaker paths are mono at 24 kHz.

use std::io::{self, Read, Write};
use std::path::Path;

/// Decoded mono audio.
pub struct Wav {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Parse a WAV byte buffer into mono f32 samples.
pub fn parse(bytes: &[u8]) -> io::Result<Wav> {
    let err = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(err("not a RIFF/WAVE file"));
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32le(&bytes[pos + 4..pos + 8]) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + sz).min(bytes.len());
        match id {
            b"fmt " if sz >= 16 => {
                let b = &bytes[body_start..body_end];
                fmt = Some((u16le(&b[0..2]), u16le(&b[2..4]), u32le(&b[4..8]), u16le(&b[14..16])));
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        pos = body_end + (sz & 1); // chunks are word-aligned
    }
    let (format, channels, rate, bits) = fmt.ok_or_else(|| err("missing fmt chunk"))?;
    let data = data.ok_or_else(|| err("missing data chunk"))?;
    let ch = channels.max(1) as usize;
    let bytes_per = (bits / 8) as usize;
    if bytes_per == 0 {
        return Err(err("zero bit depth"));
    }
    let frames = data.len() / (bytes_per * ch);
    let mut samples = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0.0f32;
        for c in 0..ch {
            let o = (f * ch + c) * bytes_per;
            let s = &data[o..o + bytes_per];
            let v = match (format, bits) {
                (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
                (1, 24) => {
                    let i = ((s[0] as i32) | ((s[1] as i32) << 8) | ((s[2] as i32) << 16)) << 8 >> 8;
                    i as f32 / 8_388_608.0
                }
                (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2_147_483_648.0,
                (1, 8) => (s[0] as f32 - 128.0) / 128.0,
                _ => return Err(err("unsupported PCM format/bit depth")),
            };
            acc += v;
        }
        samples.push(acc / ch as f32);
    }
    Ok(Wav { sample_rate: rate, samples })
}

/// Read a WAV file into mono f32 samples.
pub fn read(path: impl AsRef<Path>) -> io::Result<Wav> {
    let mut f = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    parse(&bytes)
}

/// Encode mono f32 samples (clamped to [-1,1]) as a 16-bit PCM WAV byte buffer.
pub fn encode(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    encode_multi(&[samples], sample_rate)
}

/// Write mono f32 samples as a 16-bit PCM WAV file.
pub fn write(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> io::Result<()> {
    let bytes = encode(samples, sample_rate);
    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)
}

/// Encode multi-channel f32 samples as an interleaved 16-bit PCM WAV byte
/// buffer. `channels` holds one contiguous, equal-length slice per channel
/// (planar - e.g. a vocoder's own `[channels, samples]` output row-major,
/// no interleaving needed from the caller); `channels.len()` becomes the
/// WAV's channel count (2 for the stereo vocoders this exists for, but not
/// hardcoded to 2 - a mono `encode` is just `encode_multi(&[samples], ...)`).
pub fn encode_multi(channels: &[&[f32]], sample_rate: u32) -> Vec<u8> {
    assert!(!channels.is_empty(), "wav::encode_multi: at least one channel is required");
    let n_channels = channels.len();
    let frames = channels[0].len();
    for c in channels {
        assert_eq!(c.len(), frames, "wav::encode_multi: every channel must have the same length");
    }
    let block_align = 2 * n_channels as u32;
    let data_len = frames * n_channels * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    let byte_rate = sample_rate * block_align;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&(n_channels as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for f in 0..frames {
        for c in channels {
            let v = (c[f].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Write multi-channel f32 samples ([`encode_multi`]'s own planar layout)
/// as an interleaved 16-bit PCM WAV file.
pub fn write_multi(path: impl AsRef<Path>, channels: &[&[f32]], sample_rate: u32) -> io::Result<()> {
    let bytes = encode_multi(channels, sample_rate);
    let mut f = std::fs::File::create(path)?;
    f.write_all(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_encode_is_unchanged_by_the_encode_multi_refactor() {
        let samples = [0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode(&samples, 44100);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16le(&bytes[22..24]), 1, "channel count");
        assert_eq!(u32le(&bytes[24..28]), 44100, "sample rate");
        assert_eq!(u16le(&bytes[32..34]), 2, "block align");
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed.sample_rate, 44100);
        assert_eq!(parsed.samples.len(), samples.len());
    }

    #[test]
    fn encode_multi_interleaves_channels_in_frame_order() {
        let left = [1.0f32, -1.0, 0.0];
        let right = [-1.0f32, 1.0, 0.0];
        let bytes = encode_multi(&[&left, &right], 8000);
        assert_eq!(u16le(&bytes[22..24]), 2, "channel count");
        assert_eq!(u16le(&bytes[32..34]), 4, "block align (2 channels * 2 bytes)");
        let data = &bytes[44..];
        // Frame 0: left=1.0 -> 32767, right=-1.0 -> -32767.
        assert_eq!(i16::from_le_bytes([data[0], data[1]]), 32767);
        assert_eq!(i16::from_le_bytes([data[2], data[3]]), -32767);
        // Frame 1: left=-1.0 -> -32767, right=1.0 -> 32767.
        assert_eq!(i16::from_le_bytes([data[4], data[5]]), -32767);
        assert_eq!(i16::from_le_bytes([data[6], data[7]]), 32767);
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed.samples.len(), 3, "3 frames regardless of channel count");
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn encode_multi_rejects_mismatched_channel_lengths() {
        let left = [0.0f32, 0.0];
        let right = [0.0f32];
        encode_multi(&[&left, &right], 8000);
    }
}
