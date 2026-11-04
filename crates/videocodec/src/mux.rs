// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The coded streams a container is written from, and the Opus coding of a
//! sound track.

use crate::{Audio, Error};

/// Opus codes 20 ms packets here: 960 samples at its 48 kHz clock.
pub(crate) const OPUS_FRAME_48K: usize = 960;

pub(crate) struct VideoPacket {
    pub data: Vec<u8>,
    pub key: bool,
}

/// Coded AV1 in presentation order, one temporal unit per frame.
pub(crate) struct VideoStream {
    pub w: usize,
    pub h: usize,
    /// Frames per second as `num / den`.
    pub fps: (u64, u64),
    /// The `av1C` record.
    pub config: Vec<u8>,
    pub packets: Vec<VideoPacket>,
}

/// Coded Opus, one 20 ms packet each.
pub(crate) struct OpusStream {
    pub channels: u8,
    /// Samples at 48 kHz a decoder drops from the start (the encoder's lookahead).
    pub pre_skip: u16,
    /// The rate the sound was given at, recorded for players that care.
    pub input_rate: u32,
    pub packets: Vec<Vec<u8>>,
}

impl OpusStream {
    /// The `OpusHead` identification header (RFC 7845 section 5.1): Matroska's CodecPrivate.
    pub fn head(&self) -> Vec<u8> {
        let mut h = b"OpusHead".to_vec();
        h.push(1);
        h.push(self.channels);
        h.extend(self.pre_skip.to_le_bytes());
        h.extend(self.input_rate.to_le_bytes());
        h.extend(0i16.to_le_bytes());
        h.push(0);
        h
    }

    /// Samples per channel at 48 kHz the packets decode to, before the pre-skip is dropped.
    pub fn samples_48k(&self) -> u64 {
        (self.packets.len() * OPUS_FRAME_48K) as u64
    }

    /// `audio` coded at 48 kHz. One or two channels; any input rate is
    /// resampled to 48 kHz first (linear interpolation, which is transparent
    /// for the generated rates this carries: 16-44.1 kHz speech and music).
    pub fn encode(audio: &Audio) -> Result<OpusStream, Error> {
        let channels = audio.channels.len();
        if !(1..=2).contains(&channels) {
            return Err(Error::Invalid(format!("Opus here carries one or two channels, not {channels}")));
        }
        let len = audio.channels[0].len();
        if audio.channels.iter().any(|c| c.len() != len) {
            return Err(Error::Invalid("the sound track's channels differ in length".into()));
        }
        if audio.sample_rate == 0 {
            return Err(Error::Invalid("a sound track at 0 Hz".into()));
        }
        let planes: Vec<Vec<f32>> = audio.channels.iter().map(|c| resample_linear(c, audio.sample_rate, 48_000)).collect();
        let n = planes[0].len();
        let mode = if channels == 2 { opus::Channels::Stereo } else { opus::Channels::Mono };
        let mut enc = opus::Encoder::new(48_000, mode, opus::Application::Audio).map_err(|e| Error::Invalid(format!("Opus encoder: {e}")))?;
        enc.set_bitrate(opus::Bitrate::Bits(64_000 * channels as i32)).map_err(|e| Error::Invalid(format!("Opus encoder: {e}")))?;
        let pre_skip = enc.get_lookahead().map_err(|e| Error::Invalid(format!("Opus encoder: {e}")))? as usize;
        // The lookahead is dropped at decode, so as many samples are appended to keep the end.
        let total = n + pre_skip;
        let mut packets = Vec::with_capacity(total.div_ceil(OPUS_FRAME_48K));
        let mut frame = vec![0f32; OPUS_FRAME_48K * channels];
        let mut out = vec![0u8; 4000];
        for start in (0..total).step_by(OPUS_FRAME_48K) {
            for i in 0..OPUS_FRAME_48K {
                for (c, plane) in planes.iter().enumerate() {
                    frame[i * channels + c] = plane.get(start + i).copied().unwrap_or(0.0);
                }
            }
            let len = enc.encode_float(&frame, &mut out).map_err(|e| Error::Invalid(format!("Opus encoder: {e}")))?;
            packets.push(out[..len].to_vec());
        }
        Ok(OpusStream { channels: channels as u8, pre_skip: pre_skip as u16, input_rate: audio.sample_rate, packets })
    }
}

/// `x` at `to` Hz from `from` Hz by linear interpolation.
fn resample_linear(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * to as u64).div_ceil(from as u64) as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 * from as f64 / to as f64;
            let j = t.floor() as usize;
            let f = (t - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}
