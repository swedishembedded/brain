// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for accepting untrusted media on
// inference servers for its clients. If your team needs expertise in audio
// pipelines for speech models then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Speech uploads, decoded to what every ASR model takes: 16 kHz mono f32.
//!
//! A client sends a recording as a file: WAV, Ogg/Opus (what brain's own
//! clients and Firefox produce) or WebM/Opus (what Chrome's `MediaRecorder`
//! produces). The container is told
//! from its magic bytes, never from a client's claim. Opus is decoded by
//! libopus straight at 16 kHz, so no resampling follows it.
//!
//! The input is untrusted. Every path is bounded: a recording longer than the
//! caller's `max_seconds` is refused while it is being decoded (an Opus packet
//! holds at most 120 ms, so output grows only with input), an empty packet is
//! malformed rather than handed to libopus as "conceal a loss", and a stream
//! that is not Opus (or is multichannel Opus beyond stereo) is unsupported.

use std::fmt;

use crate::asr_caps::ASR_SAMPLE_RATE;

/// What [`decode_speech`] accepts, for an error message a client can act on.
pub const ACCEPTED_FORMATS: &str = "wav, ogg (opus), webm (opus)";

/// A recording's container, from its magic bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Wav,
    Ogg,
    WebM,
}

/// Why a recording could not be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Not a container or codec this build decodes.
    Unsupported(String),
    /// The container or the codec data is broken.
    Malformed(String),
    /// Longer than the caller allows.
    TooLong { max_seconds: u32 },
    /// Decodes to no audio at all.
    Empty,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Unsupported(what) => write!(f, "unsupported audio: {what}; accepted formats are {ACCEPTED_FORMATS}"),
            DecodeError::Malformed(what) => write!(f, "malformed audio: {what}"),
            DecodeError::TooLong { max_seconds } => write!(f, "audio is longer than {max_seconds} s"),
            DecodeError::Empty => write!(f, "audio is empty"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// The container `bytes` is in, if it is one [`decode_speech`] knows.
pub fn sniff(bytes: &[u8]) -> Option<Format> {
    if crate::asr_caps::is_wav(bytes) {
        Some(Format::Wav)
    } else if bytes.starts_with(b"OggS") {
        Some(Format::Ogg)
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some(Format::WebM)
    } else {
        None
    }
}

/// Decode a recording to 16 kHz mono f32 samples, refusing anything longer
/// than `max_seconds`.
pub fn decode_speech(bytes: &[u8], max_seconds: u32) -> Result<Vec<f32>, DecodeError> {
    let limit = max_seconds as usize * ASR_SAMPLE_RATE as usize;
    let samples = match sniff(bytes) {
        Some(Format::Wav) => wav(bytes, max_seconds)?,
        Some(Format::Ogg) => compressed::ogg(bytes, limit, max_seconds)?,
        Some(Format::WebM) => compressed::webm(bytes, limit, max_seconds)?,
        None => return Err(DecodeError::Unsupported("not a recognised container".into())),
    };
    if samples.is_empty() {
        return Err(DecodeError::Empty);
    }
    Ok(samples)
}

fn wav(bytes: &[u8], max_seconds: u32) -> Result<Vec<f32>, DecodeError> {
    let wav = crate::wav::parse(bytes).map_err(|e| DecodeError::Malformed(e.to_string()))?;
    if !(8_000..=192_000).contains(&wav.sample_rate) {
        return Err(DecodeError::Unsupported(format!("WAV sample rate {} Hz", wav.sample_rate)));
    }
    if wav.samples.len() as u64 > max_seconds as u64 * wav.sample_rate as u64 {
        return Err(DecodeError::TooLong { max_seconds });
    }
    Ok(to_speech_rate(&wav.samples, wav.sample_rate))
}

/// Resample mono samples at `rate` to the recogniser's 16 kHz through the
/// Kaiser-windowed polyphase filter ([`crate::resample::rational`]), so that
/// content above 8 kHz is removed rather than folded into the band. The one
/// conversion every WAV surface uses.
pub fn to_speech_rate(samples: &[f32], rate: u32) -> Vec<f32> {
    crate::resample::rational(samples, ASR_SAMPLE_RATE, rate)
}

mod compressed {
    use std::io::Cursor;

    use matroska_demuxer::{Frame, MatroskaFile, TrackType};
    use ::opus::{Channels, Decoder};

    use super::DecodeError;
    use crate::asr_caps::ASR_SAMPLE_RATE;

    /// Opus counts pre-skip and granule positions at 48 kHz.
    const OPUS_RATE_RATIO: u64 = 48_000 / ASR_SAMPLE_RATE as u64;
    /// The longest Opus packet, 120 ms, at 16 kHz.
    const MAX_PACKET_FRAMES: usize = ASR_SAMPLE_RATE as usize * 120 / 1000;

    /// The `OpusHead` identification header (RFC 7845 section 5.1).
    struct Head {
        channels: usize,
        pre_skip: u64,
    }

    fn head(packet: &[u8]) -> Result<Head, DecodeError> {
        if packet.len() < 19 || &packet[..8] != b"OpusHead" {
            return Err(DecodeError::Unsupported("the stream is not Opus".into()));
        }
        if packet[8] & 0xf0 != 0 {
            return Err(DecodeError::Unsupported(format!("Opus header version {}", packet[8])));
        }
        let channels = packet[9] as usize;
        let mapping_family = packet[18];
        if mapping_family != 0 || !(1..=2).contains(&channels) {
            return Err(DecodeError::Unsupported(format!("Opus with {channels} channels (mapping family {mapping_family})")));
        }
        Ok(Head { channels, pre_skip: u16::from_le_bytes([packet[10], packet[11]]) as u64 })
    }

    /// Opus packets in, mono 16 kHz out, the decoder's pre-skip dropped.
    struct Stream {
        decoder: Decoder,
        channels: usize,
        skip: usize,
        frame: Vec<f32>,
        out: Vec<f32>,
        limit: usize,
        max_seconds: u32,
    }

    impl Stream {
        fn new(head: &Head, limit: usize, max_seconds: u32) -> Result<Stream, DecodeError> {
            let channels = if head.channels == 2 { Channels::Stereo } else { Channels::Mono };
            let decoder = Decoder::new(ASR_SAMPLE_RATE, channels).map_err(|e| DecodeError::Malformed(e.to_string()))?;
            Ok(Stream {
                decoder,
                channels: head.channels,
                skip: (head.pre_skip / OPUS_RATE_RATIO) as usize,
                frame: vec![0.0; MAX_PACKET_FRAMES * head.channels],
                out: Vec::new(),
                limit,
                max_seconds,
            })
        }

        fn packet(&mut self, data: &[u8]) -> Result<(), DecodeError> {
            if data.is_empty() {
                return Err(DecodeError::Malformed("an empty Opus packet".into()));
            }
            let frames = self.decoder.decode_float(data, &mut self.frame, false).map_err(|e| DecodeError::Malformed(e.to_string()))?;
            for i in 0..frames {
                let sample = match self.channels {
                    2 => (self.frame[2 * i] + self.frame[2 * i + 1]) * 0.5,
                    _ => self.frame[i],
                };
                if self.skip > 0 {
                    self.skip -= 1;
                } else {
                    self.out.push(sample);
                }
            }
            if self.out.len() > self.limit {
                return Err(DecodeError::TooLong { max_seconds: self.max_seconds });
            }
            Ok(())
        }
    }

    pub(super) fn ogg(bytes: &[u8], limit: usize, max_seconds: u32) -> Result<Vec<f32>, DecodeError> {
        let malformed = |e: ogg::OggReadError| DecodeError::Malformed(e.to_string());
        let mut reader = ogg::PacketReader::new(Cursor::new(bytes));
        let first = reader.read_packet().map_err(malformed)?.ok_or(DecodeError::Empty)?;
        let head = head(&first.data)?;
        // Only the first logical stream is the recording; another multiplexed
        // beside it is ignored rather than mixed in.
        let serial = first.stream_serial();
        let mut stream = Stream::new(&head, limit, max_seconds)?;
        let mut tags_seen = false;
        let mut last_granule = None;
        while let Some(packet) = reader.read_packet().map_err(malformed)? {
            if packet.stream_serial() != serial {
                continue;
            }
            if !tags_seen {
                if !packet.data.starts_with(b"OpusTags") {
                    return Err(DecodeError::Malformed("OpusTags header missing".into()));
                }
                tags_seen = true;
                continue;
            }
            stream.packet(&packet.data)?;
            last_granule = Some(packet.absgp_page());
        }
        // The last page's granule position marks where the recording really
        // ends; the encoder pads the final packet past it.
        let mut out = stream.out;
        if let Some(end) = last_granule.and_then(|g| g.checked_sub(head.pre_skip)) {
            out.truncate((end / OPUS_RATE_RATIO) as usize);
        }
        Ok(out)
    }

    pub(super) fn webm(bytes: &[u8], limit: usize, max_seconds: u32) -> Result<Vec<f32>, DecodeError> {
        let malformed = |e: matroska_demuxer::DemuxError| DecodeError::Malformed(e.to_string());
        let mut file = MatroskaFile::open(Cursor::new(bytes)).map_err(malformed)?;
        let track = file
            .tracks()
            .iter()
            .find(|t| t.track_type() == TrackType::Audio && t.codec_id() == "A_OPUS")
            .ok_or_else(|| DecodeError::Unsupported("WebM without an Opus audio track".into()))?;
        let number = track.track_number().get();
        let head = head(track.codec_private().ok_or_else(|| DecodeError::Malformed("Opus track without its OpusHead".into()))?)?;
        let mut stream = Stream::new(&head, limit, max_seconds)?;
        let mut frame = Frame::default();
        while file.next_frame(&mut frame).map_err(malformed)? {
            if frame.track == number {
                stream.packet(&frame.data)?;
            }
        }
        Ok(stream.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One second of a 440 Hz tone at 16 kHz.
    fn tone() -> Vec<f32> {
        (0..16_000).map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin()).collect()
    }

    fn rms(s: &[f32]) -> f32 {
        (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
    }

    /// Positive-going zero crossings per second: the tone's frequency.
    fn frequency(s: &[f32]) -> f32 {
        let crossings = s.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        crossings as f32 * 16_000.0 / s.len() as f32
    }

    #[test]
    fn wav_decodes_at_any_rate_to_16_khz() {
        let at_48k: Vec<f32> = (0..48_000).map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin()).collect();
        let out = decode_speech(&crate::wav::encode(&at_48k, 48_000), 30).unwrap();
        assert!((out.len() as i64 - 16_000).abs() <= 1, "{}", out.len());
        assert!((frequency(&out) - 440.0).abs() < 5.0);
    }

    /// Content above the 8 kHz the recogniser can represent must be filtered
    /// out on the way down, not folded into the band it reads: a 10 kHz
    /// whistle in a 48 kHz upload would otherwise arrive as a 6 kHz tone.
    #[test]
    fn a_wav_is_filtered_before_it_is_decimated() {
        let whistle: Vec<f32> = (0..48_000).map(|i| 0.5 * (2.0 * std::f32::consts::PI * 10_000.0 * i as f32 / 48_000.0).sin()).collect();
        let out = decode_speech(&crate::wav::encode(&whistle, 48_000), 30).unwrap();
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms < 0.01 * 0.5 / std::f32::consts::SQRT_2, "a 10 kHz tone survived the drop to 16 kHz at rms {rms}");
    }

    #[test]
    fn refuses_what_it_cannot_or_must_not_decode() {
        assert!(matches!(decode_speech(b"ID3\x04 an mp3", 30), Err(DecodeError::Unsupported(_))));
        assert!(matches!(decode_speech(&crate::wav::encode(&tone(), 16_000), 0), Err(DecodeError::TooLong { max_seconds: 0 })));
        assert_eq!(decode_speech(&crate::wav::encode(&[], 16_000), 30), Err(DecodeError::Empty));
        assert!(matches!(decode_speech(&crate::wav::encode(&tone(), 4_000), 30), Err(DecodeError::Unsupported(_))));
        let message = decode_speech(b"nonsense", 30).unwrap_err().to_string();
        assert!(message.contains(ACCEPTED_FORMATS), "{message}");
    }

    mod opus_tests {
        use super::*;

        /// 20 ms Opus packets of `samples` (16 kHz mono), as libopus makes them.
        fn encode(samples: &[f32]) -> Vec<Vec<u8>> {
            let mut enc = ::opus::Encoder::new(16_000, ::opus::Channels::Mono, ::opus::Application::Voip).unwrap();
            let mut padded = samples.to_vec();
            padded.resize(samples.len().div_ceil(320) * 320, 0.0);
            padded.chunks(320).map(|frame| enc.encode_vec_float(frame, 4000).unwrap()).collect()
        }

        fn opus_head(channels: u8, pre_skip: u16) -> Vec<u8> {
            let mut h = b"OpusHead".to_vec();
            h.push(1);
            h.push(channels);
            h.extend_from_slice(&pre_skip.to_le_bytes());
            h.extend_from_slice(&16_000u32.to_le_bytes());
            h.extend_from_slice(&[0, 0, 0]); // output gain, mapping family 0
            h
        }

        fn ogg_file(packets: &[Vec<u8>], head: Vec<u8>) -> Vec<u8> {
            let mut out = Vec::new();
            {
                let mut w = ogg::PacketWriter::new(&mut out);
                w.write_packet(head, 7, ogg::PacketWriteEndInfo::EndPage, 0).unwrap();
                w.write_packet(b"OpusTags\0\0\0\0\0\0\0\0".to_vec(), 7, ogg::PacketWriteEndInfo::EndPage, 0).unwrap();
                for (i, p) in packets.iter().enumerate() {
                    let end = if i + 1 == packets.len() { ogg::PacketWriteEndInfo::EndStream } else { ogg::PacketWriteEndInfo::NormalPacket };
                    w.write_packet(p.clone(), 7, end, (i as u64 + 1) * 960).unwrap();
                }
            }
            out
        }

        /// An EBML element: id, then the size as an 8-byte vint (or "unknown").
        fn element(id: &[u8], body: &[u8]) -> Vec<u8> {
            let mut e = id.to_vec();
            e.push(0x01);
            e.extend_from_slice(&(body.len() as u64).to_be_bytes()[1..]);
            e.extend_from_slice(body);
            e
        }

        /// A WebM file shaped like a live `MediaRecorder` stream: the Segment
        /// and Cluster sizes are unknown, the blocks are SimpleBlocks.
        fn webm_file(packets: &[Vec<u8>]) -> Vec<u8> {
            let unknown = [0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
            let mut f = element(&[0x1a, 0x45, 0xdf, 0xa3], &[element(&[0x42, 0x82], b"webm"), element(&[0x42, 0x87], &[4]), element(&[0x42, 0x85], &[2])].concat());
            let audio = element(&[0xe1], &[element(&[0xb5], &48_000f64.to_be_bytes()), element(&[0x9f], &[1])].concat());
            let entry = element(
                &[0xae],
                &[element(&[0xd7], &[1]), element(&[0x73, 0xc5], &[9]), element(&[0x83], &[2]), element(&[0x86], b"A_OPUS"), element(&[0x63, 0xa2], &opus_head(1, 312)), audio].concat(),
            );
            let info = element(
                &[0x15, 0x49, 0xa9, 0x66],
                &[element(&[0x2a, 0xd7, 0xb1], &1_000_000u32.to_be_bytes()), element(&[0x4d, 0x80], b"Chrome"), element(&[0x57, 0x41], b"Chrome")].concat(),
            );
            let tracks = element(&[0x16, 0x54, 0xae, 0x6b], &entry);
            let mut cluster = vec![0x1f, 0x43, 0xb6, 0x75];
            cluster.extend_from_slice(&unknown);
            cluster.extend(element(&[0xe7], &[0]));
            for (i, p) in packets.iter().enumerate() {
                let mut block = vec![0x81]; // track 1
                block.extend_from_slice(&((i * 20) as i16).to_be_bytes());
                block.push(0x80); // keyframe, no lacing
                block.extend_from_slice(p);
                cluster.extend(element(&[0xa3], &block));
            }
            f.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
            f.extend_from_slice(&unknown);
            f.extend(info);
            f.extend(tracks);
            f.extend(cluster);
            f
        }

        fn assert_is_tone(out: &[f32]) {
            assert!((out.len() as i64 - 16_000).abs() <= 400, "{} samples", out.len());
            let steady = &out[1600..out.len() - 1600];
            assert!((frequency(steady) - 440.0).abs() < 5.0, "{} Hz", frequency(steady));
            assert!((rms(steady) / rms(&tone()) - 1.0).abs() < 0.2, "rms {}", rms(steady));
        }

        #[test]
        fn ogg_opus_decodes_to_the_recorded_signal() {
            let out = decode_speech(&ogg_file(&encode(&tone()), opus_head(1, 312)), 30).unwrap();
            assert_is_tone(&out);
            // The final granule position trims the encoder's padding exactly.
            assert_eq!(out.len(), (50 * 960 - 312) / 3);
        }

        #[test]
        fn webm_opus_from_a_live_recorder_decodes() {
            assert_is_tone(&decode_speech(&webm_file(&encode(&tone())), 30).unwrap());
        }

        #[test]
        fn opus_is_bounded_and_validated() {
            let packets = encode(&tone());
            assert_eq!(decode_speech(&ogg_file(&packets, opus_head(1, 0)), 0), Err(DecodeError::TooLong { max_seconds: 0 }));
            assert!(matches!(decode_speech(&ogg_file(&packets, opus_head(6, 0)), 30), Err(DecodeError::Unsupported(_))));
            let mut with_empty = packets.clone();
            with_empty.insert(3, Vec::new());
            assert!(matches!(decode_speech(&ogg_file(&with_empty, opus_head(1, 0)), 30), Err(DecodeError::Malformed(_))));
            let mut not_opus = ogg_file(&packets, opus_head(1, 0));
            let at = not_opus.windows(8).position(|w| w == b"OpusHead").unwrap();
            not_opus[at..at + 8].copy_from_slice(b"Vorbis\0\0");
            assert!(decode_speech(&not_opus, 30).is_err());
            let truncated = &webm_file(&packets)[..200];
            assert!(decode_speech(truncated, 30).is_err());
        }
    }
}
