// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! WebM and Matroska: AV1 video (`V_AV1`, `av1C` as CodecPrivate) with Opus
//! sound (`A_OPUS`, `OpusHead` as CodecPrivate), written in one pass over
//! the coded streams, and read back through `matroska-demuxer`.
//!
//! The writer interleaves the two tracks by time in clusters that start at
//! every video key frame (or every 30 s, inside the reach of a block's 16-bit
//! relative timestamp), at a 1 ms timestamp scale. Every element size is
//! known when it is written, so the file needs no seeking back.

use std::io::{Read, Seek};
use std::path::Path;

use crate::mux::{OpusStream, VideoStream, OPUS_FRAME_48K};
use crate::{Error, Sample, SampleSource, StreamInfo};

/// Which Matroska profile a file declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocType {
    WebM,
    Matroska,
}

/// Longest cluster, in ms: a block's timestamp is a signed 16-bit offset from its cluster's.
const MAX_CLUSTER_MS: u64 = 30_000;

pub(crate) fn write(out: &mut Vec<u8>, doc: DocType, video: &VideoStream, audio: Option<&OpusStream>) {
    let header = element(0x1A45DFA3, &[
        uint(0x4286, 1),
        uint(0x42F7, 1),
        uint(0x42F2, 4),
        uint(0x42F3, 8),
        string(0x4282, if doc == DocType::WebM { "webm" } else { "matroska" }),
        uint(0x4287, 4),
        uint(0x4285, 2),
    ].concat());

    let frame_ns = |i: u64| i * 1_000_000_000 * video.fps.1 / video.fps.0;
    let video_ms = |i: u64| frame_ns(i) / 1_000_000;
    let audio_ms = |i: u64| i * OPUS_FRAME_48K as u64 * 1000 / 48_000;
    let duration_ms = video_ms(video.packets.len() as u64).max(audio.map_or(0, |a| audio_ms(a.packets.len() as u64)));

    let info = element(0x1549A966, &[
        uint(0x2AD7B1, 1_000_000),
        string(0x4D80, "brain"),
        string(0x5741, "brain"),
        float(0x4489, duration_ms as f64),
    ].concat());

    let colour = element(0x55B0, &[uint(0x55B1, 1), uint(0x55B9, 1), uint(0x55BA, 1), uint(0x55BB, 1)].concat());
    let mut tracks = element(0xAE, &[
        uint(0xD7, 1),
        uint(0x73C5, 1),
        uint(0x83, 1),
        string(0x86, "V_AV1"),
        bytes(0x63A2, &video.config),
        uint(0x23E383, frame_ns(1)),
        element(0xE0, &[uint(0xB0, video.w as u64), uint(0xBA, video.h as u64), colour].concat()),
    ].concat());
    if let Some(a) = audio {
        tracks.extend(element(0xAE, &[
            uint(0xD7, 2),
            uint(0x73C5, 2),
            uint(0x83, 2),
            string(0x86, "A_OPUS"),
            bytes(0x63A2, &a.head()),
            uint(0x56AA, a.pre_skip as u64 * 1_000_000_000 / 48_000),
            uint(0x56BB, 80_000_000),
            element(0xE1, &[float(0xB5, 48_000.0), uint(0x9F, a.channels as u64)].concat()),
        ].concat()));
    }
    let tracks = element(0x1654AE6B, &tracks);

    // Every block in presentation order, video first at equal times.
    let mut blocks: Vec<(u64, u8, bool, &[u8])> = video.packets.iter().enumerate().map(|(i, p)| (video_ms(i as u64), 1, p.key, p.data.as_slice())).collect();
    if let Some(a) = audio {
        blocks.extend(a.packets.iter().enumerate().map(|(i, p)| (audio_ms(i as u64), 2, true, p.as_slice())));
    }
    blocks.sort_by_key(|&(t, track, _, _)| (t, track));

    let mut segment = [info, tracks].concat();
    let mut cluster: Vec<u8> = Vec::new();
    let mut cluster_start = 0u64;
    for (t, track, key, data) in blocks {
        let opens = cluster.is_empty() || (track == 1 && key) || t - cluster_start >= MAX_CLUSTER_MS;
        if opens {
            if !cluster.is_empty() {
                segment.extend(element(0x1F43B675, &cluster));
            }
            cluster_start = t;
            cluster = uint(0xE7, t);
        }
        let mut block = vec![0x80 | track];
        block.extend(((t - cluster_start) as i16).to_be_bytes());
        block.push(if key { 0x80 } else { 0 });
        block.extend_from_slice(data);
        cluster.extend(element(0xA3, &block));
    }
    if !cluster.is_empty() {
        segment.extend(element(0x1F43B675, &cluster));
    }
    out.extend(header);
    out.extend(element(0x18538067, &segment));
}

fn id_bytes(id: u32) -> Vec<u8> {
    let b = id.to_be_bytes();
    let skip = b.iter().take_while(|&&x| x == 0).count();
    b[skip..].to_vec()
}

/// An EBML element size as the shortest variable-length integer that holds it.
fn size(n: u64) -> Vec<u8> {
    let len = (1..=8).find(|&l| n < (1u64 << (7 * l)) - 1).expect("an EBML size fits 56 bits");
    (n | (1u64 << (7 * len))).to_be_bytes()[8 - len..].to_vec()
}

fn element(id: u32, payload: &[u8]) -> Vec<u8> {
    [id_bytes(id), size(payload.len() as u64), payload.to_vec()].concat()
}

fn uint(id: u32, v: u64) -> Vec<u8> {
    let b = v.to_be_bytes();
    let skip = b.iter().take_while(|&&x| x == 0).count().min(7);
    element(id, &b[skip..])
}

fn float(id: u32, v: f64) -> Vec<u8> {
    element(id, &v.to_be_bytes())
}

fn string(id: u32, s: &str) -> Vec<u8> {
    element(id, s.as_bytes())
}

fn bytes(id: u32, b: &[u8]) -> Vec<u8> {
    element(id, b)
}

/// The AV1 video track of a Matroska file, sample by sample.
pub(crate) struct Reader<R: Read + Seek> {
    file: matroska_demuxer::MatroskaFile<R>,
    track: u64,
    scale_ns: f64,
    frame: matroska_demuxer::Frame,
}

/// Open `path` as Matroska; its first video track must be AV1.
pub(crate) fn open(path: &Path) -> Result<(StreamInfo, Reader<std::io::BufReader<std::fs::File>>), Error> {
    let file = std::fs::File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let mkv = matroska_demuxer::MatroskaFile::open(std::io::BufReader::new(file)).map_err(|e| Error::Malformed(format!("{}: {e}", path.display())))?;
    let track = mkv
        .tracks()
        .iter()
        .find(|t| t.track_type() == matroska_demuxer::TrackType::Video)
        .ok_or_else(|| Error::Unsupported(format!("{}: no video track", path.display())))?;
    if track.codec_id() != "V_AV1" {
        return Err(Error::Unsupported(format!("{}: video codec {}", path.display(), track.codec_id())));
    }
    let video = track.video().ok_or_else(|| Error::Malformed(format!("{}: a video track without dimensions", path.display())))?;
    let scale_ns = mkv.info().timestamp_scale().get() as f64;
    let fps = track.default_duration().map(|d| crate::snap_fps(1e9 / d.get() as f64));
    // The segment's duration covers every track (a sound track can run past the last frame), so the video's own
    // length and frame count are left to `probe`, which counts the frames.
    let info = StreamInfo { w: video.pixel_width().get() as u32, h: video.pixel_height().get() as u32, fps, duration: None, frames: None, codec: "av1".into() };
    let track = track.track_number().get();
    Ok((info, Reader { file: mkv, track, scale_ns, frame: Default::default() }))
}

impl<R: Read + Seek> SampleSource for Reader<R> {
    fn next_sample(&mut self) -> Result<Option<Sample>, Error> {
        loop {
            let more = self.file.next_frame(&mut self.frame).map_err(|e| Error::Malformed(format!("matroska: {e}")))?;
            if !more {
                return Ok(None);
            }
            if self.frame.track == self.track {
                // A block without the key flag (a BlockGroup) is not a known random-access point.
                let key = self.frame.is_keyframe.unwrap_or(false);
                return Ok(Some(Sample { time: self.frame.timestamp as f64 * self.scale_ns / 1e9, data: std::mem::take(&mut self.frame.data), key }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_use_the_shortest_marker() {
        assert_eq!(size(0), [0x80]);
        assert_eq!(size(126), [0xFE]);
        // 127 is the one-byte "unknown size" pattern, so it takes two bytes.
        assert_eq!(size(127), [0x40, 0x7F]);
        assert_eq!(size(300), [0x41, 0x2C]);
    }
}
