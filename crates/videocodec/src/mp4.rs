// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MP4 (ISO base media file format): AV1 video (`av01` with `av1C`, per the
//! AV1 ISOBMFF binding) and Opus sound (`Opus` with `dOps`, per the Opus in
//! ISOBMFF encapsulation), written as `ftyp`, `mdat`, `moov`; and the AV1
//! video track of such a file read back from its sample tables.
//!
//! Each coded unit is its own chunk, so the sample-to-chunk table is one
//! entry and every offset is listed. The sound's pre-skip is an edit list,
//! the mechanism the Opus encapsulation names for it.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::mux::{OpusStream, VideoStream, OPUS_FRAME_48K};
use crate::{Error, Sample, SampleSource, StreamInfo};

/// The movie header's clock: milliseconds.
const MOVIE_TIMESCALE: u64 = 1000;

pub(crate) fn write(out: &mut Vec<u8>, video: &VideoStream, audio: Option<&OpusStream>) {
    let ftyp = boxed(b"ftyp", &[b"isom".as_slice(), &0x200u32.to_be_bytes(), b"isom", b"iso2", b"av01", b"mp41"].concat());
    out.extend(&ftyp);

    // mdat: the video units, then the sound packets. Its header is 16 bytes (64-bit size) so the offsets below
    // do not depend on whether the payload passes 4 GiB.
    let payload: usize = video.packets.iter().map(|p| p.data.len()).sum::<usize>() + audio.map_or(0, |a| a.packets.iter().map(Vec::len).sum());
    let mdat_start = out.len() as u64;
    out.extend(1u32.to_be_bytes());
    out.extend(b"mdat");
    out.extend((16 + payload as u64).to_be_bytes());
    let mut offset = mdat_start + 16;
    let mut video_offsets = Vec::with_capacity(video.packets.len());
    for p in &video.packets {
        video_offsets.push(offset);
        out.extend_from_slice(&p.data);
        offset += p.data.len() as u64;
    }
    let mut audio_offsets = Vec::new();
    if let Some(a) = audio {
        for p in &a.packets {
            audio_offsets.push(offset);
            out.extend_from_slice(p);
            offset += p.len() as u64;
        }
    }

    let (fps_num, fps_den) = video.fps;
    let video_media_duration = video.packets.len() as u64 * fps_den;
    let video_ms = video_media_duration * MOVIE_TIMESCALE / fps_num;
    let audio_ms = audio.map_or(0, |a| (a.samples_48k() - a.pre_skip as u64) * MOVIE_TIMESCALE / 48_000);

    let mut moov = mvhd(video_ms.max(audio_ms), if audio.is_some() { 3 } else { 2 });

    // Video track.
    let mut entry = Vec::new();
    entry.extend([0u8; 6]);
    entry.extend(1u16.to_be_bytes());
    entry.extend([0u8; 16]);
    entry.extend((video.w as u16).to_be_bytes());
    entry.extend((video.h as u16).to_be_bytes());
    entry.extend(0x0048_0000u32.to_be_bytes());
    entry.extend(0x0048_0000u32.to_be_bytes());
    entry.extend(0u32.to_be_bytes());
    entry.extend(1u16.to_be_bytes());
    entry.extend([0u8; 32]);
    entry.extend(0x0018u16.to_be_bytes());
    entry.extend((-1i16).to_be_bytes());
    entry.extend(boxed(b"av1C", &video.config));
    // nclx: BT.709 primaries, transfer and matrix, limited range - what the encoder wrote.
    entry.extend(boxed(b"colr", &[b"nclx".as_slice(), &1u16.to_be_bytes(), &1u16.to_be_bytes(), &1u16.to_be_bytes(), &[0u8]].concat()));
    let stsd = full_box(b"stsd", 0, &[1u32.to_be_bytes().as_slice(), &boxed(b"av01", &entry)].concat());
    let keys: Vec<u32> = video.packets.iter().enumerate().filter(|(_, p)| p.key).map(|(i, _)| i as u32 + 1).collect();
    let sizes: Vec<u32> = video.packets.iter().map(|p| p.data.len() as u32).collect();
    let stbl = [stsd, stts(&[(video.packets.len() as u32, fps_den as u32)]), stss(&keys), stsc_one_per_chunk(), stsz(&sizes), co64(&video_offsets)].concat();
    let minf = [full_box(b"vmhd", 1, &[0u8; 8]), dinf(), boxed(b"stbl", &stbl)].concat();
    let mdia = [mdhd(fps_num, video_media_duration), hdlr(b"vide", "brain video"), boxed(b"minf", &minf)].concat();
    moov.extend(boxed(b"trak", &[tkhd(1, video_ms, false, video.w, video.h), boxed(b"mdia", &mdia)].concat()));

    if let Some(a) = audio {
        let mut entry = Vec::new();
        entry.extend([0u8; 6]);
        entry.extend(1u16.to_be_bytes());
        entry.extend([0u8; 8]);
        entry.extend((a.channels as u16).to_be_bytes());
        entry.extend(16u16.to_be_bytes());
        entry.extend([0u8; 4]);
        entry.extend((48_000u32 << 16).to_be_bytes());
        // dOps: version 0, output channels, pre-skip, input rate, output gain, mapping family 0.
        let dops = [&[0u8, a.channels][..], &a.pre_skip.to_be_bytes(), &a.input_rate.to_be_bytes(), &0i16.to_be_bytes(), &[0u8]].concat();
        entry.extend(boxed(b"dOps", &dops));
        let stsd = full_box(b"stsd", 0, &[1u32.to_be_bytes().as_slice(), &boxed(b"Opus", &entry)].concat());
        let sizes: Vec<u32> = a.packets.iter().map(|p| p.len() as u32).collect();
        let stbl = [stsd, stts(&[(a.packets.len() as u32, OPUS_FRAME_48K as u32)]), stsc_one_per_chunk(), stsz(&sizes), co64(&audio_offsets)].concat();
        let minf = [full_box(b"smhd", 0, &[0u8; 4]), dinf(), boxed(b"stbl", &stbl)].concat();
        let mdia = [mdhd(48_000, a.samples_48k()), hdlr(b"soun", "brain sound"), boxed(b"minf", &minf)].concat();
        // The edit list starts presentation after the pre-skip.
        let elst = full_box(b"elst", 0, &[1u32.to_be_bytes().as_slice(), &(audio_ms as u32).to_be_bytes(), &(a.pre_skip as u32).to_be_bytes(), &0x0001_0000u32.to_be_bytes()].concat());
        moov.extend(boxed(b"trak", &[tkhd(2, audio_ms, true, 0, 0), boxed(b"edts", &elst), boxed(b"mdia", &mdia)].concat()));
    }
    out.extend(boxed(b"moov", &moov));
}

fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    [((8 + payload.len()) as u32).to_be_bytes().as_slice(), kind, payload].concat()
}

fn full_box(kind: &[u8; 4], flags: u32, payload: &[u8]) -> Vec<u8> {
    boxed(kind, &[(flags & 0x00ff_ffff).to_be_bytes().as_slice(), payload].concat())
}

/// The identity transform matrix of `mvhd` and `tkhd`.
fn unity_matrix() -> Vec<u8> {
    [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000].iter().flat_map(|v| v.to_be_bytes()).collect()
}

fn mvhd(duration_ms: u64, next_track: u32) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend([0u8; 8]);
    p.extend((MOVIE_TIMESCALE as u32).to_be_bytes());
    p.extend((duration_ms as u32).to_be_bytes());
    p.extend(0x0001_0000u32.to_be_bytes());
    p.extend(0x0100u16.to_be_bytes());
    p.extend([0u8; 10]);
    p.extend(unity_matrix());
    p.extend([0u8; 24]);
    p.extend(next_track.to_be_bytes());
    full_box(b"mvhd", 0, &p)
}

fn tkhd(id: u32, duration_ms: u64, audio: bool, w: usize, h: usize) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend([0u8; 8]);
    p.extend(id.to_be_bytes());
    p.extend([0u8; 4]);
    p.extend((duration_ms as u32).to_be_bytes());
    p.extend([0u8; 8]);
    p.extend(0u16.to_be_bytes());
    p.extend(0u16.to_be_bytes());
    p.extend((if audio { 0x0100u16 } else { 0 }).to_be_bytes());
    p.extend([0u8; 2]);
    p.extend(unity_matrix());
    p.extend(((w as u32) << 16).to_be_bytes());
    p.extend(((h as u32) << 16).to_be_bytes());
    // Enabled and in the movie.
    full_box(b"tkhd", 0x3, &p)
}

fn mdhd(timescale: u64, duration: u64) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend([0u8; 8]);
    p.extend((timescale as u32).to_be_bytes());
    p.extend((duration as u32).to_be_bytes());
    // Language "und", packed as three 5-bit letters.
    p.extend(0x55C4u16.to_be_bytes());
    p.extend([0u8; 2]);
    full_box(b"mdhd", 0, &p)
}

fn hdlr(kind: &[u8; 4], name: &str) -> Vec<u8> {
    full_box(b"hdlr", 0, &[&[0u8; 4][..], kind, &[0u8; 12], name.as_bytes(), &[0]].concat())
}

fn dinf() -> Vec<u8> {
    // One data reference: this file (`url ` with the self-contained flag).
    boxed(b"dinf", &full_box(b"dref", 0, &[1u32.to_be_bytes().as_slice(), &full_box(b"url ", 1, &[])].concat()))
}

fn stts(runs: &[(u32, u32)]) -> Vec<u8> {
    let mut p = (runs.len() as u32).to_be_bytes().to_vec();
    for (count, delta) in runs {
        p.extend(count.to_be_bytes());
        p.extend(delta.to_be_bytes());
    }
    full_box(b"stts", 0, &p)
}

fn stss(keys: &[u32]) -> Vec<u8> {
    let mut p = (keys.len() as u32).to_be_bytes().to_vec();
    p.extend(keys.iter().flat_map(|k| k.to_be_bytes()));
    full_box(b"stss", 0, &p)
}

fn stsc_one_per_chunk() -> Vec<u8> {
    full_box(b"stsc", 0, &[1u32, 1, 1, 1].iter().flat_map(|v| v.to_be_bytes()).collect::<Vec<_>>())
}

fn stsz(sizes: &[u32]) -> Vec<u8> {
    let mut p = 0u32.to_be_bytes().to_vec();
    p.extend((sizes.len() as u32).to_be_bytes());
    p.extend(sizes.iter().flat_map(|s| s.to_be_bytes()));
    full_box(b"stsz", 0, &p)
}

fn co64(offsets: &[u64]) -> Vec<u8> {
    let mut p = (offsets.len() as u32).to_be_bytes().to_vec();
    p.extend(offsets.iter().flat_map(|o| o.to_be_bytes()));
    full_box(b"co64", 0, &p)
}

// ------------------------------------------------------------------ reading

/// One box: its type and its payload's byte range in the file.
struct BoxRef {
    kind: [u8; 4],
    start: u64,
    end: u64,
}

/// The boxes directly inside `[start, end)`.
fn children(f: &mut (impl Read + Seek), start: u64, end: u64) -> Result<Vec<BoxRef>, Error> {
    let mut out = Vec::new();
    let mut at = start;
    while at + 8 <= end {
        f.seek(SeekFrom::Start(at)).map_err(io)?;
        let mut h = [0u8; 8];
        f.read_exact(&mut h).map_err(io)?;
        let size32 = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as u64;
        let kind = [h[4], h[5], h[6], h[7]];
        let (header, size) = match size32 {
            1 => {
                let mut l = [0u8; 8];
                f.read_exact(&mut l).map_err(io)?;
                (16, u64::from_be_bytes(l))
            }
            0 => (8, end - at),
            n => (8, n),
        };
        // `size` is the file's word: a 64-bit one near 2^64 would wrap a plain
        // sum past the bound and send the walk back to the start, forever.
        let box_end = at.checked_add(size).filter(|&e| size >= header && e <= end);
        let Some(box_end) = box_end else {
            return Err(Error::Malformed(format!("mp4: box {} overruns its parent", String::from_utf8_lossy(&kind))));
        };
        out.push(BoxRef { kind, start: at + header, end: box_end });
        at = box_end;
    }
    Ok(out)
}

fn io(e: std::io::Error) -> Error {
    Error::Io(e.to_string())
}

fn find<'a>(boxes: &'a [BoxRef], kind: &[u8; 4]) -> Option<&'a BoxRef> {
    boxes.iter().find(|b| &b.kind == kind)
}

fn payload(f: &mut (impl Read + Seek), b: &BoxRef) -> Result<Vec<u8>, Error> {
    // Tables of a real file run to megabytes at most; a box claiming more is refused rather than allocated.
    const MAX_TABLE: u64 = 256 << 20;
    if b.end - b.start > MAX_TABLE {
        return Err(Error::Malformed(format!("mp4: a {} box of {} bytes", String::from_utf8_lossy(&b.kind), b.end - b.start)));
    }
    f.seek(SeekFrom::Start(b.start)).map_err(io)?;
    let mut v = vec![0u8; (b.end - b.start) as usize];
    f.read_exact(&mut v).map_err(io)?;
    Ok(v)
}

fn be32(b: &[u8], at: usize) -> Result<u32, Error> {
    b.get(at..at + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| Error::Malformed("mp4: a table is truncated".into()))
}

fn be64(b: &[u8], at: usize) -> Result<u64, Error> {
    b.get(at..at + 8).map(|s| u64::from_be_bytes(s.try_into().expect("8 bytes"))).ok_or_else(|| Error::Malformed("mp4: a table is truncated".into()))
}

/// Where one sample lies and when it is shown: (file offset, size, time in seconds).
type SampleRef = (u64, u32, f64);

/// The AV1 video track of an MP4 file, sample by sample.
pub(crate) struct Reader {
    file: std::io::BufReader<std::fs::File>,
    samples: std::vec::IntoIter<SampleRef>,
}

pub(crate) fn open(path: &Path) -> Result<(StreamInfo, Reader), Error> {
    let named = |e: Error| match e {
        Error::Malformed(m) => Error::Malformed(format!("{}: {m}", path.display())),
        Error::Unsupported(m) => Error::Unsupported(format!("{}: {m}", path.display())),
        other => other,
    };
    let file = std::fs::File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let len = file.metadata().map_err(io)?.len();
    let mut f = std::io::BufReader::new(file);
    let (info, samples) = video_track(&mut f, len).map_err(named)?;
    Ok((info, Reader { file: f, samples: samples.into_iter() }))
}

fn video_track(f: &mut (impl Read + Seek), len: u64) -> Result<(StreamInfo, Vec<SampleRef>), Error> {
    let top = children(f, 0, len)?;
    let moov = find(&top, b"moov").ok_or_else(|| Error::Malformed("no moov box".into()))?;
    let moov_children = children(f, moov.start, moov.end)?;
    if find(&moov_children, b"mvex").is_some() {
        return Err(Error::Unsupported("fragmented MP4".into()));
    }
    for trak in moov_children.iter().filter(|b| &b.kind == b"trak") {
        let trak_children = children(f, trak.start, trak.end)?;
        let Some(mdia) = find(&trak_children, b"mdia") else { continue };
        let mdia_children = children(f, mdia.start, mdia.end)?;
        let Some(hdlr) = find(&mdia_children, b"hdlr") else { continue };
        if payload(f, hdlr)?.get(8..12) != Some(b"vide".as_slice()) {
            continue;
        }
        let mdhd = payload(f, find(&mdia_children, b"mdhd").ok_or_else(|| Error::Malformed("a track without mdhd".into()))?)?;
        let timescale = if mdhd.first() == Some(&1) { be32(&mdhd, 20)? } else { be32(&mdhd, 12)? };
        if timescale == 0 {
            return Err(Error::Malformed("a video timescale of 0".into()));
        }
        let minf = find(&mdia_children, b"minf").ok_or_else(|| Error::Malformed("a track without minf".into()))?;
        let minf_children = children(f, minf.start, minf.end)?;
        let stbl = find(&minf_children, b"stbl").ok_or_else(|| Error::Malformed("a track without stbl".into()))?;
        let stbl_children = children(f, stbl.start, stbl.end)?;
        let table = |f: &mut _, kind: &[u8; 4]| -> Result<Option<Vec<u8>>, Error> { find(&stbl_children, kind).map(|b| payload(f, b)).transpose() };

        let stsd = table(f, b"stsd")?.ok_or_else(|| Error::Malformed("no stsd".into()))?;
        let codec = stsd.get(12..16).ok_or_else(|| Error::Malformed("an empty stsd".into()))?;
        if codec != b"av01" {
            return Err(Error::Unsupported(format!("video codec {}", String::from_utf8_lossy(codec))));
        }
        // The visual sample entry's width and height, after its 8-byte box header and 24 bytes of fixed fields.
        let dims = stsd.get(40..44).ok_or_else(|| Error::Malformed("a truncated video sample entry".into()))?;
        let (w, h) = (u16::from_be_bytes([dims[0], dims[1]]) as u32, u16::from_be_bytes([dims[2], dims[3]]) as u32);

        let stsz = table(f, b"stsz")?.ok_or_else(|| Error::Malformed("no stsz".into()))?;
        let uniform = be32(&stsz, 4)?;
        let count = be32(&stsz, 8)? as usize;
        // Every count below comes from the file. A sample takes at least a
        // byte of it, so no table may describe more samples than the file
        // holds bytes; this bounds each allocation by the input's own size.
        if count as u64 > len || (uniform as u64).saturating_mul(count as u64) > len || (uniform == 0 && count > (stsz.len().saturating_sub(12)) / 4) {
            return Err(Error::Malformed(format!("mp4: {count} samples do not fit a {len}-byte file")));
        }
        let sizes: Vec<u32> = if uniform != 0 { vec![uniform; count] } else { (0..count).map(|i| be32(&stsz, 12 + 4 * i)).collect::<Result<_, _>>()? };

        let chunk_offsets: Vec<u64> = match (table(f, b"stco")?, table(f, b"co64")?) {
            (Some(t), _) => (0..be32(&t, 4)? as usize).map(|i| be32(&t, 8 + 4 * i).map(u64::from)).collect::<Result<_, _>>()?,
            (None, Some(t)) => (0..be32(&t, 4)? as usize).map(|i| be64(&t, 8 + 8 * i)).collect::<Result<_, _>>()?,
            (None, None) => return Err(Error::Malformed("no chunk offsets".into())),
        };
        let stsc = table(f, b"stsc")?.ok_or_else(|| Error::Malformed("no stsc".into()))?;
        let runs: Vec<(u32, u32)> = (0..be32(&stsc, 4)? as usize).map(|i| Ok((be32(&stsc, 8 + 12 * i)?, be32(&stsc, 12 + 12 * i)?))).collect::<Result<_, Error>>()?;
        let stts = table(f, b"stts")?.ok_or_else(|| Error::Malformed("no stts".into()))?;
        let deltas: Vec<(u32, u32)> = (0..be32(&stts, 4)? as usize).map(|i| Ok((be32(&stts, 8 + 8 * i)?, be32(&stts, 12 + 8 * i)?))).collect::<Result<_, Error>>()?;

        // Offsets: walk the chunks, each holding its run's samples back to back.
        let mut offsets = Vec::with_capacity(count);
        for (c, &chunk_offset) in chunk_offsets.iter().enumerate() {
            let chunk = c as u32 + 1;
            let per_chunk = runs.iter().rev().find(|(first, _)| *first <= chunk).map_or(0, |r| r.1);
            let mut at = chunk_offset;
            for _ in 0..per_chunk {
                let Some(&size) = sizes.get(offsets.len()) else { break };
                offsets.push((at, size));
                at = at.saturating_add(size as u64);
            }
        }
        if offsets.len() != count {
            return Err(Error::Malformed(format!("the chunk tables place {} of {count} samples", offsets.len())));
        }
        let mut times = Vec::with_capacity(count);
        let mut t = 0u64;
        'runs: for &(n, d) in &deltas {
            for _ in 0..n {
                // A run may claim 2^32 samples; only `count` exist.
                if times.len() == count {
                    break 'runs;
                }
                times.push(t as f64 / timescale as f64);
                t = t.saturating_add(d as u64);
            }
        }
        times.resize(count, t as f64 / timescale as f64);
        let duration = t as f64 / timescale as f64;
        let fps = (duration > 0.0).then(|| crate::snap_fps(count as f64 / duration));
        let info = StreamInfo { w, h, fps, duration: Some(duration), frames: Some(count as u64), codec: "av1".into() };
        let samples = offsets.into_iter().zip(times).map(|((o, s), t)| (o, s, t)).collect();
        return Ok((info, samples));
    }
    Err(Error::Unsupported("no video track".into()))
}

impl SampleSource for Reader {
    fn next_sample(&mut self) -> Result<Option<Sample>, Error> {
        let Some((offset, size, time)) = self.samples.next() else { return Ok(None) };
        // A coded frame is far below this; a size past it is a corrupt table, not an allocation to attempt.
        if size > 256 << 20 {
            return Err(Error::Malformed(format!("mp4: a sample of {size} bytes")));
        }
        self.file.seek(SeekFrom::Start(offset)).map_err(io)?;
        let mut data = vec![0u8; size as usize];
        self.file.read_exact(&mut data).map_err(io)?;
        Ok(Some(Sample { time, data }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux::{VideoPacket, VideoStream};
    use std::io::Cursor;

    /// A small valid AV1 MP4 as brain writes it (two frames of filler bytes;
    /// the demuxer never decodes them).
    fn valid() -> Vec<u8> {
        let packets = vec![VideoPacket { data: vec![0x12, 0, 0x0a, 0x0b], key: true }, VideoPacket { data: vec![0x12, 0, 0x32], key: false }];
        let video = VideoStream { w: 16, h: 16, fps: (24, 1), config: vec![0x81, 0, 0, 0], packets };
        let mut out = Vec::new();
        write(&mut out, &video, None);
        out
    }

    fn open_bytes(bytes: &[u8]) -> Result<(StreamInfo, Vec<SampleRef>), Error> {
        video_track(&mut Cursor::new(bytes), bytes.len() as u64)
    }

    /// The payload offset of the first `kind` box in `bytes`.
    fn payload_at(bytes: &[u8], kind: &[u8; 4]) -> usize {
        bytes.windows(4).position(|w| w == kind).expect("box present") + 4
    }

    #[test]
    fn the_fixture_reads() {
        let (_, samples) = open_bytes(&valid()).expect("a file brain wrote reads back");
        assert_eq!(samples.len(), 2);
    }

    /// A 64-bit box size that wraps the walk back to the file's start must be
    /// refused, not followed: it used to loop forever, growing a Vec.
    #[test]
    fn a_box_size_that_wraps_is_refused() {
        let mut bytes = vec![0, 0, 0, 16, b'f', b't', b'y', b'p', b'i', b's', b'o', b'm', 0, 0, 0, 0];
        bytes.extend_from_slice(&[0, 0, 0, 1, b'f', b'r', b'e', b'e']);
        bytes.extend_from_slice(&(u64::MAX - 15).to_be_bytes());
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = done.send(open_bytes(&bytes).is_err());
        });
        assert_eq!(finished.recv_timeout(std::time::Duration::from_secs(5)), Ok(true), "the walk must end, refusing the box");
    }

    /// Sample tables whose declared counts the file cannot hold are refused
    /// before anything is allocated for them.
    #[test]
    fn sample_counts_beyond_the_file_are_refused() {
        let mut uniform = valid();
        let at = payload_at(&uniform, b"stsz") + 4; // past version and flags
        uniform[at..at + 4].copy_from_slice(&1u32.to_be_bytes());
        uniform[at + 4..at + 8].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(open_bytes(&uniform), Err(Error::Malformed(_))), "a uniform-size table of 2^32 samples");

        let mut listed = valid();
        let at = payload_at(&listed, b"stsz") + 8;
        listed[at..at + 4].copy_from_slice(&1_000_000u32.to_be_bytes());
        assert!(matches!(open_bytes(&listed), Err(Error::Malformed(_))), "a size list longer than its box");
    }

    /// A time-to-sample run claiming 2^32 samples stops at the samples there are.
    #[test]
    fn a_time_run_longer_than_the_track_is_cut_at_its_samples() {
        let mut bytes = valid();
        let at = payload_at(&bytes, b"stts") + 8; // past version, flags and entry count
        bytes[at..at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        let (_, samples) = open_bytes(&bytes).expect("the track still reads");
        assert_eq!(samples.len(), 2);
    }
}
