// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Video in brain's own process: AV1 written and read without ffmpeg.
//!
//! Swedish Embedded AB implements solutions for shipping generative video
//! without external tools or royalty-bearing codecs for its clients. If your
//! team needs expertise in video coding and media containers then you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! **Writing.** [`VideoWriter`] takes RGB frames at a fixed size and rate,
//! codes them as AV1 ([`av1::Encoder`], rav1e) and, given a sound track, the
//! sound as Opus, and writes WebM (`.webm`), Matroska (`.mkv`) or MP4
//! (`.mp4`, `.mov`, `.m4v`) by the path's extension. AV1 and Opus are
//! royalty-free, play in current browsers and players, and need nothing
//! installed.
//!
//! **Reading.** [`decode`] reads the AV1 video of a WebM/Matroska or MP4
//! file through rav1d. A file in another codec or container is
//! [`Error::Unsupported`], which a caller can hand to an external decoder.
//!
//! **Colour.** Frames are written as BT.709 limited-range 4:2:0 and labelled
//! so; frames are read in the colour their stream declares ([`color`]).
//!
//! **Memory.** The writer holds the coded streams (not the frames) until
//! [`VideoWriter::finish`] writes the file: a few megabytes per minute at the
//! default quality, so the file is complete and interleaved in one pass.

pub mod av1;
pub mod color;
mod mp4;
mod mux;
mod webm;

use std::path::Path;

pub use av1::EncodeSettings;
use mux::{OpusStream, VideoPacket, VideoStream};
pub use webm::DocType;

/// Why a video could not be written or read.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A container or codec this crate does not read; another decoder may.
    Unsupported(String),
    /// The file is not what it claims to be.
    Malformed(String),
    /// A request that cannot be coded (no pixels, a frame of the wrong size, ...).
    Invalid(String),
    Io(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(m) => write!(f, "unsupported video: {m}"),
            Error::Malformed(m) => write!(f, "malformed video: {m}"),
            Error::Invalid(m) => write!(f, "{m}"),
            Error::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

/// A sound track: one plane per channel, samples in `[-1, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    pub channels: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

/// The container a path names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Container {
    Matroska(DocType),
    Mp4,
}

impl Container {
    /// By extension (case-insensitive); `None` for one this crate does not write.
    pub fn for_path(path: &Path) -> Option<Container> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "webm" => Some(Container::Matroska(DocType::WebM)),
            "mkv" => Some(Container::Matroska(DocType::Matroska)),
            "mp4" | "mov" | "m4v" => Some(Container::Mp4),
            _ => None,
        }
    }
}

/// `fps` as an exact rational: integer rates exactly, the NTSC family
/// (23.976, 29.97, 59.94) as `n * 1000 / 1001`, anything else to a
/// millisecond-of-a-frame.
pub fn fps_rational(fps: f64) -> Result<(u64, u64), Error> {
    if !(fps.is_finite() && fps > 0.0 && fps <= 1000.0) {
        return Err(Error::Invalid(format!("a frame rate of {fps}")));
    }
    if (fps - fps.round()).abs() < 1e-9 {
        return Ok((fps.round() as u64, 1));
    }
    let ntsc = fps * 1001.0 / 1000.0;
    if (ntsc - ntsc.round()).abs() < 1e-6 {
        return Ok((ntsc.round() as u64 * 1000, 1001));
    }
    let num = (fps * 1000.0).round() as u64;
    let g = gcd(num, 1000);
    Ok((num / g, 1000 / g))
}

/// A frame rate read from a container's integer clock, snapped to the exact
/// rate it rounds: an integer rate or one of the NTSC family when within a
/// millionth, otherwise as read. (Matroska stores a frame's duration in whole
/// nanoseconds, so 24 fps reads back as 24.0000004.)
pub(crate) fn snap_fps(fps: f64) -> f64 {
    let close = |x: f64| (fps - x).abs() <= 1e-6 * fps;
    if close(fps.round()) {
        return fps.round();
    }
    let ntsc = (fps * 1.001).round() / 1.001;
    if close(ntsc) {
        return ntsc;
    }
    fps
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Writes one video file: frames in, AV1 (and Opus) coded as they come, the
/// file written by [`Self::finish`].
pub struct VideoWriter {
    path: std::path::PathBuf,
    container: Container,
    encoder: av1::Encoder,
    stream: VideoStream,
    frames: u64,
}

impl VideoWriter {
    /// A writer of `w` x `h` RGB frames at `fps`, to `path` (whose extension
    /// picks the container, see [`Container::for_path`]).
    pub fn create(path: &Path, w: u32, h: u32, fps: f64, settings: EncodeSettings) -> Result<VideoWriter, Error> {
        let container = Container::for_path(path)
            .ok_or_else(|| Error::Invalid(format!("{}: write video as .webm, .mkv, .mp4, .mov or .m4v", path.display())))?;
        let fps = fps_rational(fps)?;
        let encoder = av1::Encoder::new(w as usize, h as usize, fps, settings)?;
        let stream = VideoStream { w: w as usize, h: h as usize, fps, config: encoder.codec_config(), packets: Vec::new() };
        Ok(VideoWriter { path: path.to_path_buf(), container, encoder, stream, frames: 0 })
    }

    /// Code one frame: `w * h * 3` bytes, R,G,B interleaved, row-major.
    pub fn push(&mut self, rgb: &[u8]) -> Result<(), Error> {
        let need = self.stream.w * self.stream.h * 3;
        if rgb.len() != need {
            return Err(Error::Invalid(format!("a {}x{} frame is {need} bytes, got {}", self.stream.w, self.stream.h, rgb.len())));
        }
        self.frames += 1;
        take(&mut self.stream, self.encoder.push(rgb)?);
        Ok(())
    }

    /// Frames pushed so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Code the last frames and the sound (if any) and write the file.
    pub fn finish(self, audio: Option<&Audio>) -> Result<(), Error> {
        if self.frames == 0 {
            return Err(Error::Invalid(format!("{}: a video needs at least one frame", self.path.display())));
        }
        let VideoWriter { path, container, encoder, mut stream, frames } = self;
        take(&mut stream, encoder.finish()?);
        if stream.packets.len() as u64 != frames {
            return Err(Error::Invalid(format!("the AV1 encoder coded {} of {frames} frames", stream.packets.len())));
        }
        let audio = audio.filter(|a| a.channels.first().is_some_and(|c| !c.is_empty())).map(OpusStream::encode).transpose()?;
        let mut bytes = Vec::new();
        match container {
            Container::Matroska(doc) => webm::write(&mut bytes, doc, &stream, audio.as_ref()),
            Container::Mp4 => mp4::write(&mut bytes, &stream, audio.as_ref()),
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
        }
        std::fs::write(&path, bytes).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
    }
}

/// Append coded units to `stream`. AV1 units come out in presentation order, one per input frame.
fn take(stream: &mut VideoStream, coded: Vec<av1::Packet>) {
    for p in coded {
        debug_assert_eq!(p.frame, stream.packets.len() as u64);
        stream.packets.push(VideoPacket { data: p.data, key: p.key });
    }
}

/// What a container says about its video track.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamInfo {
    pub w: u32,
    pub h: u32,
    /// Frames per second, when the container records it.
    pub fps: Option<f64>,
    /// Seconds, when the container records it.
    pub duration: Option<f64>,
    /// Coded frames, when the container records it.
    pub frames: Option<u64>,
    pub codec: String,
}

/// One coded unit and its presentation time in seconds.
pub(crate) struct Sample {
    pub time: f64,
    pub data: Vec<u8>,
}

pub(crate) trait SampleSource {
    fn next_sample(&mut self) -> Result<Option<Sample>, Error>;
}

/// Open the AV1 video of the file at `path`, by its first bytes.
fn open(path: &Path) -> Result<(StreamInfo, Box<dyn SampleSource>), Error> {
    let mut magic = [0u8; 12];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let n = f.read(&mut magic).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        if n < magic.len() {
            return Err(Error::Malformed(format!("{}: {n} bytes is no video", path.display())));
        }
    }
    if magic[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        let (info, r) = webm::open(path)?;
        return Ok((info, Box::new(r)));
    }
    if &magic[4..8] == b"ftyp" {
        let (info, r) = mp4::open(path)?;
        return Ok((info, Box::new(r)));
    }
    Err(Error::Unsupported(format!("{}: not a WebM, Matroska or MP4 file", path.display())))
}

/// The video track's description, counting the coded frames (without decoding
/// them) when the container does not record the count.
pub fn probe(path: &Path) -> Result<StreamInfo, Error> {
    let (mut info, mut source) = open(path)?;
    if info.frames.is_none() || info.duration.is_none() {
        let (mut n, mut first, mut last, mut step) = (0u64, None, 0.0f64, None);
        while let Some(s) = source.next_sample()? {
            if n == 1 {
                step = first.map(|f: f64| s.time - f);
            }
            first.get_or_insert(s.time);
            last = s.time;
            n += 1;
        }
        info.frames.get_or_insert(n);
        let fps = info.fps.or(step.filter(|&d| d > 0.0).map(|d| 1.0 / d));
        info.fps = fps;
        info.duration.get_or_insert(last - first.unwrap_or(0.0) + fps.map_or(0.0, |f| 1.0 / f));
    }
    Ok(info)
}

/// A decoded frame: its index in the stream, its time in seconds, and its RGB.
pub struct Frame {
    pub index: u64,
    pub time: f64,
    pub w: u32,
    pub h: u32,
    pub rgb: Vec<u8>,
}

/// The most pixels one decoded frame may hold (an 8K frame is 33.2 million).
/// A stream declaring more is refused before the frame is allocated: the
/// bytes of a video may come from a client, and a few of them can declare a
/// frame of gigabytes.
pub const MAX_FRAME_PIXELS: u64 = 36_000_000;

/// Decode the video of `path` frame by frame, in order. `on_frame` returns
/// whether to go on; decoding stops at the first `false`. Returns the
/// stream's description.
pub fn decode(path: &Path, on_frame: &mut dyn FnMut(Frame) -> bool) -> Result<StreamInfo, Error> {
    let (info, mut source) = open(path)?;
    let mut decoder = av1::Decoder::new()?;
    // Times of units sent and not yet output: AV1 shows each unit's frame in order.
    let mut times = std::collections::VecDeque::new();
    let mut index = 0u64;
    let mut emit = |pics: Vec<av1::Picture>, times: &mut std::collections::VecDeque<f64>| -> bool {
        for p in pics {
            let time = times.pop_front().unwrap_or(0.0);
            let go = on_frame(Frame { index, time, w: p.w as u32, h: p.h as u32, rgb: p.rgb });
            index += 1;
            if !go {
                return false;
            }
        }
        true
    };
    while let Some(s) = source.next_sample()? {
        times.push_back(s.time);
        let pics = decoder.push(&s.data)?;
        if !emit(pics, &mut times) {
            return Ok(info);
        }
    }
    let pics = decoder.finish()?;
    emit(pics, &mut times);
    Ok(info)
}
