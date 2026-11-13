// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Video files <-> RGB frames, in process.
//!
//! **Writing** ([`encode_frames`]) is always brain's own: AV1 video, and Opus
//! sound when there is a track, in WebM (`.webm`), Matroska (`.mkv`) or MP4
//! (`.mp4`, `.mov`, `.m4v`), through `videocodec`; or an animated `.gif`. No
//! external tool is involved, so a generation is never left as loose frames.
//!
//! **Reading** ([`decode_frames`], [`decode_frames_at`], [`probe_video`] and
//! the `probe_*` helpers) decodes AV1 in those containers itself. A file in
//! another codec or container (H.264 from a phone, say) is handed to the
//! `ffmpeg`/`ffprobe` CLIs when they are installed, and refused with that
//! advice when they are not: brain needs no ffmpeg for anything it writes.
//!
//! Frame selection (`fps`, `spread`, `max_frames`) is done on the stream's
//! own timestamps: an output instant takes the source frame nearest to it.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::pixels::Rgb8;

/// True if the `ffmpeg` CLI is on `PATH` and runs - the decoder of input
/// videos brain does not read itself.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// Frame-selection knobs for [`decode_frames`].
pub struct VideoDecodeOpts {
    /// Sample at this many frames per second. `None` keeps every frame.
    pub fps: Option<f64>,
    /// Stop after this many frames. `0` means unbounded.
    ///
    /// This TRUNCATES: it keeps the first N frames and stops. For a capture
    /// that has to be covered rather than sampled from its opening seconds -
    /// an orbit, a turntable - use `spread` instead.
    pub max_frames: u32,
    /// Ask for about this many frames spaced evenly across the WHOLE clip,
    /// `0` to disable. Overrides `fps`, which it derives from the clip's
    /// duration.
    pub spread: u32,
}

impl Default for VideoDecodeOpts {
    /// A short clip at a low frame rate — sane for a multimodal prompt (a
    /// few seconds of video, not a full-length feature), matching the scale
    /// `qwen3omnimoe::mm::encode_video_frames` is validated at.
    fn default() -> VideoDecodeOpts {
        VideoDecodeOpts { fps: Some(1.0), max_frames: 32, spread: 0 }
    }
}

/// Decode a video file to RGB frames: one `(hwc_f32_unit, w, h)` tuple per
/// frame, in order - the exact shape `qwen3omnimoe::mm::encode_video_frames` takes
/// (`Rgb8::to_hwc_unit`'s output, `capability::blob::decode_video`'s wire
/// convention).
pub fn decode_frames(path: &Path, opts: &VideoDecodeOpts) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
    Ok(decode_frames_rgb8(path, opts)?.into_iter().map(|img| (img.to_hwc_unit(), img.w, img.h)).collect())
}

/// The same decode, kept as [`Rgb8`]: what a caller handing the frames back
/// to a codec wants, without a `u8 -> f32 -> u8` round trip.
pub fn decode_frames_rgb8(path: &Path, opts: &VideoDecodeOpts) -> Result<Vec<Rgb8>, String> {
    if !path.exists() {
        return Err(format!("imaging::video: {} does not exist", path.display()));
    }
    // `spread` needs the clip's length to turn "about N frames" into a rate.
    let rate = |duration: Option<f64>| -> Result<Option<f64>, String> {
        match (opts.spread, duration) {
            (n, Some(d)) if n > 0 && d > 0.0 => Ok(Some((n as f64 - 0.5).max(1.0) / d)),
            (n, _) if n > 0 => Err(format!("imaging::video: spread={n} needs the duration of {}, which it does not record", path.display())),
            _ => Ok(opts.fps),
        }
    };
    let native = videocodec::probe(path).and_then(|info| {
        let fps = rate(info.duration).map_err(videocodec::Error::Invalid)?;
        let mut picker = Picker::new(fps, opts.max_frames);
        videocodec::decode(path, &mut |f| picker.push(f))?;
        Ok(picker.finish())
    });
    let frames = match native {
        Ok(frames) => frames,
        Err(videocodec::Error::Invalid(m)) | Err(videocodec::Error::Io(m)) => return Err(format!("imaging::video: {m}")),
        Err(e) => {
            let fallback = external(path, &e)?;
            let fps = rate(Some(fallback.duration))?;
            ffmpeg::decode(path, fallback.w, fallback.h, fps, opts.max_frames)?
        }
    };
    if frames.is_empty() {
        return Err(format!("imaging::video: {} holds no frames", path.display()));
    }
    Ok(frames)
}

/// How close to the midpoint of two frames an instant counts as a tie.
const TIE_SECONDS: f64 = 1e-3;

/// The output instants of a frame-rate resample, each taking the source frame
/// nearest to it; or every frame when no rate is asked for.
struct Picker {
    fps: Option<f64>,
    max: usize,
    out: Vec<Rgb8>,
    /// The previous source frame and its time.
    prev: Option<(f64, Rgb8)>,
    /// The interval between the last two source frames: how long the last one is shown.
    interval: Option<f64>,
    /// The time of the first source frame (output instants count from it).
    start: Option<f64>,
}

impl Picker {
    fn new(fps: Option<f64>, max_frames: u32) -> Picker {
        Picker { fps, max: if max_frames == 0 { usize::MAX } else { max_frames as usize }, out: Vec::new(), prev: None, interval: None, start: None }
    }

    fn full(&self) -> bool {
        self.out.len() >= self.max
    }

    /// The next output instant, relative to the stream's own clock.
    fn next_instant(&self, fps: f64) -> f64 {
        self.start.unwrap_or(0.0) + self.out.len() as f64 / fps
    }

    /// Take one decoded frame; whether more are wanted.
    fn push(&mut self, f: videocodec::Frame) -> bool {
        let frame = Rgb8 { w: f.w, h: f.h, px: f.rgb };
        let Some(fps) = self.fps else {
            self.out.push(frame);
            return !self.full();
        };
        self.start.get_or_insert(f.time);
        if let Some((prev_time, prev)) = &self.prev {
            self.interval = Some(f.time - prev_time);
            // Every instant nearer the previous frame than this one belongs to it, and so does a tie (as
            // ffmpeg's `fps` filter fills a gap with the frame before it). A tie is judged to the precision of
            // the coarsest common container clock, Matroska's millisecond, which rounds frame times.
            let midpoint = (prev_time + f.time) / 2.0;
            while !self.full() && self.next_instant(fps) <= midpoint + TIE_SECONDS {
                self.out.push(prev.clone());
            }
        }
        self.prev = Some((f.time, frame));
        !self.full()
    }

    /// The instants the last frame covers: those before the end of its display,
    /// one source interval after it starts (a lone frame covers one output instant).
    fn finish(mut self) -> Vec<Rgb8> {
        if let (Some(fps), Some((time, last))) = (self.fps, self.prev.take()) {
            let shown_until = time + self.interval.unwrap_or(1.0 / fps) - 1e-9;
            while !self.full() && self.next_instant(fps) < shown_until {
                self.out.push(last.clone());
            }
        }
        self.out
    }
}

/// A video stream's metadata, as the container reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoMeta {
    pub w: u32,
    pub h: u32,
    /// Average frame rate.
    pub fps: f64,
    /// Frames in the stream (counted when the container does not record it).
    pub frames: u64,
    /// The stream's duration in seconds.
    pub duration: f64,
}

/// The first video stream's [`VideoMeta`].
pub fn probe_video(path: &Path) -> Result<VideoMeta, String> {
    match videocodec::probe(path) {
        Ok(info) => {
            let fps = info.fps.filter(|f| *f > 0.0).ok_or_else(|| format!("imaging::video: {} reports no frame rate", path.display()))?;
            let frames = info.frames.unwrap_or(0);
            if frames == 0 {
                return Err(format!("imaging::video: {} has no frames", path.display()));
            }
            Ok(VideoMeta { w: info.w, h: info.h, fps, frames, duration: info.duration.filter(|d| *d > 0.0).unwrap_or(frames as f64 / fps) })
        }
        Err(videocodec::Error::Io(m)) => Err(format!("imaging::video: {m}")),
        Err(e) => external(path, &e),
    }
}

/// [`probe_video`] for a file brain does not read itself: through `ffprobe`,
/// or the reason it cannot be read at all.
fn external(path: &Path, native: &videocodec::Error) -> Result<VideoMeta, String> {
    if !ffmpeg_available() {
        return Err(format!(
            "imaging::video: {native}. brain reads AV1 video in WebM, Matroska and MP4 itself; install ffmpeg to read other formats"
        ));
    }
    ffmpeg::probe_video(path)
}

/// Decode exactly the frames at `indices` (0-based, in decode order), in the
/// order asked - an index may repeat. Only the asked frames are kept; decoding
/// starts at the key frame at or before the first one asked
/// (`videocodec::decode_from`) and stops after the last. An index past the
/// clip's end is an error naming it.
pub fn decode_frames_at(path: &Path, indices: &[u64]) -> Result<Vec<Rgb8>, String> {
    let (Some(&first), Some(&last)) = (indices.iter().min(), indices.iter().max()) else { return Ok(Vec::new()) };
    let mut kept: std::collections::BTreeMap<u64, Option<Rgb8>> = indices.iter().map(|&i| (i, None)).collect();
    let native = videocodec::decode_from(path, first, &mut |f| {
        if let Some(slot) = kept.get_mut(&f.index) {
            *slot = Some(Rgb8 { w: f.w, h: f.h, px: f.rgb.clone() });
        }
        f.index < last
    });
    match native {
        Ok(_) => {}
        Err(videocodec::Error::Io(m)) => return Err(format!("imaging::video: {m}")),
        Err(e) => {
            let meta = external(path, &e)?;
            ffmpeg::stream_frames(path, meta.w, meta.h, None, 0, &mut |i, px| {
                if let Some(slot) = kept.get_mut(&i) {
                    *slot = Some(Rgb8 { w: meta.w, h: meta.h, px: px.to_vec() });
                }
                i < last
            })?;
        }
    }
    indices.iter().map(|i| kept[i].clone().ok_or_else(|| format!("imaging::video: {} has no frame {i}", path.display()))).collect()
}

/// The video's pixel dimensions.
pub fn probe_size(path: &Path) -> Option<(u32, u32)> {
    probe_video(path).ok().map(|m| (m.w, m.h))
}

/// The video's duration in seconds.
pub fn probe_duration(path: &Path) -> Option<f64> {
    probe_video(path).ok().map(|m| m.duration)
}

/// The video's own average frame rate; `None` when it cannot be read - an
/// "ask the caller instead" case, not an error worth failing a run over.
pub fn probe_fps(path: &Path) -> Option<f64> {
    probe_video(path).ok().map(|m| m.fps)
}

// ---------------------------------------------------------------- encoding

/// A sound track to carry with the picture: one plane per channel, samples in
/// `[-1, 1]`.
pub use videocodec::Audio as AudioTrack;

/// Container and codec settings for [`encode_frames`].
#[derive(Default)]
pub struct VideoEncodeOpts {
    /// The AV1 encoder's size/quality/time trade-off.
    pub settings: videocodec::EncodeSettings,
    /// A sound track to carry in the same file, coded as Opus. A `.gif`
    /// cannot hold one; it is written silent with a line on stderr rather
    /// than failing a generation over a container choice.
    pub audio: Option<AudioTrack>,
}

/// Write `frames` at `fps` to `path`: AV1 (and Opus sound) in the container
/// its extension names, or an animated GIF. Every frame must have the same
/// non-zero dimensions.
pub fn encode_frames(frames: &[Rgb8], path: &Path, fps: f64, opts: &VideoEncodeOpts) -> Result<PathBuf, String> {
    let (w, h) = check_frames(frames)?;
    if !(fps.is_finite() && fps > 0.0) {
        return Err(format!("imaging::video: fps must be finite and positive (got {fps})"));
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "gif" {
        if opts.audio.is_some() {
            eprintln!("imaging::video: a .gif carries no sound; the clip is written silent (use .mp4, .webm or .mkv to keep it)");
        }
        write_gif(frames, path, fps)?;
        return Ok(path.to_path_buf());
    }
    let named = |e: videocodec::Error| format!("imaging::video: {}: {e}", path.display());
    let mut writer = videocodec::VideoWriter::create(path, w, h, fps, opts.settings).map_err(named)?;
    for f in frames {
        writer.push(&f.px).map_err(named)?;
    }
    writer.finish(opts.audio.as_ref()).map_err(named)?;
    Ok(path.to_path_buf())
}

fn write_gif(frames: &[Rgb8], path: &Path, fps: f64) -> Result<(), String> {
    use image::codecs::gif::{GifEncoder, Repeat};
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("imaging::video: creating {}: {e}", dir.display()))?;
    }
    let file = std::fs::File::create(path).map_err(|e| format!("imaging::video: {}: {e}", path.display()))?;
    let mut enc = GifEncoder::new_with_speed(std::io::BufWriter::new(file), 10);
    enc.set_repeat(Repeat::Infinite).map_err(|e| format!("imaging::video: {}: {e}", path.display()))?;
    let delay = image::Delay::from_saturating_duration(std::time::Duration::from_secs_f64(1.0 / fps));
    for f in frames {
        let rgba: Vec<u8> = f.px.as_chunks::<3>().0.iter().flat_map(|&[r, g, b]| [r, g, b, 255]).collect();
        let buf = image::RgbaImage::from_raw(f.w, f.h, rgba).expect("the frame was checked");
        enc.encode_frame(image::Frame::from_parts(buf, 0, 0, delay)).map_err(|e| format!("imaging::video: {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Same dimensions on every frame, at least one frame, nothing degenerate.
fn check_frames(frames: &[Rgb8]) -> Result<(u32, u32), String> {
    let first = frames.first().ok_or("imaging::video: no frames to encode")?;
    let (w, h) = (first.w, first.h);
    if w == 0 || h == 0 {
        return Err(format!("imaging::video: frame 0 is {w}x{h}"));
    }
    for (i, f) in frames.iter().enumerate() {
        if (f.w, f.h) != (w, h) {
            return Err(format!("imaging::video: frame {i} is {}x{}, frame 0 is {w}x{h}", f.w, f.h));
        }
    }
    Ok((w, h))
}

// ---------------------------------------------------------------- clips as blobs

/// Write a generated clip to `path`: `video` in the video blob wire format
/// (`capability::blob::decode_video`; frame rate in `meta.fps`) and, when the
/// action made one, `sound` as its track. This is the one way an outcome
/// becomes a playable file, for every caller that saves or serves one.
pub fn encode_clip(video: &capability::Blob, sound: Option<&capability::Blob>, path: &Path, settings: videocodec::EncodeSettings) -> Result<PathBuf, String> {
    let frames = capability::blob::decode_video(&capability::Invocation::new().blob("video", video.clone()), "video")?;
    // Only the producing action knows the rate; the fallback is the one every
    // current producer would have written anyway.
    let fps = video.meta.get("fps").and_then(|v| v.as_f64()).unwrap_or(16.0);
    let rgb: Vec<Rgb8> = frames
        .iter()
        .map(|(hwc, w, h)| crate::pixels::hwc_to_rgb8(hwc, *w, *h, 3, crate::ChannelPolicy::ReplicateFirst))
        .collect::<Result<_, _>>()?;
    drop(frames);
    let audio = sound.map(sound_track).transpose()?;
    encode_frames(&rgb, path, fps, &VideoEncodeOpts { settings, audio })
}

/// A sound blob as a track: a WAV container, or headerless f32-LE PCM with
/// `meta.sample_rate` (the two conventions audio-producing actions use).
pub fn sound_track(b: &capability::Blob) -> Result<AudioTrack, String> {
    let (sample_rate, channels) = if audio::asr_caps::is_wav(&b.bytes) {
        audio::wav::parse_planar(&b.bytes).map_err(|e| format!("the sound track: {e}"))?
    } else {
        let rate = b.meta.get("sample_rate").and_then(|v| v.as_u64()).ok_or("the sound track has no sample_rate and is not a WAV")?;
        let rate = audio::wav::checked_rate(rate).map_err(|e| format!("the sound track: {e}"))?;
        (rate, vec![b.bytes.as_chunks::<4>().0.iter().map(|q| f32::from_le_bytes(*q)).collect()])
    };
    Ok(AudioTrack { channels, sample_rate })
}

/// Read a whole clip from `path` into the video blob wire format, every frame
/// at the file's own rate (recorded in `meta.fps`), refusing one longer than
/// `max_frames`. The input of an edit (extend, retake, upscale).
pub fn clip_from_file(path: &Path, max_frames: u32) -> Result<capability::Blob, String> {
    let fps = probe_fps(path).filter(|f| f.is_finite() && *f > 0.0).ok_or("imaging::video: the clip states no frame rate")?;
    let frames = decode_frames(path, &VideoDecodeOpts { fps: None, max_frames: max_frames.saturating_add(1), spread: 0 })?;
    if frames.len() > max_frames as usize {
        return Err(format!("imaging::video: the clip is longer than {max_frames} frames"));
    }
    let mut blob = capability::blob::video_blob(&frames)?;
    if let Some(meta) = blob.meta.as_object_mut() {
        meta.insert("fps".into(), serde_json::json!(fps));
    }
    Ok(blob)
}

/// [`clip_from_file`] for bytes a client sent: decoded by brain's own
/// demuxers and AV1 decoder only, never handed to `ffmpeg` (whose demuxers,
/// protocols and resource use a request must not choose), and refused as soon
/// as the frames decoded so far hold more than `max_pixels` together, before
/// the next is kept. Errors name no path of this machine.
pub fn clip_from_untrusted_file(path: &Path, max_frames: u32, max_pixels: u64) -> Result<capability::Blob, String> {
    let info = videocodec::probe(path).map_err(untrusted)?;
    let fps = info.fps.filter(|f| f.is_finite() && *f > 0.0).ok_or("the clip states no frame rate")?;
    let mut frames = Vec::new();
    let mut pixels = 0u64;
    let mut over = None;
    videocodec::decode(path, &mut |f| {
        pixels = pixels.saturating_add(f.w as u64 * f.h as u64);
        if frames.len() >= max_frames as usize {
            over = Some(format!("the clip is longer than {max_frames} frames"));
        } else if pixels > max_pixels {
            over = Some(format!("the clip holds more than {max_pixels} pixels"));
        } else {
            frames.push(Rgb8 { w: f.w, h: f.h, px: f.rgb });
        }
        over.is_none()
    })
    .map_err(untrusted)?;
    if let Some(over) = over {
        return Err(over);
    }
    if frames.is_empty() {
        return Err("the clip holds no frames".into());
    }
    let frames: Vec<(Vec<f32>, u32, u32)> = frames.into_iter().map(|img| (img.to_hwc_unit(), img.w, img.h)).collect();
    let mut blob = capability::blob::video_blob(&frames)?;
    if let Some(meta) = blob.meta.as_object_mut() {
        meta.insert("fps".into(), serde_json::json!(fps));
    }
    Ok(blob)
}

/// A decoder error, said without the path it was read from.
fn untrusted(e: videocodec::Error) -> String {
    match e {
        videocodec::Error::Unsupported(_) => "the clip is not AV1 video in WebM, Matroska or MP4".into(),
        videocodec::Error::Malformed(_) | videocodec::Error::Invalid(_) => "the clip is malformed".into(),
        videocodec::Error::Io(_) => "the clip could not be read".into(),
    }
}

/// The external decoder for input videos brain does not read itself.
mod ffmpeg {
    use std::path::Path;
    use std::process::Command;

    use super::VideoMeta;
    use crate::pixels::Rgb8;

    /// Every frame (or `fps` of them, nearest-sampled by ffmpeg's `fps` filter), at most `max_frames` (0: all).
    pub(super) fn decode(path: &Path, w: u32, h: u32, fps: Option<f64>, max_frames: u32) -> Result<Vec<Rgb8>, String> {
        let mut out = Vec::new();
        stream_frames(path, w, h, fps.map(|f| format!("fps={f}")), max_frames, &mut |_, px| {
            out.push(Rgb8 { w, h, px: px.to_vec() });
            true
        })?;
        Ok(out)
    }

    pub(super) fn probe_video(path: &Path) -> Result<VideoMeta, String> {
        let probe = |entries: &str, count: bool| -> Result<serde_json::Value, String> {
            let mut cmd = Command::new("ffprobe");
            cmd.args(["-v", "error", "-select_streams", "v:0"]);
            if count {
                cmd.arg("-count_frames");
            }
            let out = cmd.args(["-show_entries", entries, "-of", "json"]).arg(path).output().map_err(|e| format!("imaging::video: running ffprobe: {e}"))?;
            if !out.status.success() {
                return Err(format!("imaging::video: ffprobe could not read {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim()));
            }
            serde_json::from_slice(&out.stdout).map_err(|e| format!("imaging::video: ffprobe output for {}: {e}", path.display()))
        };
        let v = probe("stream=width,height,avg_frame_rate,nb_frames,duration:format=duration", false)?;
        let stream = &v["streams"][0];
        let num = |x: &serde_json::Value| x.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| x.as_f64());
        let w = stream["width"].as_u64().ok_or_else(|| format!("imaging::video: {} has no video stream", path.display()))? as u32;
        let h = stream["height"].as_u64().unwrap_or(0) as u32;
        let fps = stream["avg_frame_rate"]
            .as_str()
            .and_then(|r| r.split_once('/'))
            .and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)))
            .filter(|(a, b)| *a > 0.0 && *b > 0.0)
            .map(|(a, b)| a / b)
            .ok_or_else(|| format!("imaging::video: {} reports no frame rate", path.display()))?;
        let frames = match num(&stream["nb_frames"]).filter(|n| *n > 0.0) {
            Some(n) => n as u64,
            None => num(&probe("stream=nb_read_frames", true)?["streams"][0]["nb_read_frames"]).filter(|n| *n > 0.0).ok_or_else(|| format!("imaging::video: {} has no frames", path.display()))? as u64,
        };
        let duration = num(&stream["duration"]).or_else(|| num(&v["format"]["duration"])).filter(|d| *d > 0.0).unwrap_or(frames as f64 / fps);
        Ok(VideoMeta { w, h, fps, frames, duration })
    }

    /// Run ffmpeg over `path` and hand each decoded `w x h` RGB frame to
    /// `on_frame(index, bytes)` until the stream ends, `max_frames` (if non-zero)
    /// are produced, or `on_frame` returns `false`. `vf` is an optional ffmpeg
    /// filter graph applied before conversion.
    pub(super) fn stream_frames(path: &Path, w: u32, h: u32, vf: Option<String>, max_frames: u32, on_frame: &mut dyn FnMut(u64, &[u8]) -> bool) -> Result<(), String> {
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-v").arg("error").arg("-i").arg(path);
        match vf {
            Some(vf) => {
                cmd.arg("-vf").arg(vf);
            }
            // Unfiltered: every decoded frame exactly once, so a frame's position
            // in the pipe is its decode index.
            None => {
                cmd.args(["-fps_mode", "passthrough"]);
            }
        }
        if max_frames > 0 {
            cmd.arg("-frames:v").arg(max_frames.to_string());
        }
        // Raw RGB straight down a pipe: no container, no per-frame encode, and
        // nothing written to disk. The frame size is fixed and known, so the
        // stream is self-delimiting.
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]);
        cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
    
        let mut child = cmd.spawn().map_err(|e| format!("imaging::video: spawning ffmpeg: {e}"))?;
        let mut stdout = child.stdout.take().expect("piped");
        let frame_bytes = (w as usize) * (h as usize) * 3;
        let mut buf = vec![0u8; frame_bytes];
        let mut index = 0u64;
        let mut stopped = false;
        loop {
            match read_exact_or_eof(&mut stdout, &mut buf)? {
                0 => break,
                n if n == frame_bytes => {
                    let more = on_frame(index, &buf);
                    index += 1;
                    if !more {
                        stopped = true;
                        break;
                    }
                }
                n => return Err(format!("imaging::video: {} ended mid-frame ({n} of {frame_bytes} bytes)", path.display())),
            }
        }
        if stopped {
            // The rest of the stream is not wanted: stop ffmpeg rather than read it.
            let _ = child.kill();
            let _ = child.wait();
            return Ok(());
        }
        let status = child.wait().map_err(|e| format!("imaging::video: waiting for ffmpeg: {e}"))?;
        if !status.success() {
            let mut err = String::new();
            if let Some(mut e) = child.stderr.take() {
                use std::io::Read;
                let _ = e.read_to_string(&mut err);
            }
            return Err(format!("imaging::video: ffmpeg exited {status}: {err}"));
        }
        Ok(())
    }
    
    /// Fill `buf`, returning how many bytes were read: `buf.len()` normally, `0`
    /// at a clean end of stream, anything else for a truncated final frame.
    fn read_exact_or_eof(r: &mut impl std::io::Read, buf: &mut [u8]) -> Result<usize, String> {
        let mut got = 0;
        while got < buf.len() {
            match r.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(format!("imaging::video: reading ffmpeg output: {e}")),
            }
        }
        Ok(got)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ramp that brightens with time under a moving bar: every frame differs.
    fn moving(n: usize, w: u32, h: u32) -> Vec<Rgb8> {
        (0..n)
            .map(|i| {
                let px = (0..h)
                    .flat_map(|y| (0..w).flat_map(move |x| if (x as usize + 3 * i).is_multiple_of(w as usize) { [255, 255, 255] } else { [(i * 8) as u8 + 20, (y * 255 / h) as u8, (x * 255 / w) as u8] }))
                    .collect();
                Rgb8 { w, h, px }
            })
            .collect()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-imaging-video-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// What brain writes it reads back without any external tool: every
    /// container, every frame, at the size written, in order.
    #[test]
    fn written_clips_read_back_in_process() {
        let frames = moving(12, 48, 32);
        for name in ["c.mp4", "c.webm", "c.mkv"] {
            let path = scratch(name);
            let track = AudioTrack { channels: vec![vec![0.1; 8000]], sample_rate: 16_000 };
            encode_frames(&frames, &path, 24.0, &VideoEncodeOpts { audio: Some(track), ..Default::default() }).expect("encode");
            let meta = probe_video(&path).expect("probe");
            assert_eq!((meta.w, meta.h, meta.frames), (48, 32, 12), "{name}: {meta:?}");
            assert!((meta.fps - 24.0).abs() < 1e-9, "{name}: {meta:?}");
            let back = decode_frames_rgb8(&path, &VideoDecodeOpts { fps: None, max_frames: 0, spread: 0 }).expect("decode");
            assert_eq!(back.len(), 12, "{name}");
            let d = |a: &Rgb8, b: &Rgb8| a.px.iter().zip(&b.px).map(|(x, y)| x.abs_diff(*y) as u64).sum::<u64>();
            // Each decoded frame is nearest the frame written at its place.
            for (i, f) in back.iter().enumerate() {
                let nearest = (0..frames.len()).min_by_key(|&j| d(f, &frames[j])).unwrap();
                assert_eq!(nearest, i, "{name}: decoded frame {i} looks like written frame {nearest}");
            }
        }
    }

    /// Resampling takes, at each output instant, the source frame nearest it;
    /// `max_frames` truncates.
    #[test]
    fn resampling_picks_the_nearest_frame_and_stops_at_the_cap() {
        let frames = moving(24, 32, 16);
        let path = scratch("rate.webm");
        encode_frames(&frames, &path, 24.0, &VideoEncodeOpts::default()).expect("encode");
        let all = decode_frames_rgb8(&path, &VideoDecodeOpts { fps: None, max_frames: 0, spread: 0 }).expect("all");
        let four = decode_frames_rgb8(&path, &VideoDecodeOpts { fps: Some(4.0), max_frames: 0, spread: 0 }).expect("sampled");
        // perf-number: sampling-rate arithmetic of a decode option, not a measurement
        // One second at 4 fps: the instants 0, 0.25, 0.5 and 0.75 s are source frames 0, 6, 12 and 18.
        assert_eq!(four.len(), 4);
        for (k, f) in four.iter().enumerate() {
            assert!(f.px == all[k * 6].px, "the {k}th sampled frame is not source frame {}", k * 6);
        }
        let capped = decode_frames_rgb8(&path, &VideoDecodeOpts { fps: Some(48.0), max_frames: 5, spread: 0 }).expect("capped");
        assert_eq!(capped.len(), 5);
        // At twice the source rate every source frame is shown twice.
        assert!(capped[0].px == capped[1].px && capped[2].px == capped[3].px && capped[1].px != capped[2].px);
    }

    #[test]
    fn a_gif_is_an_animated_gif() {
        let path = scratch("anim.gif");
        encode_frames(&moving(5, 16, 8), &path, 10.0, &VideoEncodeOpts::default()).expect("gif");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
        let decoder = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes)).unwrap();
        use image::AnimationDecoder;
        assert_eq!(decoder.into_frames().count(), 5);
    }

    #[test]
    fn mismatched_frames_and_empty_input_are_clean_errors() {
        let mut frames = moving(2, 8, 8);
        frames.push(Rgb8 { w: 4, h: 4, px: vec![0; 48] });
        let never = scratch("never.mp4");
        let e = encode_frames(&frames, &never, 8.0, &VideoEncodeOpts::default()).unwrap_err();
        assert!(e.contains("frame 2 is 4x4"), "{e}");
        let e = encode_frames(&[], &never, 8.0, &VideoEncodeOpts::default()).unwrap_err();
        assert!(e.contains("no frames"), "{e}");
        let e = encode_frames(&moving(1, 8, 8), &scratch("x.avi"), 8.0, &VideoEncodeOpts::default()).unwrap_err();
        assert!(e.contains(".webm"), "{e}");
    }

    #[test]
    fn missing_file_is_a_clean_error() {
        let err = decode_frames(Path::new("/nonexistent/brain-video-test.mp4"), &VideoDecodeOpts::default()).unwrap_err();
        assert!(err.contains("does not exist"), "expected a clear missing-file error, got: {err}");
    }
}
