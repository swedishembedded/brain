// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Video file → RGB frames, via the `ffmpeg` CLI (subprocess, not a build
//! dependency). The workspace has no video-container/codec crate at all
//! (`qwen3omnimoe::mm::encode_video_frames` already took already-decoded frames, but
//! nothing turned a real video FILE into them until this) -- pure-Rust
//! demuxers/decoders for real containers are
//! immature or absent, and this repo is deliberately dependency-light
//! (`AGENTS.md`), so this follows the same subprocess pattern already used
//! elsewhere (`crates/perf/src/env.rs`, `crates/npu/src/openvino/real.rs`).
//!
//! Frames are written by `ffmpeg` to numbered PPM (P6) files in a fresh temp
//! directory, then read back through [`crate::codec::load`] — NOT parsed from
//! one concatenated stdout stream: `events::ppm::decode_p6` tolerates
//! trailing bytes but does not report how many it consumed, so multi-frame
//! stream-splitting would need a second PPM parser tracking byte offsets.
//! One frame per file reuses the existing, already-tested decoder instead.
//!
//! [`ffmpeg_available`] lets a caller skip cleanly when the binary is absent
//! (the pattern `examples/omni/omni.py`'s PyAV path already documents for
//! its own optional dependency) rather than making `ffmpeg` a hard
//! requirement of every brain build.
//!
//! [`encode_frames`] is the mirror image: RGB frames -> numbered PPMs ->
//! one `ffmpeg` invocation. When `ffmpeg` is absent it still writes the
//! numbered PPMs (to a directory beside the requested output) and hands back
//! the exact command that finishes the job, because a video model that
//! produced 81 good frames and then found no encoder has not failed - it has
//! one step left, and the user needs to be told which.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::pixels::Rgb8;

/// True if the `ffmpeg` CLI is on `PATH` and runs. Cheap (`ffmpeg -version`,
/// no video touched) — call this before [`decode_frames`] to skip cleanly
/// with a caller-chosen message, rather than parsing [`decode_frames`]'s
/// error string to detect the same condition.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// Frame-selection knobs for [`decode_frames`].
pub struct VideoDecodeOpts {
    /// Resample to this many frames per second before selecting frames.
    /// `None` keeps the source's native frame rate.
    pub fps: Option<f64>,
    /// Stop after this many frames. `0` means unbounded (still implicitly
    /// capped by the clip's own duration at the chosen `fps`).
    pub max_frames: u32,
}

impl Default for VideoDecodeOpts {
    /// A short clip at a low frame rate — sane for a multimodal prompt (a
    /// few seconds of video, not a full-length feature), matching the scale
    /// `qwen3omnimoe::mm::encode_video_frames` is validated at.
    fn default() -> VideoDecodeOpts {
        VideoDecodeOpts { fps: Some(1.0), max_frames: 32 }
    }
}

/// Decode a video file to RGB frames: one `(hwc_f32_unit, w, h)` tuple per
/// frame, in order - the exact shape `qwen3omnimoe::mm::encode_video_frames` takes
/// (`Rgb8::to_hwc_unit`'s output, `capability::blob::decode_video`'s wire
/// convention). Returns a clear error (not a panic) when `ffmpeg` is absent,
/// the path doesn't exist, ffmpeg fails, or it produces zero frames.
pub fn decode_frames(path: &Path, opts: &VideoDecodeOpts) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
    Ok(decode_frames_rgb8(path, opts)?.into_iter().map(|img| (img.to_hwc_unit(), img.w, img.h)).collect())
}

/// The same decode, kept as [`Rgb8`].
///
/// [`decode_frames`]' `f32` unit form is what a model PREPROCESSOR wants; a
/// caller that will hand the frames straight back to a codec (`brain ltxv
/// upscale` re-encoding what it decoded) wants the bytes ffmpeg actually
/// produced, without a `u8 -> f32 -> u8` round trip in the middle.
pub fn decode_frames_rgb8(path: &Path, opts: &VideoDecodeOpts) -> Result<Vec<Rgb8>, String> {
    if !ffmpeg_available() {
        return Err("imaging::video: ffmpeg not found on PATH -- video file decoding needs the ffmpeg CLI (install it, or supply already-decoded frames directly)".to_string());
    }
    if !path.exists() {
        return Err(format!("imaging::video: {} does not exist", path.display()));
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-video-decode-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("imaging::video: creating {}: {e}", dir.display()))?;
    let _cleanup = TempDirGuard(&dir);

    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y").arg("-i").arg(path);
    if let Some(fps) = opts.fps {
        cmd.arg("-vf").arg(format!("fps={fps}"));
    }
    if opts.max_frames > 0 {
        cmd.arg("-frames:v").arg(opts.max_frames.to_string());
    }
    cmd.arg(dir.join("frame_%05d.ppm"));
    let out = cmd.output().map_err(|e| format!("imaging::video: spawning ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!("imaging::video: ffmpeg exited {}: {}", out.status, String::from_utf8_lossy(&out.stderr)));
    }

    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("imaging::video: reading {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "ppm"))
        .collect();
    entries.sort();
    if entries.is_empty() {
        return Err(format!("imaging::video: ffmpeg produced no frames for {}", path.display()));
    }

    entries.iter().map(crate::codec::load).collect()
}

/// The video's own average frame rate, via `ffprobe`.
///
/// `None` when `ffprobe` is absent (it ships with `ffmpeg` but is a separate
/// binary and can be packaged apart from it), when the stream reports no
/// rate, or when the rate is not a positive number - every one of which is a
/// "ask the caller instead" case, not an error worth failing a run over.
/// Returned as the exact `num/den` rational ffmpeg carries, evaluated, so
/// 24000/1001 comes back as 23.976 rather than 24.
pub fn probe_fps(path: &Path) -> Option<f64> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1"])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (num, den) = text.trim().split_once('/')?;
    let (num, den) = (num.parse::<f64>().ok()?, den.parse::<f64>().ok()?);
    (den > 0.0 && num > 0.0).then_some(num / den)
}

// ---------------------------------------------------------------- encoding

/// Container/codec knobs for [`encode_frames`].
pub struct VideoEncodeOpts {
    /// x264 constant-rate factor: 0 lossless, 18 visually near-lossless, 51
    /// worst. Ignored for containers that do not take an x264 stream (`.gif`).
    pub crf: u32,
    /// Where the numbered PPMs go when `ffmpeg` is missing. `None` derives
    /// `<output>.frames/` so the frames land beside the file the caller asked
    /// for, not in a temp directory that the next reboot deletes.
    pub frames_dir: Option<PathBuf>,
}

impl Default for VideoEncodeOpts {
    fn default() -> VideoEncodeOpts {
        VideoEncodeOpts { crf: 18, frames_dir: None }
    }
}

/// What [`encode_frames`] actually produced.
///
/// The two arms are both successes: a caller with `ffmpeg` gets a container, a
/// caller without one gets the frames plus the command line. Returning an
/// error for the second case would throw away a generation that may have taken
/// an hour.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Encoded {
    /// `ffmpeg` wrote the container at this path.
    Video(PathBuf),
    /// `ffmpeg` was not on `PATH`. The numbered PPMs are in `dir`; `command`
    /// is the invocation that turns them into the requested file.
    Frames { dir: PathBuf, command: String },
}

/// Write `frames` as `frame_00001.ppm`, `frame_00002.ppm`, … into `dir`
/// (created if needed) and return the `ffmpeg` command that would assemble
/// them into `out` at `fps`.
///
/// Public and separately tested because it is the no-ffmpeg path: on a machine
/// that HAS ffmpeg, [`encode_frames`] never reaches it, so a test that only
/// drives `encode_frames` would leave the fallback unexercised exactly where it
/// matters.
pub fn write_frame_dir(frames: &[Rgb8], dir: &Path, out: &Path, fps: f64, crf: u32) -> Result<String, String> {
    let (w, h) = check_frames(frames)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("imaging::video: creating {}: {e}", dir.display()))?;
    for (i, f) in frames.iter().enumerate() {
        crate::codec::save_ppm(dir.join(format!("frame_{:05}.ppm", i + 1)), f)?;
    }
    let args = ffmpeg_args(dir, out, fps, crf, w, h);
    Ok(std::iter::once("ffmpeg".to_string())
        .chain(args.into_iter().map(|a| shell_quote(&a)))
        .collect::<Vec<_>>()
        .join(" "))
}

/// Encode RGB frames into a video file at `fps`.
///
/// Every frame must have the same non-zero dimensions. `-pix_fmt yuv420p` is
/// forced for the H.264/VP9 containers, because the default for RGB input is
/// `yuv444p`, which Safari, QuickTime and most browsers refuse to play - a
/// file that "works" only in the tool that wrote it.
///
/// **Odd dimensions are padded, loudly.** 4:2:0 chroma subsampling needs even
/// width and height; without the pad, libx264 rejects the stream outright and
/// with some other encoders the last row/column is quietly dropped. One black
/// row/column plus a line on stderr is the honest version of both.
pub fn encode_frames(frames: &[Rgb8], path: &Path, fps: f64, opts: &VideoEncodeOpts) -> Result<Encoded, String> {
    let (w, h) = check_frames(frames)?;
    if !(fps.is_finite() && fps > 0.0) {
        return Err(format!("imaging::video: fps must be finite and positive (got {fps})"));
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("imaging::video: creating {}: {e}", dir.display()))?;
    }

    if !ffmpeg_available() {
        let dir = opts.frames_dir.clone().unwrap_or_else(|| frames_dir_for(path));
        let command = write_frame_dir(frames, &dir, path, fps, opts.crf)?;
        return Ok(Encoded::Frames { dir, command });
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("brain-video-encode-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("imaging::video: creating {}: {e}", dir.display()))?;
    let _cleanup = TempDirGuard(&dir);
    for (i, f) in frames.iter().enumerate() {
        crate::codec::save_ppm(dir.join(format!("frame_{:05}.ppm", i + 1)), f)?;
    }
    if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        eprintln!("imaging::video: {w}x{h} has an odd dimension; padding to {}x{} so 4:2:0 chroma is representable", w.next_multiple_of(2), h.next_multiple_of(2));
    }

    let out = Command::new("ffmpeg")
        .args(ffmpeg_args(&dir, path, fps, opts.crf, w, h))
        .output()
        .map_err(|e| format!("imaging::video: spawning ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!("imaging::video: ffmpeg exited {}: {}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    if !path.exists() {
        return Err(format!("imaging::video: ffmpeg reported success but {} does not exist", path.display()));
    }
    Ok(Encoded::Video(path.to_path_buf()))
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

/// `out.mp4` -> `out.mp4.frames`. Deliberately keeps the full file name rather
/// than the stem, so two requests differing only in extension cannot collide.
fn frames_dir_for(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_else(|| std::ffi::OsStr::new("video")).to_os_string();
    name.push(".frames");
    path.parent().map(|p| p.join(&name)).unwrap_or_else(|| PathBuf::from(name))
}

/// The one place the command line is built, so the fallback's printed command
/// and the command this module runs itself cannot drift.
fn ffmpeg_args(dir: &Path, out: &Path, fps: f64, crf: u32, w: u32, h: u32) -> Vec<String> {
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let mut v: Vec<String> = vec![
        "-y".into(),
        "-framerate".into(),
        format!("{fps}"),
        "-i".into(),
        dir.join("frame_%05d.ppm").to_string_lossy().into_owned(),
    ];
    if !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        // `pad` keeps the pixels the model produced and adds at most one black
        // row/column; `scale` would resample every frame instead.
        v.push("-vf".into());
        v.push(format!("pad={}:{}:0:0", w.next_multiple_of(2), h.next_multiple_of(2)));
    }
    if ext != "gif" {
        if matches!(ext.as_str(), "mp4" | "mov" | "m4v" | "mkv") {
            v.push("-c:v".into());
            v.push("libx264".into());
            v.push("-crf".into());
            v.push(crf.to_string());
        }
        v.push("-pix_fmt".into());
        v.push("yuv420p".into());
    }
    v.push(out.to_string_lossy().into_owned());
    v
}

/// Minimal POSIX single-quote quoting - the printed fallback command has to be
/// copy-pasteable even when a path has a space in it.
fn shell_quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./%=:,+".contains(c)) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Removes the temp frame directory on drop, success or error alike — so a
/// mid-decode failure never leaks numbered PPM files into the system temp
/// directory.
struct TempDirGuard<'a>(&'a std::path::Path);
impl Drop for TempDirGuard<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real, not mocked: encodes a tiny synthetic clip with `ffmpeg` itself
    /// (`-f lavfi -i testsrc`, a built-in test-pattern source — no fixture
    /// file needed), then decodes it back through THIS module and checks the
    /// frame count and shape are exactly what was asked for. Skips cleanly
    /// if `ffmpeg` is not installed in this environment, matching
    /// [`ffmpeg_available`]'s own contract.
    #[test]
    fn round_trips_a_real_synthetic_clip() {
        if !ffmpeg_available() {
            brain_testutil::skip_unavailable("ffmpeg not installed");
            return;
        }
        let dir = std::env::temp_dir().join("brain-imaging-video-test");
        let _ = std::fs::create_dir_all(&dir);
        let clip = dir.join("tiny.mp4");
        let enc = Command::new("ffmpeg")
            .args(["-y", "-f", "lavfi", "-i", "testsrc=size=32x16:rate=4:duration=2", "-pix_fmt", "yuv420p"])
            .arg(&clip)
            .output()
            .expect("spawning ffmpeg to encode the test clip");
        assert!(enc.status.success(), "encoding the test clip failed: {}", String::from_utf8_lossy(&enc.stderr));

        let frames = decode_frames(&clip, &VideoDecodeOpts { fps: Some(2.0), max_frames: 3 }).expect("decode_frames on a real clip must succeed");
        assert_eq!(frames.len(), 3, "fps=2 duration=2 max_frames=3 must yield exactly 3 frames");
        for (hwc, w, h) in &frames {
            assert_eq!((*w, *h), (32, 16), "frame dims must match the encoded clip");
            assert_eq!(hwc.len(), 32 * 16 * 3, "HWC f32 buffer must be w*h*3");
            assert!(hwc.iter().all(|v| (0.0..=1.0).contains(v)), "HWC values must be in [0,1]");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A synthetic clip whose frames all differ: a moving bright block on a
    /// ramp. "All frames identical" is the failure mode a video pipeline hides
    /// behind "it ran", so the fixture itself has to be non-constant.
    fn moving_block(n: usize, w: u32, h: u32) -> Vec<Rgb8> {
        (0..n)
            .map(|f| {
                let mut px = vec![0u8; (w * h * 3) as usize];
                for y in 0..h {
                    for x in 0..w {
                        let i = ((y * w + x) * 3) as usize;
                        let on = (x as usize + f).is_multiple_of(w as usize);
                        px[i] = if on { 255 } else { (x * 255 / w.max(1)) as u8 };
                        px[i + 1] = (y * 255 / h.max(1)) as u8;
                        px[i + 2] = (f * 20) as u8;
                    }
                }
                Rgb8::new(w, h, px).unwrap()
            })
            .collect()
    }

    /// The real path: encode with the real `ffmpeg`, then decode the result
    /// back through this module's own [`decode_frames`] and check the count,
    /// the dimensions, and that the frames are not all the same picture.
    #[test]
    fn encodes_a_real_playable_clip() {
        if !ffmpeg_available() {
            brain_testutil::skip_unavailable("ffmpeg not installed");
            return;
        }
        let dir = std::env::temp_dir().join("brain-imaging-video-encode-test");
        let _ = std::fs::remove_dir_all(&dir);
        let out = dir.join("clip.mp4");
        let frames = moving_block(8, 32, 16);
        let got = encode_frames(&frames, &out, 8.0, &VideoEncodeOpts::default()).expect("encode");
        assert_eq!(got, Encoded::Video(out.clone()));
        assert!(out.metadata().expect("output exists").len() > 0);

        let back = decode_frames(&out, &VideoDecodeOpts { fps: Some(8.0), max_frames: 8 }).expect("decode back");
        assert_eq!(back.len(), 8, "8 frames in, 8 frames out");
        assert_eq!((back[0].1, back[0].2), (32, 16));
        let d: f32 = back[0].0.iter().zip(&back[7].0).map(|(a, b)| (a - b).abs()).sum();
        assert!(d > 1.0, "first and last decoded frames are identical - the clip is static");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// yuv420p cannot represent odd dimensions, so the encoder pads. The
    /// contract is that the file EXISTS and decodes at the padded size, not
    /// that it silently keeps the odd one.
    #[test]
    fn odd_dimensions_are_padded_not_dropped() {
        if !ffmpeg_available() {
            brain_testutil::skip_unavailable("ffmpeg not installed");
            return;
        }
        let dir = std::env::temp_dir().join("brain-imaging-video-odd-test");
        let _ = std::fs::remove_dir_all(&dir);
        let out = dir.join("odd.mp4");
        encode_frames(&moving_block(4, 7, 5), &out, 4.0, &VideoEncodeOpts::default()).expect("encode odd dims");
        let back = decode_frames(&out, &VideoDecodeOpts { fps: Some(4.0), max_frames: 4 }).expect("decode back");
        assert_eq!((back[0].1, back[0].2), (8, 6), "odd dims must be padded up, not truncated");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The no-ffmpeg path, driven directly so it is covered on a machine that
    /// DOES have ffmpeg: every frame on disk as a numbered PPM, and a command
    /// string that actually runs.
    #[test]
    fn the_frame_directory_fallback_writes_frames_and_a_runnable_command() {
        let root = std::env::temp_dir().join("brain-imaging-video-fallback-test");
        let _ = std::fs::remove_dir_all(&root);
        let out = root.join("clip.mp4");
        let frames = moving_block(5, 16, 8);
        let dir = root.join("clip.mp4.frames");
        let cmd = write_frame_dir(&frames, &dir, &out, 12.0, 18).expect("fallback write");

        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .expect("frames dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["frame_00001.ppm", "frame_00002.ppm", "frame_00003.ppm", "frame_00004.ppm", "frame_00005.ppm"]);
        // The frames must be the real pictures, readable by the normal loader.
        let img = crate::codec::load(dir.join("frame_00003.ppm")).expect("load a fallback frame");
        assert_eq!((img.w, img.h), (16, 8));
        assert_eq!(img.px, frames[2].px);

        // The printed command must name the input pattern, the frame rate and
        // the output - a fallback message that does not actually work is worse
        // than none.
        assert!(cmd.starts_with("ffmpeg "), "{cmd}");
        assert!(cmd.contains("frame_%05d.ppm"), "{cmd}");
        assert!(cmd.contains("-framerate 12"), "{cmd}");
        assert!(cmd.contains("yuv420p"), "{cmd}");
        assert!(cmd.ends_with(&out.to_string_lossy().into_owned()), "{cmd}");
        if ffmpeg_available() {
            let st = Command::new("sh").arg("-c").arg(&cmd).output().expect("run the printed command");
            assert!(st.status.success(), "the printed fallback command failed: {}", String::from_utf8_lossy(&st.stderr));
            assert!(out.exists(), "the printed fallback command produced no file");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Mismatched frame sizes are the classic way a frame list turns into a
    /// corrupt container, and ffmpeg's own error for it is unreadable.
    #[test]
    fn mismatched_frames_and_empty_input_are_clean_errors() {
        let mut frames = moving_block(2, 8, 8);
        frames.push(Rgb8::new(4, 4, vec![0; 48]).unwrap());
        let never = std::env::temp_dir().join("brain-never-written.mp4");
        let e = encode_frames(&frames, &never, 8.0, &VideoEncodeOpts::default()).unwrap_err();
        assert!(e.contains("frame 2 is 4x4"), "{e}");
        let e = encode_frames(&[], &never, 8.0, &VideoEncodeOpts::default()).unwrap_err();
        assert!(e.contains("no frames"), "{e}");
    }

    #[test]
    fn missing_file_is_a_clean_error() {
        if !ffmpeg_available() {
            brain_testutil::skip_unavailable("ffmpeg not installed");
            return;
        }
        let err = decode_frames(Path::new("/nonexistent/brain-video-test.mp4"), &VideoDecodeOpts::default()).unwrap_err();
        assert!(err.contains("does not exist"), "expected a clear missing-file error, got: {err}");
    }
}
