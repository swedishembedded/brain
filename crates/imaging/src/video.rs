// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Video file → RGB frames, via the `ffmpeg` CLI (subprocess, not a build
//! dependency). The workspace has no video-container/codec crate at all
//! (`qwen3omnimoe::mm::encode_video_frames` already took already-decoded frames, but
//! nothing turned a real video FILE into them until this) -- pure-Rust
//! demuxers/decoders for real containers are
//! immature or absent, and this repo is deliberately dependency-light
//! (`AGENTS.md`), so this follows the same subprocess pattern already used
//! elsewhere (`crates/perf/src/energy.rs`, `crates/npu/src/openvino/real.rs`).
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

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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
/// (`Rgb8::to_hwc_unit`'s output, `capability::blob::decode_video_hwc`'s wire
/// convention). Returns a clear error (not a panic) when `ffmpeg` is absent,
/// the path doesn't exist, ffmpeg fails, or it produces zero frames.
pub fn decode_frames(path: &Path, opts: &VideoDecodeOpts) -> Result<Vec<(Vec<f32>, u32, u32)>, String> {
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

    entries.iter().map(|p| crate::codec::load(p).map(|img| (img.to_hwc_unit(), img.w, img.h))).collect()
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
            eprintln!("skip: ffmpeg not installed");
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

    #[test]
    fn missing_file_is_a_clean_error() {
        if !ffmpeg_available() {
            eprintln!("skip: ffmpeg not installed");
            return;
        }
        let err = decode_frames(Path::new("/nonexistent/brain-video-test.mp4"), &VideoDecodeOpts::default()).unwrap_err();
        assert!(err.contains("does not exist"), "expected a clear missing-file error, got: {err}");
    }
}
