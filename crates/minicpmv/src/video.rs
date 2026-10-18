// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V 4.6's video processor (`MiniCPMV4_6VideoProcessor`): which
//! frames of a clip are seen, and how they are laid out.
//!
//! Frames are chosen by INDEX from the container's metadata, never by
//! timestamp: one per second (`int(i * avg_fps)`), or, for a clip longer than
//! `max_num_frames` seconds, every tenth of a second thinned evenly to
//! `max_num_frames` (`np.linspace(..., dtype=int)`). Each chosen frame is then
//! processed exactly like an image (source view, slices) and laid out as an
//! image placeholder of `<|video_pad|>` tokens; a video carries no timestamp
//! tokens.
//!
//! `stack_frames > 1` (per-second composites of sub-second frames) is refused:
//! the reference builds those composites and then processes the raw frames,
//! so its placeholders and features disagree and its own forward raises (the
//! golden manifest records `tokens: 756, features: 1134`). There is no
//! working reference to match.

use crate::config::Downsample;
use crate::preprocess::ProcessedImage;
use crate::prompt::{view_layout, VIDEO_PAD};

/// The processor's default frame budget.
pub const MAX_NUM_FRAMES: u32 = 128;

/// What the container reports about a clip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipMeta {
    pub frames: u64,
    pub fps: f64,
    pub duration: f64,
}

/// Python's `round(x, 1)`: the decimal nearest the exact binary value, ties
/// to even - Rust's float formatting rounds the same way.
fn round1(x: f64) -> f64 {
    format!("{x:.1}").parse().expect("a formatted float parses")
}

/// `np.linspace(0, n - 1, num, dtype=int)`: `num` indices spread evenly over
/// `0..n`, first and last included.
pub fn linspace_int(n: usize, num: usize) -> Vec<usize> {
    if num == 1 {
        return vec![0];
    }
    let stop = (n - 1) as f64;
    let step = stop / (num - 1) as f64;
    (0..num).map(|i| if i == num - 1 { n - 1 } else { (i as f64 * step) as usize }).collect()
}

/// The frame indices the reference's `sample_frames` picks.
pub fn sample_frames(meta: ClipMeta, max_num_frames: u32, stack_frames: u32) -> Result<Vec<u64>, String> {
    if stack_frames > 1 {
        return Err(format!(
            "minicpmv: stack_frames {stack_frames} is not supported - the reference processor builds the per-second composites and then discards them, so its own forward fails on them"
        ));
    }
    let positive = |x: f64| x.is_finite() && x > 0.0;
    if meta.frames == 0 || !positive(meta.fps) || !positive(meta.duration) {
        return Err(format!("minicpmv: a video needs a positive frame count, fps and duration, got {meta:?}"));
    }
    if max_num_frames == 0 {
        return Err("minicpmv: max_num_frames must be at least 1".into());
    }
    let last = meta.frames - 1;
    if meta.duration > max_num_frames as f64 {
        let steps = (meta.duration / 0.1) as usize;
        let main: Vec<u64> = (0..steps).map(|i| ((round1(i as f64 * 0.1) * meta.fps) as u64).min(last)).collect();
        if main.len() > max_num_frames as usize {
            return Ok(linspace_int(main.len(), max_num_frames as usize).into_iter().map(|i| main[i]).collect());
        }
        return Ok(main);
    }
    Ok((0..meta.duration.ceil() as u64).map(|i| (i as f64 * meta.fps) as u64).collect())
}

/// The frames of a clip on disk the reference would see: the container's
/// metadata, [`sample_frames`], then exactly those frames decoded.
pub fn load_clip(path: &std::path::Path, max_num_frames: u32, stack_frames: u32) -> Result<Vec<imaging::Rgb8>, String> {
    let meta = imaging::video::probe_video(path)?;
    let indices = sample_frames(ClipMeta { frames: meta.frames, fps: meta.fps, duration: meta.duration }, max_num_frames, stack_frames)?;
    imaging::video::decode_frames_at(path, &indices)
}

/// The placeholder text of one video: each frame's image layout in
/// `<|video_pad|>` tokens, back to back (the reference's
/// `replace_video_token`).
pub fn video_layout(frames: &[ProcessedImage], mode: Downsample) -> String {
    frames.iter().map(|f| view_layout(f, mode, VIDEO_PAD)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_round_and_numpy_linspace_are_reproduced() {
        assert_eq!(round1(3.0 * 0.1), 0.3);
        assert_eq!(round1(0.25), 0.2, "ties to even on the exact value");
        assert_eq!(linspace_int(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(linspace_int(5, 1), vec![0]);
    }

    #[test]
    fn a_file_that_is_not_a_video_is_an_error_not_an_empty_clip() {
        if !imaging::video::ffmpeg_available() {
            return;
        }
        let path = std::env::temp_dir().join(format!("brain-minicpmv-notavideo-{}.mp4", std::process::id()));
        std::fs::write(&path, b"this is not a video container").unwrap();
        let e = load_clip(&path, MAX_NUM_FRAMES, 1).unwrap_err();
        assert!(e.contains("ffprobe"), "{e}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_short_clip_is_one_frame_per_second_and_stacking_is_refused() {
        let meta = ClipMeta { frames: 60, fps: 10.0, duration: 6.0 };
        assert_eq!(sample_frames(meta, 128, 1).unwrap(), vec![0, 10, 20, 30, 40, 50]);
        assert!(sample_frames(meta, 128, 3).unwrap_err().contains("stack_frames"));
    }
}
