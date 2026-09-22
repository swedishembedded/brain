// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Record what is being drawn, straight into an MP4.
//!
//! Frames go to an `ffmpeg` subprocess's STDIN as raw RGB and are encoded as
//! they arrive. Nothing is buffered: the memory cost of a recording is one
//! frame, whether it is ten seconds long or an hour, and no PNGs are written
//! to a directory to be swept up afterwards.
//!
//! That last point is the whole design. Writing numbered images and encoding
//! them later costs disk proportional to the run, needs a cleanup step that
//! will eventually not happen, and turns "record this" into two operations
//! that can disagree about how many frames there were. A pipe has none of
//! those properties and is thirty lines.
//!
//! **A recording never fails a run.** `ffmpeg` is an optional external tool, so
//! an absent binary is reported once and recording is simply off. A pipe that
//! breaks mid-run - ffmpeg killed, disk full - is reported once and recording
//! stops; whatever was encoded up to that point is still a file, because
//! ffmpeg finalizes on EOF.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

/// An open MP4 being written frame by frame.
pub struct Recorder {
    child: Child,
    stdin: Option<ChildStdin>,
    path: PathBuf,
    width: u32,
    height: u32,
    frames: u64,
    /// Set once the pipe has failed, so a broken recording reports itself
    /// exactly once instead of once per frame for the rest of the run.
    broken: bool,
}

impl Recorder {
    /// Start encoding `width` x `height` RGB frames at `fps` into `path`.
    pub fn start(
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Recorder, String> {
        if !imaging::video::ffmpeg_available() {
            return Err(
                "ffmpeg is not on PATH; install it to record (the run continues without)"
                    .to_string(),
            );
        }
        let path = path.as_ref().to_path_buf();
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            }
        }
        let mut child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            // The input is exactly what the canvas holds: tightly packed RGB8,
            // no container, no header. ffmpeg has to be told its shape because
            // raw video carries none.
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &fps.to_string()])
            .args(["-i", "-"])
            // yuv420p and an even frame size, because a great deal of software
            // - browsers, phones, QuickTime - silently refuses anything else,
            // and a recording nobody can play is not a recording.
            .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
            .args([
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            ])
            .arg(&path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawning ffmpeg: {e}"))?;
        let stdin = child.stdin.take();
        Ok(Recorder {
            child,
            stdin,
            path,
            width,
            height,
            frames: 0,
            broken: false,
        })
    }

    /// Append one frame. `rgb` must be `width * height * 3` bytes.
    pub fn frame(&mut self, rgb: &[u8]) {
        if self.broken {
            return;
        }
        let want = (self.width * self.height * 3) as usize;
        if rgb.len() != want {
            eprintln!(
                "viewport: recording {} expected {want} bytes a frame and got {}; stopping",
                self.path.display(),
                rgb.len()
            );
            self.broken = true;
            return;
        }
        let Some(stdin) = self.stdin.as_mut() else {
            self.broken = true;
            return;
        };
        if let Err(e) = stdin.write_all(rgb) {
            eprintln!("viewport: recording {} stopped: {e}", self.path.display());
            self.broken = true;
            self.stdin = None;
            return;
        }
        self.frames += 1;
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Close the pipe and wait for ffmpeg to finalize the file.
    ///
    /// Must happen, and happens in [`Drop`] too: an MP4 whose trailer was never
    /// written is a file that exists and will not play, which is worse than no
    /// file at all.
    pub fn finish(mut self) -> Result<(u64, PathBuf), String> {
        let frames = self.frames;
        let path = self.path.clone();
        self.close()?;
        Ok((frames, path))
    }

    fn close(&mut self) -> Result<(), String> {
        // Dropping stdin sends EOF, which is what tells ffmpeg to write the
        // trailer and exit.
        self.stdin = None;
        match self.child.wait() {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(format!("ffmpeg exited {s} writing {}", self.path.display())),
            Err(e) => Err(format!("waiting for ffmpeg: {e}")),
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if self.stdin.is_some() {
            let _ = self.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recording_is_a_playable_file_with_the_frames_that_were_written() {
        if !imaging::video::ffmpeg_available() {
            brain_testutil::skip_unavailable("ffmpeg not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join("brain-viewport-record");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("test.mp4");

        let (w, h) = (64u32, 32u32);
        let mut rec = Recorder::start(&path, w, h, 10).expect("starts");
        for i in 0..20u32 {
            let mut frame = vec![0u8; (w * h * 3) as usize];
            // Something that changes, so a decoder that returns the same frame
            // twenty times would be visible rather than plausible.
            for px in frame.chunks_exact_mut(3) {
                px[0] = (i * 12) as u8;
            }
            rec.frame(&frame);
        }
        let (frames, out) = rec.finish().expect("finishes");
        assert_eq!(frames, 20);

        // Read it back through the decoder, which is the only check that the
        // file is actually a video rather than a plausible pile of bytes.
        // NOT Default: that resamples to 1fps and caps at 32 frames, which is
        // right for feeding a few seconds to a multimodal model and wrong for
        // asking "did every frame I wrote come back".
        let opts = imaging::video::VideoDecodeOpts {
            fps: None,
            max_frames: 0,
            ..Default::default()
        };
        let decoded =
            imaging::video::decode_frames_rgb8(&out, &opts).expect("the recording decodes");
        assert_eq!(decoded.len(), 20, "every frame written came back");
        assert_eq!((decoded[0].w, decoded[0].h), (w, h));
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn a_wrong_sized_frame_stops_the_recording_instead_of_corrupting_it() {
        if !imaging::video::ffmpeg_available() {
            brain_testutil::skip_unavailable("ffmpeg not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join("brain-viewport-record");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("wrong-size.mp4");
        let mut rec = Recorder::start(&path, 8, 8, 10).expect("starts");
        // Raw video has no framing, so a short frame would not be rejected by
        // ffmpeg - it would shift every following frame by the difference and
        // produce a diagonally sheared video that looks like a driver bug.
        rec.frame(&vec![0u8; 8 * 8 * 3 - 1]);
        assert_eq!(rec.frames(), 0);
        let _ = rec.finish();
        let _ = std::fs::remove_file(&path);
    }
}
