// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Record what is being drawn, straight into a video file.
//!
//! Frames are coded as AV1 as they arrive, by brain's own encoder
//! (`videocodec`), on a thread of the recorder's own behind a queue of a few
//! frames: drawing goes on while a frame is coded, and a run that draws
//! faster than frames code is held back rather than piling frames up in
//! memory. Nothing is written to a directory to be swept up afterwards.
//!
//! **A recording never fails a run.** A frame of the wrong size, or a coder
//! that fails mid-run, is reported once and recording stops; what was coded
//! by then is still written when the recording finishes.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread::JoinHandle;

/// Frames queued for the coder before drawing waits for it.
const QUEUE_FRAMES: usize = 4;

/// An open video file being written frame by frame.
pub struct Recorder {
    queue: Option<SyncSender<Vec<u8>>>,
    coder: Option<JoinHandle<Result<(), String>>>,
    path: PathBuf,
    width: u32,
    height: u32,
    frames: u64,
    /// Set once the recording has failed, so it reports itself exactly once
    /// instead of once per frame for the rest of the run.
    broken: bool,
}

impl Recorder {
    /// Start coding `width` x `height` RGB frames at `fps` into `path`, whose
    /// extension picks the container (`.mp4`, `.mov`, `.mkv` or `.webm`).
    pub fn start(path: impl AsRef<Path>, width: u32, height: u32, fps: u32) -> Result<Recorder, String> {
        let path = path.as_ref().to_path_buf();
        let mut writer = videocodec::VideoWriter::create(&path, width, height, fps as f64, videocodec::EncodeSettings::default())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let (queue, frames) = sync_channel::<Vec<u8>>(QUEUE_FRAMES);
        let named = path.clone();
        let coder = std::thread::Builder::new()
            .name("viewport-record".into())
            .spawn(move || {
                let mut failed = None;
                for frame in frames {
                    if failed.is_none() {
                        failed = writer.push(&frame).err();
                    }
                }
                if let Some(e) = failed {
                    eprintln!("viewport: recording {} stopped: {e}", named.display());
                }
                if writer.frames() == 0 {
                    return Err(format!("{}: no frame was recorded", named.display()));
                }
                writer.finish(None).map_err(|e| format!("{}: {e}", named.display()))
            })
            .map_err(|e| format!("starting the recorder: {e}"))?;
        Ok(Recorder { queue: Some(queue), coder: Some(coder), path, width, height, frames: 0, broken: false })
    }

    /// Append one frame. `rgb` must be `width * height * 3` bytes.
    pub fn frame(&mut self, rgb: &[u8]) {
        if self.broken {
            return;
        }
        let want = (self.width * self.height * 3) as usize;
        if rgb.len() != want {
            eprintln!("viewport: recording {} expected {want} bytes a frame and got {}; stopping", self.path.display(), rgb.len());
            self.broken = true;
            return;
        }
        let sent = self.queue.as_ref().is_some_and(|q| q.send(rgb.to_vec()).is_ok());
        if !sent {
            eprintln!("viewport: recording {} stopped: the coder has exited", self.path.display());
            self.broken = true;
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

    /// Code the queued frames and write the file.
    ///
    /// Happens in [`Drop`] too: a recording that is never finished is never written.
    pub fn finish(mut self) -> Result<(u64, PathBuf), String> {
        let frames = self.frames;
        let path = self.path.clone();
        self.close()?;
        Ok((frames, path))
    }

    fn close(&mut self) -> Result<(), String> {
        // Closing the queue ends the coder's loop, and it writes the file.
        self.queue = None;
        match self.coder.take() {
            Some(coder) => coder.join().map_err(|_| format!("the recorder of {} panicked", self.path.display()))?,
            None => Ok(()),
        }
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if self.coder.is_some() {
            if let Err(e) = self.close() {
                eprintln!("viewport: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recording_is_a_playable_file_with_the_frames_that_were_written() {
        let dir = std::env::temp_dir().join("brain-viewport-record");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("test.mp4");

        let (w, h) = (64u32, 32u32);
        let mut rec = Recorder::start(&path, w, h, 10).expect("starts");
        for i in 0..20u32 {
            let mut frame = vec![0u8; (w * h * 3) as usize];
            // Something that changes, so a decoder that returns the same frame
            // twenty times would be visible rather than plausible.
            for px in frame.as_chunks_mut::<3>().0 {
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
        let dir = std::env::temp_dir().join("brain-viewport-record");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("wrong-size.mp4");
        let mut rec = Recorder::start(&path, 16, 16, 10).expect("starts");
        // A short frame coded as if it were whole would shift every pixel
        // after it - a sheared picture that looks like a driver bug.
        rec.frame(&[0u8; 16 * 16 * 3 - 1]);
        assert_eq!(rec.frames(), 0);
        let _ = rec.finish();
        let _ = std::fs::remove_file(&path);
    }
}
