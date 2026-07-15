// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Recording sink: tee play-session frames into an episode dataset
//! (`data::episode`), so any `brain wm play` run doubles as training-data
//! collection.
//!
//! [`RecorderSink`] wraps [`data::episode::EpisodeWriter`]: each
//! [`FrameSink::frame`] call converts the interleaved RGB8 frame to CHW `u8`
//! and pushes it with the action carried in [`Hud::action`]; a [`Hud::reset`]
//! closes the current episode (Reset in the window / play loop = new
//! episode). A byte budget (default 2 GiB) caps the recording: past it the
//! sink stops recording (one stderr warning) but NEVER fails the session —
//! what was captured before the cap is still finalized.

use std::path::PathBuf;

use crate::sink::{FrameSink, Hud};
use data::episode::EpisodeWriter;

/// Default recording byte budget: 2 GiB.
pub const DEFAULT_BYTE_BUDGET: u64 = 2 * 1024 * 1024 * 1024;

/// Convert an interleaved HWC RGB8 frame (`w*h*3` bytes) to planar CHW `u8`
/// (the episode-dataset frame layout).
pub fn rgb8_to_chw(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let plane = (w * h) as usize;
    assert_eq!(rgb.len(), plane * 3, "rgb8_to_chw: expected {}x{}x3 bytes", w, h);
    let mut chw = vec![0u8; plane * 3];
    for i in 0..plane {
        for c in 0..3 {
            chw[c * plane + i] = rgb[i * 3 + c];
        }
    }
    chw
}

/// A [`FrameSink`] that records every frame + action into an episode
/// dataset. Create it, tee it with the display sink, and call
/// [`RecorderSink::finalize`] when the session ends — without finalize no
/// dataset directory appears (the writer's atomicity contract).
pub struct RecorderSink {
    dir: PathBuf,
    num_actions: u32,
    fps: u32,
    budget: u64,
    bytes: u64,
    /// Created lazily on the first frame (that is when `w`/`h` are known).
    writer: Option<EpisodeWriter>,
    /// Set on budget exhaustion or a write error: silently drop frames.
    stopped: bool,
}

impl RecorderSink {
    /// Record into `dir` (must not exist yet) with the default 2 GiB budget.
    /// `num_actions`/`fps` go into the dataset meta.
    pub fn new(dir: impl Into<PathBuf>, num_actions: u32, fps: u32) -> RecorderSink {
        Self::with_budget(dir, num_actions, fps, DEFAULT_BYTE_BUDGET)
    }

    /// [`RecorderSink::new`] with an explicit byte budget for frame data.
    pub fn with_budget(
        dir: impl Into<PathBuf>,
        num_actions: u32,
        fps: u32,
        byte_budget: u64,
    ) -> RecorderSink {
        RecorderSink {
            dir: dir.into(),
            num_actions,
            fps,
            budget: byte_budget,
            bytes: 0,
            writer: None,
            stopped: false,
        }
    }

    /// Frames recorded so far.
    pub fn frames_recorded(&self) -> usize {
        self.writer.as_ref().map_or(0, |w| w.len())
    }

    /// Finish the recording: atomically publish the dataset directory.
    /// A recorder that never saw a frame finalizes to nothing (Ok, no dir).
    pub fn finalize(mut self) -> Result<(), String> {
        match self.writer.take() {
            Some(w) => w.finalize(),
            None => Ok(()),
        }
    }

    fn stop(&mut self, why: &str) {
        // Log ONCE, keep the session running; already-recorded frames are
        // preserved and still finalized.
        eprintln!("record: {why}; recording stopped (frames so far are kept)");
        self.stopped = true;
    }
}

impl FrameSink for RecorderSink {
    fn frame(&mut self, rgb: &[u8], w: u32, h: u32, hud: &Hud) {
        if self.stopped {
            return;
        }
        if self.writer.is_none() {
            match EpisodeWriter::create(&self.dir, 3, h, w, self.num_actions, self.fps) {
                Ok(wr) => self.writer = Some(wr),
                Err(e) => return self.stop(&format!("cannot create {}: {e}", self.dir.display())),
            }
        }
        if hud.reset {
            // First frame after a reset: the previous episode is over.
            self.writer.as_mut().unwrap().end_episode();
        }
        // Frame bytes + the 4-byte action; recordings carry no rewards.
        let cost = rgb.len() as u64 + 4;
        if self.bytes + cost > self.budget {
            let n = self.frames_recorded();
            return self.stop(&format!(
                "byte budget reached ({} of {} bytes after {n} frames)",
                self.bytes, self.budget
            ));
        }
        if let Err(e) = self.writer.as_mut().unwrap().push(&rgb8_to_chw(rgb, w, h), hud.action, None)
        {
            return self.stop(&e);
        }
        self.bytes += cost;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keymap::KeyChordMap;
    use crate::pacing::{FixedTimestep, MockClock};
    use crate::{chw_to_rgb8, play_loop, ScriptInput, SplitIo};
    use data::episode::EpisodeDataset;
    use wm_core::{FakeWorldModel, WorldModel};

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("wm_record_{name}_{}", std::process::id()))
    }

    /// Record a FakeWorldModel session through the real play loop, then step
    /// an identically-reset model over the dataset's actions: every recorded
    /// frame must match EXACTLY, and the recorded actions must equal the
    /// script (Hud::action plumbing).
    #[test]
    fn record_replay_roundtrip_fake_model_exact() {
        let dir = tmp("roundtrip");
        let _ = std::fs::remove_dir_all(&dir);
        let script = vec![1u32, 1, 4, 4, 0, 3, 2, 2, 1, 0];

        let mut model = FakeWorldModel::new();
        model.reset(&[], &[]);
        let mut input = ScriptInput::new(script.clone());
        let mut rec = RecorderSink::new(&dir, model.num_actions(), 15);
        let pacer = FixedTimestep::new(MockClock { now: 0 }, 15, false);
        let mut io = SplitIo { input: &mut input, sink: &mut rec };
        let report =
            play_loop(&mut model, &mut io, &KeyChordMap::wasd(0), pacer, 15, "fake", None);
        assert_eq!(report.steps, script.len() as u64);
        assert!(!dir.exists(), "no dataset before finalize");
        rec.finalize().unwrap();

        let ds = EpisodeDataset::open(&dir).unwrap();
        assert_eq!((ds.n, ds.c, ds.h, ds.w), (script.len(), 3, 64, 64));
        assert_eq!(ds.num_actions, 5);
        assert_eq!(ds.actions(), &script[..], "recorded actions must equal the script");
        assert!(ds.rewards().is_none());

        // Replay: same reset -> byte-identical frames through the same
        // f32 -> RGB8 -> CHW conversion the recorder used.
        let mut replay = FakeWorldModel::new();
        replay.reset(&[], &[]);
        let (c, h, w) = replay.frame_shape();
        for (i, &a) in ds.actions().iter().enumerate() {
            let f = replay.step(a);
            let got = rgb8_to_chw(&chw_to_rgb8(&f, c, h, w), w, h);
            assert_eq!(got, ds.frame(i).unwrap(), "frame {i} differs");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_tiny_cap_stops_cleanly_and_keeps_prefix() {
        let dir = tmp("cap");
        let _ = std::fs::remove_dir_all(&dir);
        let frame_cost = (64 * 64 * 3 + 4) as u64;
        let mut model = FakeWorldModel::new();
        model.reset(&[], &[]);
        let mut input = ScriptInput::new(vec![4; 10]);
        // Budget for exactly 3 frames.
        let mut rec = RecorderSink::with_budget(&dir, model.num_actions(), 15, 3 * frame_cost);
        let pacer = FixedTimestep::new(MockClock { now: 0 }, 15, false);
        let mut io = SplitIo { input: &mut input, sink: &mut rec };
        let report =
            play_loop(&mut model, &mut io, &KeyChordMap::wasd(0), pacer, 15, "fake", None);
        // The session itself is unaffected by the cap.
        assert_eq!(report.steps, 10);
        assert_eq!(rec.frames_recorded(), 3);
        rec.finalize().unwrap();
        let ds = EpisodeDataset::open(&dir).unwrap();
        assert_eq!(ds.n, 3);
        assert_eq!(ds.actions(), &[4, 4, 4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_hud_reset_splits_episodes() {
        let dir = tmp("reset");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = RecorderSink::new(&dir, 5, 15);
        let rgb = vec![9u8; 4 * 4 * 3];
        let mut hud = Hud::default();
        for i in 0..5 {
            hud.action = i;
            hud.reset = i == 3; // frame 3 starts a new episode
            rec.frame(&rgb, 4, 4, &hud);
        }
        rec.finalize().unwrap();
        let ds = EpisodeDataset::open(&dir).unwrap();
        assert_eq!(ds.n, 5);
        assert_eq!(ds.episodes.len(), 2);
        assert_eq!((ds.episodes[0].start, ds.episodes[0].len), (0, 3));
        assert_eq!((ds.episodes[1].start, ds.episodes[1].len), (3, 2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_finalize_without_frames_creates_nothing() {
        let dir = tmp("empty");
        let _ = std::fs::remove_dir_all(&dir);
        let rec = RecorderSink::new(&dir, 5, 15);
        rec.finalize().unwrap();
        assert!(!dir.exists());
    }
}
