// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A window for a sample application, and the same pixels without one.
//!
//! Every sample that wants to SHOW something needs the same four things: an
//! RGB8 buffer to draw into, a way to get it on screen, a way to notice the
//! user pressing a key, and a way to save it when there is no screen. Before
//! this crate each one grew its own, or did without.
//!
//! The rule that shapes the design: **the overlay is drawn into the canvas,
//! not composited by the window**. A headless run therefore produces exactly
//! the image a human would have been looking at, which is what makes a
//! screenshot in a README evidence rather than decoration, and what lets the
//! whole visual layer be tested with no display attached. Anything drawn by
//! the presentation layer instead would be missing from every artifact a
//! server-side run can produce.
//!
//! ```no_run
//! # fn main() -> Result<(), String> {
//! let mut v = viewport::Viewport::open("my sample", 960, 600)?;
//! v.canvas().clear([12, 12, 16]);
//! v.canvas().text(8, 8, "hello", 2, [255, 255, 255]);
//! v.present();                       // no-op when headless
//! v.save("out/frame.png")?;          // the same pixels, either way
//! # Ok(()) }
//! ```
//!
//! It is NOT linked by the `brain` CLI: this is the applications' half of the
//! workspace, reached through the SDK's `viewport` surface.
//!
//! Swedish Embedded AB builds the operator-facing views that make an embedded
//! or edge-AI system inspectable while it runs - on the desk and on a headless
//! target, from one implementation. If your team needs that, you can procure
//! our services by sending an email to info@swedishembedded.com.

pub mod canvas;
pub mod record;

pub use canvas::Canvas;
pub use record::Recorder;

use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;

/// Why a window could not be opened. Never fatal by itself: [`Viewport::open`]
/// falls back to a canvas with no window, because a run on a build server
/// should produce its artifacts rather than stop.
pub type OpenError = String;

/// A canvas, and a window onto it when this machine has one.
pub struct Viewport {
    canvas: Canvas,
    window: Option<SdlWindow>,
    title: String,
    recorder: Option<Recorder>,
    /// Why there is no window, if there isn't one. Reported once by the
    /// caller; a silent fallback to headless is how a run ends up with nobody
    /// noticing the display never came up.
    pub headless_because: Option<String>,
}

/// What the user did since the last pump.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub quit: bool,
    /// Space, as a play/pause toggle.
    pub pause: bool,
    /// Enter, as "start the next one".
    pub next: bool,
    /// `p`, as "save what I am looking at".
    pub screenshot: bool,
}

impl Viewport {
    /// Open a window of `width` x `height`, or fall back to a canvas alone.
    ///
    /// The fallback is the normal case on a server and is not an error: a
    /// training run that writes frames to disk wants the canvas and would be
    /// wrong to fail because `$DISPLAY` is unset.
    pub fn open(title: &str, width: u32, height: u32) -> Result<Viewport, OpenError> {
        let canvas = Canvas::new(width, height);
        // scale 1: the canvas is already at window resolution, because the
        // overlay is drawn at its own size rather than magnified with the
        // frame it sits on.
        let (window, why) = match SdlWindow::new(title, width, height, 1) {
            Ok(w) => (Some(w), None),
            Err(e) => (None, Some(e)),
        };
        Ok(Viewport {
            canvas,
            window,
            title: title.to_string(),
            recorder: None,
            headless_because: why,
        })
    }

    /// Open without even trying to find a display.
    pub fn headless(width: u32, height: u32) -> Viewport {
        Viewport {
            canvas: Canvas::new(width, height),
            window: None,
            title: String::new(),
            recorder: None,
            headless_because: Some("asked for a headless run".into()),
        }
    }

    pub fn has_window(&self) -> bool {
        self.window.is_some()
    }

    pub fn canvas(&mut self) -> &mut Canvas {
        &mut self.canvas
    }

    /// Record every presented frame into an MP4 at `path`.
    ///
    /// Encoded as it goes, through an `ffmpeg` pipe - see [`record`]. An
    /// absent ffmpeg is an error the caller can report and carry on from
    /// rather than a reason to stop.
    pub fn record(&mut self, path: impl AsRef<std::path::Path>, fps: u32) -> Result<(), String> {
        self.recorder =
            Some(Recorder::start(path, self.canvas.width(), self.canvas.height(), fps.max(1))?);
        Ok(())
    }

    /// Close the recording and return how many frames it holds and where.
    pub fn finish_recording(&mut self) -> Option<Result<(u64, std::path::PathBuf), String>> {
        self.recorder.take().map(|r| r.finish())
    }

    /// Put the canvas on screen, and into the recording if one is open.
    ///
    /// Recording happens HERE rather than at the caller so that what is
    /// recorded is exactly what was presented - the two cannot drift, and a
    /// headless run records the frames it would have shown.
    pub fn present(&mut self) {
        if let Some(r) = self.recorder.as_mut() {
            r.frame(self.canvas.pixels());
        }
        if let Some(w) = self.window.as_mut() {
            let hud = Hud { model: self.title.clone(), ..Hud::default() };
            w.frame(self.canvas.pixels(), self.canvas.width(), self.canvas.height(), &hud);
        }
    }

    /// Drain input. Always returns a default when there is no window, so a
    /// caller's loop reads the same either way.
    pub fn input(&mut self) -> Input {
        let Some(w) = self.window.as_mut() else {
            return Input::default();
        };
        let raw = w.pump();
        let mut out = Input { quit: raw.quit, ..Input::default() };
        for ux in raw.ux {
            match ux {
                wm_display::keymap::UxKey::Quit => out.quit = true,
                wm_display::keymap::UxKey::Pause => out.pause = true,
                wm_display::keymap::UxKey::Reset => out.next = true,
                wm_display::keymap::UxKey::Screenshot => out.screenshot = true,
                _ => {}
            }
        }
        out
    }

    /// Write the canvas to a PNG.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        self.canvas.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_headless_viewport_still_draws_and_saves() {
        // The property the whole crate exists for: no display, same pixels.
        let dir = std::env::temp_dir().join("brain-viewport-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("headless.png");

        let mut v = Viewport::headless(64, 32);
        v.canvas().clear([1, 2, 3]);
        v.canvas().text(2, 2, "OK", 1, [255, 255, 255]);
        assert!(!v.has_window());
        // A present with no window must be a no-op, not a panic: the same loop
        // runs on a desk and on a build server.
        v.present();
        assert_eq!(v.input().quit, false);
        v.save(&path).expect("saves a png");
        assert!(std::fs::metadata(&path).expect("written").len() > 0);
        let _ = std::fs::remove_file(&path);
    }
}
