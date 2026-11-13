// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The depth window both workflows draw into: `v` cycles the view (the
//! window resizes to the view's natural aspect), `[` and `]` the colormap,
//! Esc quits. Every frame is drawn by `zipdepth::view::render`, the renderer
//! the served `infer` action's `view` output uses.
//!
//! Swedish Embedded AB implements live visualization of sensor and model
//! output on embedded displays for its clients. If your team needs expertise
//! in real-time camera pipelines then you can procure our services by
//! sending an email to info@swedishembedded.com.

use imaging::viz::{Bounds, Colormap};
use wm_display::keymap::UxKey;
use wm_display::sink::{FrameSink, Hud};
use wm_display::window::SdlWindow;
use zipdepth::view::{render, Style, View};
use zipdepth::StereoOpts;

pub struct DepthWindow {
    title: String,
    scale: u32,
    pub view: View,
    pub colormap: Colormap,
    stripes: u32,
    window: Option<(SdlWindow, u32, u32)>,
    /// Frames drawn, and how many to draw before closing (scripted runs).
    frames: u64,
    max_frames: Option<u64>,
}

impl DepthWindow {
    pub fn new(title: &str, scale: u32, view: View, colormap: Colormap, stripes: u32, max_frames: Option<u64>) -> DepthWindow {
        DepthWindow { title: title.to_string(), scale, view, colormap, stripes, window: None, frames: 0, max_frames }
    }

    /// Read the keys since the last frame. False once the window should close.
    pub fn keys(&mut self) -> bool {
        let Some((window, _, _)) = self.window.as_mut() else { return true };
        let input = window.pump();
        for key in &input.ux {
            match key {
                UxKey::QualityUp | UxKey::QualityDown => self.colormap = self.colormap.next(),
                UxKey::CycleView => self.view = self.view.cycle(),
                _ => {}
            }
        }
        !input.quit && self.max_frames.is_none_or(|max| self.frames < max)
    }

    /// Draw `depth` of the `w x h` frame `rgb8` in the current view, with
    /// `status` written onto it. Opens (or resizes) the window to the view.
    pub fn draw(&mut self, rgb8: &[u8], depth: &[f32], w: u32, h: u32, bounds: Bounds, status: &str, fps: f32) -> Result<(), String> {
        let (cw, ch) = self.view.canvas(w, h);
        if self.window.as_ref().is_none_or(|&(_, ww, wh)| (ww, wh) != (cw, ch)) {
            let window = SdlWindow::new(&self.title, cw, ch, self.scale).map_err(|e| format!("no display ({e})"))?;
            self.window = Some((window, cw, ch));
        }
        let style = Style { bounds, colormap: self.colormap, stereo: StereoOpts::with_stripes(w, self.stripes) };
        let mut canvas = render(rgb8, depth, w, h, self.view, &style);
        let line = format!("{} {status}", self.view.name().to_uppercase());
        imaging::viz::draw_text(&mut canvas, cw, ch, 6, 6, &line, 2, [0, 255, 0]);
        let hud = Hud { model: format!("zipdepth {line}"), fps, quality: self.colormap as u32, ..Default::default() };
        let (window, _, _) = self.window.as_mut().expect("opened above");
        window.frame(&canvas, cw, ch, &hud);
        self.frames += 1;
        Ok(())
    }
}
