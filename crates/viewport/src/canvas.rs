// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! An RGB8 buffer with the drawing a live instrument panel needs.
//!
//! Every method forwards to `imaging::viz`, which is where this workspace
//! keeps its drawing primitives - the font, the colormaps, the compositor.
//! What is added here is only the shape: a buffer that remembers its own size,
//! so a caller writes `c.text(x, y, ..)` instead of threading `(&mut buf, w, h)`
//! through forty call sites and getting one of them wrong.

use imaging::viz;

/// A row-major RGB8 image being drawn into.
pub struct Canvas {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Canvas {
        Canvas { pixels: vec![0u8; (width * height * 3) as usize], width, height }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn clear(&mut self, c: [u8; 3]) {
        for px in self.pixels.chunks_exact_mut(3) {
            px.copy_from_slice(&c);
        }
    }

    pub fn fill(&mut self, x: i32, y: i32, w: u32, h: u32, c: [u8; 3]) {
        viz::fill_rect(&mut self.pixels, self.width, self.height, x, y, w, h, c);
    }

    /// Fill, but letting what is underneath show through. `alpha` 0..=255.
    pub fn shade(&mut self, x: i32, y: i32, w: u32, h: u32, c: [u8; 3], alpha: u8) {
        viz::blend_rect(&mut self.pixels, self.width, self.height, x, y, w, h, c, alpha);
    }

    pub fn outline(&mut self, x: i32, y: i32, w: u32, h: u32, c: [u8; 3]) {
        viz::stroke_rect(&mut self.pixels, self.width, self.height, x, y, w, h, c);
    }

    /// Text at `px` times the 5x7 font's own size.
    pub fn text(&mut self, x: i32, y: i32, s: &str, px: u32, c: [u8; 3]) {
        if x < 0 || y < 0 {
            return;
        }
        viz::draw_text(
            &mut self.pixels,
            self.width,
            self.height,
            x as u32,
            y as u32,
            s,
            px,
            c,
        );
    }

    /// Width in pixels that [`Canvas::text`] will occupy, so a caller can lay
    /// out against it rather than guessing and overlapping.
    pub fn text_width(s: &str, px: u32) -> u32 {
        6 * px * s.chars().count() as u32
    }

    /// Height of one line of text at scale `px`, including the backing box.
    pub const fn line_height(px: u32) -> u32 {
        9 * px
    }

    pub fn bar(&mut self, x: i32, y: i32, w: u32, h: u32, frac: f32, fg: [u8; 3], track: [u8; 3]) {
        viz::bar(&mut self.pixels, self.width, self.height, x, y, w, h, frac, fg, track);
    }

    pub fn plot(&mut self, x: i32, y: i32, w: u32, h: u32, series: &[f32], c: [u8; 3]) {
        viz::plot(&mut self.pixels, self.width, self.height, x, y, w, h, series, c);
    }

    /// Draw an indexed image through its palette, scaled up by whole pixels.
    pub fn blit_indexed(
        &mut self,
        x: i32,
        y: i32,
        src: &[u8],
        sw: u32,
        sh: u32,
        palette: &[u8],
        scale: u32,
    ) {
        viz::blit_indexed(
            &mut self.pixels,
            self.width,
            self.height,
            x,
            y,
            src,
            sw,
            sh,
            palette,
            scale,
        );
    }

    /// The largest whole-pixel scale at which `sw x sh` fits in `w x h`.
    ///
    /// Whole pixels because these are low-resolution frames from a 1993
    /// renderer: a fractional scale resamples them into mush, and a reader
    /// looking at a game frame to see what the agent saw needs the pixels.
    pub fn fit_scale(sw: u32, sh: u32, w: u32, h: u32) -> u32 {
        if sw == 0 || sh == 0 {
            return 1;
        }
        (w / sw).min(h / sh).max(1)
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        if let Some(dir) = path.as_ref().parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            }
        }
        let img = imaging::Rgb8::new(self.width, self.height, self.pixels.clone())
            .map_err(|e| format!("{e}"))?;
        imaging::codec::save_png(path, &img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_helpers_agree_with_what_gets_drawn() {
        // A panel is laid out against text_width/line_height; if they drift
        // from the font the labels overlap, which is the kind of thing nobody
        // notices in a screenshot until it is in a README.
        let mut c = Canvas::new(80, 20);
        c.clear([0, 0, 0]);
        let s = "AB";
        c.text(0, 0, s, 1, [255, 255, 255]);
        let w = Canvas::text_width(s, 1);
        // Nothing of the glyphs may land at or past the reported width.
        for y in 0..Canvas::line_height(1).min(20) {
            for x in w..80 {
                let o = ((y * 80 + x) * 3) as usize;
                assert_eq!(c.pixels()[o], 0, "text spilled past text_width at {x},{y}");
            }
        }
    }

    #[test]
    fn fit_scale_never_returns_zero() {
        // A window smaller than the frame still has to draw something; a zero
        // scale silently draws nothing at all.
        assert_eq!(Canvas::fit_scale(320, 200, 960, 600), 3);
        assert_eq!(Canvas::fit_scale(320, 200, 100, 100), 1);
        assert_eq!(Canvas::fit_scale(0, 0, 100, 100), 1);
    }
}
