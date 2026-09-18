// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Turning a single-channel field into pixels: colormaps, robust range
//! estimation, and a minimal in-frame HUD font.
//!
//! **Moved from `crates/zipdepth/src/viz.rs`**, which now re-exports this
//! module rather than defining it a second time - it started as ZipDepth's
//! depth-map colorizer, but nothing here is depth-specific (a colormap over
//! `[f32]`, robust bounds, an RGB8 compositor, a bitmap font) and every
//! consumer of a single-channel model output - a SAM 2 mask, a depth map, any
//! future heatmap - needs the same "make this viewable" step. Per this
//! crate's own charter (see `lib.rs`'s module docs): one home, not a second
//! copy behind a different crate's name.
//!
//! All pure and host-side - no GPU, no SDL - so the whole visualization path
//! is unit-testable without a window or a camera.

/// A perceptual colormap, as a 256-entry `[R,G,B]` lookup table.
///
/// Generated, never hand-typed: `Turbo` is Google's turbo approximation
/// (a smooth rainbow that, unlike jet, has monotone luminance so near/far read
/// correctly in greyscale too), `Gray` is the identity ramp for a falsifiable
/// baseline. The table is built once; `colorize` just indexes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Colormap {
    Turbo,
    Gray,
    /// Inverted grey - nearer is brighter, which some find more intuitive for
    /// inverse depth.
    GrayInv,
}

impl Colormap {
    /// Cycle to the next map, for the demo's `[`/`]` keys.
    pub fn next(self) -> Colormap {
        match self {
            Colormap::Turbo => Colormap::Gray,
            Colormap::Gray => Colormap::GrayInv,
            Colormap::GrayInv => Colormap::Turbo,
        }
    }

    pub fn lut(self) -> [[u8; 3]; 256] {
        let mut t = [[0u8; 3]; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let x = i as f32 / 255.0;
            *e = match self {
                Colormap::Turbo => turbo(x),
                Colormap::Gray => {
                    let v = (x * 255.0).round() as u8;
                    [v, v, v]
                }
                Colormap::GrayInv => {
                    let v = ((1.0 - x) * 255.0).round() as u8;
                    [v, v, v]
                }
            };
        }
        t
    }
}

/// Parse a colormap name (`turbo` / `gray` / `grayinv`, case-insensitive) -
/// the shared spelling every CLI flag exposing a `--colormap` should accept.
pub fn parse_colormap(s: &str) -> Result<Colormap, String> {
    match s.to_ascii_lowercase().as_str() {
        "turbo" => Ok(Colormap::Turbo),
        "gray" | "grey" => Ok(Colormap::Gray),
        "grayinv" | "greyinv" | "gray-inv" | "grey-inv" => Ok(Colormap::GrayInv),
        other => Err(format!("unknown colormap {other:?} (expected turbo, gray, or grayinv)")),
    }
}

/// A blue -> cyan -> green -> yellow -> red rainbow, linearly interpolated between
/// five anchors. Named `turbo` for the "warm = near" intent it shares with Google's
/// turbo, but defined by explicit anchors rather than turbo's polynomial: the
/// polynomial approximation is muddy at both ends (x=0 lands on a dark red-grey, not
/// blue), which reads wrong on a depth map. These anchors give a clean, monotone-hue
/// ramp with the endpoints the demo (and the tests) actually rely on.
fn turbo(x: f32) -> [u8; 3] {
    // (position, R, G, B).
    const A: [(f32, [f32; 3]); 5] = [
        (0.00, [0.0, 0.0, 130.0]),   // navy - far
        (0.25, [0.0, 200.0, 255.0]), // cyan
        (0.50, [0.0, 200.0, 0.0]),   // green
        (0.75, [255.0, 255.0, 0.0]), // yellow
        (1.00, [210.0, 0.0, 0.0]),   // red - near
    ];
    let x = x.clamp(0.0, 1.0);
    let mut hi = 1;
    while hi < A.len() - 1 && x > A[hi].0 {
        hi += 1;
    }
    let (x0, c0) = A[hi - 1];
    let (x1, c1) = A[hi];
    let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
    let mix = |a: f32, b: f32| chan((a + (b - a) * t) / 255.0);
    [mix(c0[0], c1[0]), mix(c0[1], c1[1]), mix(c0[2], c1[2])]
}
fn chan(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The `[lo, hi]` field window a frame is normalized against before coloring.
///
/// Deliberately NOT per-frame min/max: `max` is one pixel, and every pixel divides
/// by `(max-min)`, so a single specular highlight would swing the whole image's hue
/// at 30 Hz - a static scene appears to breathe. [`from_percentiles`] uses robust
/// percentiles instead, and a caller EMAs them across frames (`ema`, below) so the
/// mapping is stable frame to frame.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub lo: f32,
    pub hi: f32,
}

impl Bounds {
    /// Robust bounds from the `plo`/`phi` percentiles of a strided subsample.
    ///
    /// A stride keeps this ~O(4000) samples regardless of resolution (a 1e6 outlier
    /// then moves p98 by zero and min/max by everything), and `select_nth_unstable`
    /// finds each percentile in linear time without a full sort. NaNs are dropped.
    pub fn from_percentiles(depth: &[f32], plo: f32, phi: f32) -> Bounds {
        let stride = (depth.len() / 4096).max(1);
        let mut s: Vec<f32> = depth.iter().step_by(stride).copied().filter(|v| v.is_finite()).collect();
        if s.is_empty() {
            return Bounds { lo: 0.0, hi: 1.0 };
        }
        let pick = |s: &mut [f32], p: f32| -> f32 {
            let k = ((p.clamp(0.0, 1.0) * (s.len() - 1) as f32).round() as usize).min(s.len() - 1);
            s.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1.to_owned()
        };
        let lo = pick(&mut s, plo);
        let hi = pick(&mut s, phi);
        // Never a degenerate window (all-equal frame, or hi<=lo from ties).
        if hi - lo < 1e-6 {
            Bounds { lo, hi: lo + 1e-6 }
        } else {
            Bounds { lo, hi }
        }
    }

    /// Map one value to `[0,1]`, clamped.
    pub fn norm(&self, v: f32) -> f32 {
        ((v - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)
    }

    /// EMA toward `target` (a demo loop smooths bounds frame-to-frame, α≈0.1); a
    /// scene cut should `snap` instead so the map does not crawl for a second.
    pub fn ema(self, target: Bounds, alpha: f32) -> Bounds {
        Bounds {
            lo: self.lo + alpha * (target.lo - self.lo),
            hi: self.hi + alpha * (target.hi - self.hi),
        }
    }
}

/// Colorize a `[H*W]` single-channel field into row-major `[H*W*3]` RGB8 via
/// `bounds` + `map`.
pub fn colorize(depth: &[f32], bounds: Bounds, map: Colormap) -> Vec<u8> {
    let lut = map.lut();
    let mut out = Vec::with_capacity(depth.len() * 3);
    for &v in depth {
        let idx = (bounds.norm(v) * 255.0).round() as usize;
        out.extend_from_slice(&lut[idx.min(255)]);
    }
    out
}

/// The side-by-side canvas a demo shows: one image on the left, a colorized
/// field on the right, at a shared height. Both inputs are row-major RGB8 at
/// their own size; the output is `[H * (Wl+Wr) * 3]`.
///
/// Self-evidencing on purpose - wave a hand and the right half goes red at the
/// same instant, which the colorized field alone (unfalsifiable) cannot show.
pub fn composite_side_by_side(
    left: &[u8],
    lw: u32,
    lh: u32,
    right: &[u8],
    rw: u32,
    rh: u32,
) -> (Vec<u8>, u32, u32) {
    assert_eq!(lh, rh, "side-by-side needs a shared height ({lh} vs {rh})");
    let h = lh;
    let w = lw + rw;
    let mut out = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        let orow = (y * w * 3) as usize;
        let lrow = (y * lw * 3) as usize;
        out[orow..orow + (lw * 3) as usize].copy_from_slice(&left[lrow..lrow + (lw * 3) as usize]);
        let rrow = (y * rw * 3) as usize;
        let off = orow + (lw * 3) as usize;
        out[off..off + (rw * 3) as usize].copy_from_slice(&right[rrow..rrow + (rw * 3) as usize]);
    }
    (out, w, h)
}

// ---------------------------------------------------------------------------
// Minimal in-frame HUD text (5x7 bitmap font), so fps/latency/labels show ON
// the image rather than only in a window title.
// ---------------------------------------------------------------------------

/// 5x7 glyphs for the characters the HUD uses, row-major: 7 rows, low 5 bits each
/// (bit 4 = leftmost column). Uppercase only - the HUD text is upcased before draw.
fn glyph(c: u8) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        b'0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        b'1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        b'3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        b'4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        b'5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        b'6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        b'7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        b'8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        b'9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        b'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        b'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        b'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        b'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        b'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        b'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        b'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        b'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        b'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        b'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        b'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        b'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        b'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        b'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        b'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        b'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        b'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        b'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        b'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        b'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        b'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        b'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        b'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        b':' => [0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00],
        b'.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
        b'/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        b'-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        _ => [0; 7], // space and anything unknown
    }
}

/// Draw `text` into a row-major RGB8 buffer at `(x0, y0)`, scaled `px` times, in
/// `color`, over a translucent dark box for legibility on any background. Used for
/// the camera HUD (fps / latency / drops) so it reads directly on the frame.
pub fn draw_text(rgb: &mut [u8], w: u32, h: u32, x0: u32, y0: u32, text: &str, px: u32, color: [u8; 3]) {
    let cw = 6 * px; // 5 glyph cols + 1 space
    // Dark backing box, alpha-ish (halve the underlying pixels), for contrast.
    let bw = cw * text.len() as u32 + 2 * px;
    let bh = 7 * px + 2 * px;
    for yy in y0.saturating_sub(px)..(y0 + bh).min(h) {
        for xx in x0.saturating_sub(px)..(x0 + bw).min(w) {
            let o = ((yy * w + xx) * 3) as usize;
            if o + 2 < rgb.len() {
                rgb[o] /= 3;
                rgb[o + 1] /= 3;
                rgb[o + 2] /= 3;
            }
        }
    }
    for (ci, ch) in text.bytes().enumerate() {
        let g = glyph(ch);
        let gx = x0 + ci as u32 * cw;
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5u32 {
                if bits & (1 << (4 - col)) != 0 {
                    // filled px x px block
                    for dy in 0..px {
                        for dx in 0..px {
                            let xx = gx + col * px + dx;
                            let yy = y0 + row as u32 * px + dy;
                            if xx < w && yy < h {
                                let o = ((yy * w + xx) * 3) as usize;
                                rgb[o] = color[0];
                                rgb[o + 1] = color[1];
                                rgb[o + 2] = color[2];
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Panels, bars and plots
//
//  Enough of an immediate-mode drawing surface to build an instrument panel
//  around a model's output: filled and outlined boxes, labelled bars, and a
//  line plot with its own axis range.
//
//  Here rather than in the window crate on purpose. These are pure functions
//  over an RGB8 buffer, so a headless run draws EXACTLY the panel a human sees
//  and can write it to a PNG - which is what makes a screenshot in a README a
//  measurement rather than an illustration, and what lets the whole overlay be
//  unit-tested with no display attached.
// ---------------------------------------------------------------------------

/// Clamp a rect to the buffer and run `f(x, y, offset)` over every pixel in it.
fn for_rect(w: u32, h: u32, x: i32, y: i32, rw: u32, rh: u32, mut f: impl FnMut(u32, u32, usize)) {
    let x0 = x.max(0) as u32;
    let y0 = y.max(0) as u32;
    let x1 = ((x + rw as i32).max(0) as u32).min(w);
    let y1 = ((y + rh as i32).max(0) as u32).min(h);
    for yy in y0..y1 {
        for xx in x0..x1 {
            f(xx, yy, ((yy * w + xx) * 3) as usize);
        }
    }
}

/// Fill a rectangle with a solid colour.
pub fn fill_rect(rgb: &mut [u8], w: u32, h: u32, x: i32, y: i32, rw: u32, rh: u32, c: [u8; 3]) {
    for_rect(w, h, x, y, rw, rh, |_, _, o| {
        if o + 2 < rgb.len() {
            rgb[o] = c[0];
            rgb[o + 1] = c[1];
            rgb[o + 2] = c[2];
        }
    });
}

/// Blend a colour over a rectangle. `alpha` is 0..=255, 255 being opaque.
///
/// Panels sit ON the frame rather than beside it, so the game stays visible
/// underneath a readout instead of being cropped by it.
pub fn blend_rect(
    rgb: &mut [u8],
    w: u32,
    h: u32,
    x: i32,
    y: i32,
    rw: u32,
    rh: u32,
    c: [u8; 3],
    alpha: u8,
) {
    let a = alpha as u32;
    let ia = 255 - a;
    for_rect(w, h, x, y, rw, rh, |_, _, o| {
        if o + 2 < rgb.len() {
            for k in 0..3 {
                rgb[o + k] = ((rgb[o + k] as u32 * ia + c[k] as u32 * a) / 255) as u8;
            }
        }
    });
}

/// Outline a rectangle, one pixel wide.
pub fn stroke_rect(rgb: &mut [u8], w: u32, h: u32, x: i32, y: i32, rw: u32, rh: u32, c: [u8; 3]) {
    fill_rect(rgb, w, h, x, y, rw, 1, c);
    fill_rect(rgb, w, h, x, y + rh as i32 - 1, rw, 1, c);
    fill_rect(rgb, w, h, x, y, 1, rh, c);
    fill_rect(rgb, w, h, x + rw as i32 - 1, y, 1, rh, c);
}

/// A horizontal bar filled to `frac` of its width, on a darker track.
///
/// `frac` is clamped rather than asserted: these display live model output
/// (a probability that summed to 1.0 in f32, a health fraction during a frame
/// where the value changed), and a panic in the drawing code is a far worse
/// outcome than a bar pinned at one end.
pub fn bar(
    rgb: &mut [u8],
    w: u32,
    h: u32,
    x: i32,
    y: i32,
    rw: u32,
    rh: u32,
    frac: f32,
    fg: [u8; 3],
    track: [u8; 3],
) {
    fill_rect(rgb, w, h, x, y, rw, rh, track);
    let f = frac.clamp(0.0, 1.0);
    let filled = (rw as f32 * f).round() as u32;
    if filled > 0 {
        fill_rect(rgb, w, h, x, y, filled, rh, fg);
    }
}

/// A line plot of `series` inside the given rect, auto-scaled to its own range.
///
/// Draws a zero line when the range spans it, because for a reward trace "above
/// or below zero" is the first thing a reader wants and a plot that only shows
/// shape cannot answer it.
pub fn plot(
    rgb: &mut [u8],
    w: u32,
    h: u32,
    x: i32,
    y: i32,
    rw: u32,
    rh: u32,
    series: &[f32],
    c: [u8; 3],
) {
    if rw < 2 || rh < 2 || series.is_empty() {
        return;
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for v in series {
        if v.is_finite() {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return;
    }
    if (hi - lo).abs() < 1e-9 {
        lo -= 0.5;
        hi += 0.5;
    }
    let to_y = |v: f32| -> i32 {
        let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
        y + rh as i32 - 1 - (t * (rh - 1) as f32).round() as i32
    };

    if lo < 0.0 && hi > 0.0 {
        fill_rect(rgb, w, h, x, to_y(0.0), rw, 1, [70, 70, 80]);
    }

    // One column per pixel: with more samples than columns each column shows
    // the mean of its bucket, so a long run stays readable instead of aliasing
    // down to whichever sample happened to land on a pixel.
    let cols = rw as usize;
    let mut prev: Option<i32> = None;
    for col in 0..cols {
        let a = series.len() * col / cols;
        let b = (series.len() * (col + 1) / cols).max(a + 1).min(series.len());
        if a >= series.len() {
            break;
        }
        let bucket = &series[a..b];
        let mean = bucket.iter().copied().filter(|v| v.is_finite()).sum::<f32>()
            / bucket.len().max(1) as f32;
        let yy = to_y(mean);
        let xx = x + col as i32;
        match prev {
            // Join to the previous column so a steep change reads as a line
            // rather than as two disconnected dots.
            Some(py) => {
                let (from, to) = if py <= yy { (py, yy) } else { (yy, py) };
                fill_rect(rgb, w, h, xx, from, 1, (to - from + 1) as u32, c);
            }
            None => fill_rect(rgb, w, h, xx, yy, 1, 1, c),
        }
        prev = Some(yy);
    }
}

/// Expand an indexed image through a 256-entry RGB palette, scaled `scale`x,
/// into an RGB8 buffer at `(x, y)`.
///
/// The palette arrives with the pixels rather than being assumed: an engine
/// that tints its own output (Doom's damage flash, a radiation suit) changes
/// the palette and not the indices, so a fixed table would show the wrong
/// picture at exactly the moments worth looking at.
pub fn blit_indexed(
    rgb: &mut [u8],
    w: u32,
    h: u32,
    x: i32,
    y: i32,
    src: &[u8],
    sw: u32,
    sh: u32,
    palette: &[u8],
    scale: u32,
) {
    if palette.len() < 768 || scale == 0 {
        return;
    }
    for sy in 0..sh {
        for sx in 0..sw {
            let Some(&idx) = src.get((sy * sw + sx) as usize) else {
                return;
            };
            let p = idx as usize * 3;
            let c = [palette[p], palette[p + 1], palette[p + 2]];
            fill_rect(
                rgb,
                w,
                h,
                x + (sx * scale) as i32,
                y + (sy * scale) as i32,
                scale,
                scale,
                c,
            );
        }
    }
}

#[cfg(test)]
mod panel_tests {
    use super::*;

    fn px(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 3] {
        let o = ((y * w + x) * 3) as usize;
        [buf[o], buf[o + 1], buf[o + 2]]
    }

    #[test]
    fn drawing_outside_the_buffer_clips_instead_of_panicking() {
        // Every one of these is a real call shape: a panel anchored to the
        // right edge, a bar whose value overflowed, a plot in a window the
        // user made smaller than the layout assumed.
        let (w, h) = (16u32, 8u32);
        let mut buf = vec![0u8; (w * h * 3) as usize];
        fill_rect(&mut buf, w, h, -4, -4, 8, 8, [255, 0, 0]);
        fill_rect(&mut buf, w, h, 12, 4, 100, 100, [0, 255, 0]);
        stroke_rect(&mut buf, w, h, -2, 6, 40, 40, [0, 0, 255]);
        blend_rect(&mut buf, w, h, -5, -5, 100, 100, [0, 0, 0], 128);
        plot(&mut buf, w, h, 0, 0, 40, 40, &[1.0, -1.0, f32::NAN], [255, 255, 0]);
        // Reaching here at all is the assertion: every call above indexes
        // outside the buffer somewhere. The value check belongs to the tests
        // below, which draw inside it.
        assert_eq!(buf.len(), (w * h * 3) as usize, "nothing resized the buffer");
    }

    #[test]
    fn a_bar_shows_the_fraction_it_was_given() {
        let (w, h) = (10u32, 1u32);
        let mut buf = vec![0u8; (w * h * 3) as usize];
        bar(&mut buf, w, h, 0, 0, 10, 1, 0.3, [255, 255, 255], [0, 0, 0]);
        assert_eq!(px(&buf, w, 2, 0), [255, 255, 255]);
        assert_eq!(px(&buf, w, 3, 0), [0, 0, 0]);
    }

    #[test]
    fn an_indexed_blit_uses_the_palette_it_was_handed() {
        // The tint case: the same indices through a different palette must
        // produce a different picture, which is the whole reason the palette
        // travels with the frame.
        let (w, h) = (4u32, 4u32);
        let mut buf = vec![0u8; (w * h * 3) as usize];
        let src = [1u8, 1, 1, 1];
        let mut pal = vec![0u8; 768];
        pal[3..6].copy_from_slice(&[10, 20, 30]);
        blit_indexed(&mut buf, w, h, 0, 0, &src, 2, 2, &pal, 2);
        assert_eq!(px(&buf, w, 3, 3), [10, 20, 30]);

        pal[3..6].copy_from_slice(&[200, 0, 0]);
        blit_indexed(&mut buf, w, h, 0, 0, &src, 2, 2, &pal, 2);
        assert_eq!(px(&buf, w, 3, 3), [200, 0, 0]);
    }
}
