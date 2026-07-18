// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Turning a depth map into pixels: colormaps, robust range estimation, and the
//! side-by-side composite the demo shows.
//!
//! All pure and host-side — no GPU, no SDL — so the whole visualization path is
//! unit-testable without a window or a camera. The demo's display layer only has to
//! hand these bytes to a texture.

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
    /// Inverted grey — nearer is brighter, which some find more intuitive for
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

/// A blue -> cyan -> green -> yellow -> red rainbow, linearly interpolated between
/// five anchors. Named `turbo` for the "warm = near" intent it shares with Google's
/// turbo, but defined by explicit anchors rather than turbo's polynomial: the
/// polynomial approximation is muddy at both ends (x=0 lands on a dark red-grey, not
/// blue), which reads wrong on a depth map. These anchors give a clean, monotone-hue
/// ramp with the endpoints the demo (and the tests) actually rely on.
fn turbo(x: f32) -> [u8; 3] {
    // (position, R, G, B).
    const A: [(f32, [f32; 3]); 5] = [
        (0.00, [0.0, 0.0, 130.0]),   // navy — far
        (0.25, [0.0, 200.0, 255.0]), // cyan
        (0.50, [0.0, 200.0, 0.0]),   // green
        (0.75, [255.0, 255.0, 0.0]), // yellow
        (1.00, [210.0, 0.0, 0.0]),   // red — near
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

/// The `[lo, hi]` depth window a frame is normalized against before coloring.
///
/// Deliberately NOT per-frame min/max: `max` is one pixel, and every pixel divides
/// by `(max-min)`, so a single specular highlight would swing the whole image's hue
/// at 30 Hz — a static scene appears to breathe. [`from_percentiles`] uses robust
/// percentiles instead, and the demo EMAs them across frames (see the loop, not
/// here) so the mapping is stable.
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

    /// Map one depth value to `[0,1]`, clamped.
    pub fn norm(&self, v: f32) -> f32 {
        ((v - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)
    }

    /// EMA toward `target` (the demo smooths bounds frame-to-frame, α≈0.1); a scene
    /// cut should `snap` instead so the map does not crawl for a second.
    pub fn ema(self, target: Bounds, alpha: f32) -> Bounds {
        Bounds {
            lo: self.lo + alpha * (target.lo - self.lo),
            hi: self.hi + alpha * (target.hi - self.hi),
        }
    }
}

/// Colorize a `[H*W]` depth map into row-major `[H*W*3]` RGB8 via `bounds` + `map`.
pub fn colorize(depth: &[f32], bounds: Bounds, map: Colormap) -> Vec<u8> {
    let lut = map.lut();
    let mut out = Vec::with_capacity(depth.len() * 3);
    for &v in depth {
        let idx = (bounds.norm(v) * 255.0).round() as usize;
        out.extend_from_slice(&lut[idx.min(255)]);
    }
    out
}

/// The side-by-side canvas the demo shows: RGB on the left, colorized depth on the
/// right, at a shared height. Both inputs are row-major RGB8 at their own size; the
/// output is `[H * (Wl+Wr) * 3]`.
///
/// Self-evidencing on purpose — wave a hand and the depth half goes red at the same
/// instant, which a depth map alone (unfalsifiable) cannot show.
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
// Minimal in-frame HUD text (5x7 bitmap font), so fps/latency show ON the image
// rather than only in the window title.
// ---------------------------------------------------------------------------

/// 5x7 glyphs for the characters the HUD uses, row-major: 7 rows, low 5 bits each
/// (bit 4 = leftmost column). Uppercase only — the HUD text is upcased before draw.
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
