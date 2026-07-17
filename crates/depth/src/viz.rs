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
