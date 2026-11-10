// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Image preparation: letterbox sizing, resampling and the 512 canvas.
//!
//! Each side is scaled by `min(512/w, 512/h)` and then rounded to a multiple
//! of 32, up unless that overshoots the exact size by more than 8 percent, in
//! which case down (never below 32). A shrink uses exact area averaging; a
//! grow uses half-pixel bilinear sampling. Results round half up. The resized
//! image sits at the top-left of a zero canvas and pixels map to `x / 127.5 - 1`.

use std::fmt;

/// Side of the square canvas.
pub const CANVAS: usize = 512;

/// Largest image side accepted.
pub const MAX_SIDE: usize = 1 << 14;

/// An input the preprocessor refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageError(pub String);

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mirage image: {}", self.0)
    }
}

impl std::error::Error for ImageError {}

/// A resized RGB image, row-major `[h, w, 3]` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resized {
    /// Pixels.
    pub rgb: Vec<u8>,
    /// Height.
    pub h: usize,
    /// Width.
    pub w: usize,
}

fn check(rgb: &[u8], h: usize, w: usize) -> Result<(), ImageError> {
    if h == 0 || w == 0 || h > MAX_SIDE || w > MAX_SIDE {
        return Err(ImageError(format!("{w}x{h} is outside 1..={MAX_SIDE} on each side")));
    }
    if rgb.len() != h * w * 3 {
        return Err(ImageError(format!("{} bytes is not a {w}x{h} RGB image ({})", rgb.len(), h * w * 3)));
    }
    Ok(())
}

/// The letterboxed `(height, width)` for an input of `h x w`.
pub fn letterbox_size(h: usize, w: usize) -> (usize, usize) {
    let scale = (CANVAS as f64 / w as f64).min(CANVAS as f64 / h as f64);
    let side = |dim: usize| {
        let t = scale * dim as f64;
        let up = (((t / 32.0).ceil() as usize) * 32).min(CANVAS);
        if up < 1 || up as f64 / t.max(1e-6) > 1.08 {
            (((t / 32.0).floor() as usize) * 32).max(32)
        } else {
            up
        }
    };
    (side(h), side(w))
}

/// Per-output-sample `(first source index, weights)` for an axis.
type Taps = Vec<(usize, Vec<f64>)>;

fn area_taps(n_out: usize, n_in: usize) -> Taps {
    let scale = n_in as f64 / n_out as f64;
    (0..n_out)
        .map(|o| {
            let (a, b) = (o as f64 * scale, (o + 1) as f64 * scale);
            let first = a.floor() as usize;
            let last = (b.ceil() as usize).min(n_in);
            let mut w: Vec<f64> = (first..last).map(|i| (b.min((i + 1) as f64) - a.max(i as f64)).max(0.0)).collect();
            let total: f64 = w.iter().sum();
            w.iter_mut().for_each(|v| *v /= total);
            (first, w)
        })
        .collect()
}

fn bilinear_taps(n_out: usize, n_in: usize) -> Taps {
    let scale = n_in as f64 / n_out as f64;
    (0..n_out)
        .map(|o| {
            let x = ((o as f64 + 0.5) * scale - 0.5).clamp(0.0, (n_in - 1) as f64);
            let i0 = x.floor() as usize;
            let i1 = (i0 + 1).min(n_in - 1);
            let f = x - i0 as f64;
            if i1 == i0 {
                (i0, vec![1.0])
            } else {
                (i0, vec![1.0 - f, f])
            }
        })
        .collect()
}

/// Resample `[h, w, 3]` to `[oh, ow, 3]` with the given per-axis taps: rows
/// first, then columns, accumulated in `f64` and rounded half up.
fn resample(rgb: &[u8], h: usize, w: usize, oh: usize, ow: usize, ty: &Taps, tx: &Taps) -> Vec<u8> {
    let mut rows = vec![0f64; oh * w * 3];
    for (o, (first, weights)) in ty.iter().enumerate() {
        for (k, &wt) in weights.iter().enumerate() {
            let src = &rgb[(first + k).min(h - 1) * w * 3..][..w * 3];
            rows[o * w * 3..(o + 1) * w * 3].iter_mut().zip(src).for_each(|(a, &s)| *a += wt * f64::from(s));
        }
    }
    let mut out = vec![0u8; oh * ow * 3];
    for y in 0..oh {
        for (x, (first, weights)) in tx.iter().enumerate() {
            for c in 0..3 {
                let v: f64 = weights.iter().enumerate().map(|(k, &wt)| wt * rows[(y * w + first + k) * 3 + c]).sum();
                out[(y * ow + x) * 3 + c] = (v + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Resize to `oh x ow`: area averaging when either side shrinks, bilinear otherwise.
pub fn resize(rgb: &[u8], h: usize, w: usize, oh: usize, ow: usize) -> Result<Resized, ImageError> {
    check(rgb, h, w)?;
    if oh == 0 || ow == 0 || oh > MAX_SIDE || ow > MAX_SIDE {
        return Err(ImageError(format!("target {ow}x{oh} is outside 1..={MAX_SIDE}")));
    }
    let shrink = ow < w || oh < h;
    let (ty, tx) = if shrink { (area_taps(oh, h), area_taps(ow, w)) } else { (bilinear_taps(oh, h), bilinear_taps(ow, w)) };
    Ok(Resized { rgb: resample(rgb, h, w, oh, ow, &ty, &tx), h: oh, w: ow })
}

/// Bilinear resize regardless of direction (used for reader strips).
pub fn resize_bilinear(rgb: &[u8], h: usize, w: usize, oh: usize, ow: usize) -> Result<Resized, ImageError> {
    check(rgb, h, w)?;
    if oh == 0 || ow == 0 || oh > MAX_SIDE || ow > MAX_SIDE {
        return Err(ImageError(format!("target {ow}x{oh} is outside 1..={MAX_SIDE}")));
    }
    Ok(Resized { rgb: resample(rgb, h, w, oh, ow, &bilinear_taps(oh, h), &bilinear_taps(ow, w)), h: oh, w: ow })
}

/// Letterbox an `h x w` RGB image.
pub fn letterbox(rgb: &[u8], h: usize, w: usize) -> Result<Resized, ImageError> {
    check(rgb, h, w)?;
    let (oh, ow) = letterbox_size(h, w);
    resize(rgb, h, w, oh, ow)
}

/// The `512 x 512 x 3` network input: the image at the top-left, `x / 127.5 - 1`,
/// zero elsewhere.
pub fn canvas(img: &Resized) -> Vec<f32> {
    let mut out = vec![0f32; CANVAS * CANVAS * 3];
    for y in 0..img.h.min(CANVAS) {
        for x in 0..img.w.min(CANVAS) {
            for c in 0..3 {
                out[(y * CANVAS + x) * 3 + c] = f32::from(img.rgb[(y * img.w + x) * 3 + c]) / 127.5 - 1.0;
            }
        }
    }
    out
}
