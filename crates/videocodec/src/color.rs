// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RGB <-> Y'CbCr 4:2:0, the planes an AV1 stream codes.
//!
//! Written frames are converted with the BT.709 matrix at limited ("studio")
//! range, and the stream says so in its sequence header and container, so
//! every player reverses it the same way. Read frames are converted with the
//! matrix and range their own stream declares; a stream that declares none
//! is taken as BT.709 at 720 lines and above and BT.601 below, the usual
//! reading of an unlabelled HD/SD stream.
//!
//! Chroma is the mean of each 2x2 block (an odd last row or column averages
//! what it has), and read back by nearest sample.

/// The Y'CbCr matrix of a stream: the luma weights of red and blue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix {
    kr: f32,
    kb: f32,
}

impl Matrix {
    pub const BT709: Matrix = Matrix { kr: 0.2126, kb: 0.0722 };
    pub const BT601: Matrix = Matrix { kr: 0.299, kb: 0.114 };
    pub const BT2020: Matrix = Matrix { kr: 0.2627, kb: 0.0593 };
}

/// One 4:2:0 picture, 8 bits per sample: `y` is `w * h`, `u` and `v` are
/// `((w + 1) / 2) * ((h + 1) / 2)`, all tightly packed.
#[derive(Clone, Debug, PartialEq)]
pub struct Yuv420 {
    pub w: usize,
    pub h: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

impl Yuv420 {
    pub fn chroma_size(w: usize, h: usize) -> (usize, usize) {
        (w.div_ceil(2), h.div_ceil(2))
    }
}

/// `rgb` (`w * h * 3` bytes, R,G,B interleaved) as BT.709 limited-range 4:2:0.
pub fn rgb_to_yuv420(rgb: &[u8], w: usize, h: usize) -> Yuv420 {
    assert_eq!(rgb.len(), w * h * 3, "videocodec: an RGB frame of {w}x{h} is {} bytes, got {}", w * h * 3, rgb.len());
    let Matrix { kr, kb } = Matrix::BT709;
    let kg = 1.0 - kr - kb;
    let (cw, ch) = Yuv420::chroma_size(w, h);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    // Per pixel, in units of full scale: luma, and the two colour differences scaled to [-0.5, 0.5].
    let luma = |r: f32, g: f32, b: f32| kr * r + kg * g + kb * b;
    for (row, out) in y.chunks_exact_mut(w).enumerate() {
        for (col, o) in out.iter_mut().enumerate() {
            let p = &rgb[(row * w + col) * 3..][..3];
            let l = luma(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0;
            *o = quantize(16.0 + 219.0 * l);
        }
    }
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut cb, mut cr, mut n) = (0.0f32, 0.0f32, 0.0f32);
            for row in (cy * 2)..(cy * 2 + 2).min(h) {
                for col in (cx * 2)..(cx * 2 + 2).min(w) {
                    let p = &rgb[(row * w + col) * 3..][..3];
                    let (r, g, b) = (p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
                    let l = luma(r, g, b);
                    cb += (b - l) / (2.0 * (1.0 - kb));
                    cr += (r - l) / (2.0 * (1.0 - kr));
                    n += 1.0;
                }
            }
            u[cy * cw + cx] = quantize(128.0 + 224.0 * cb / n);
            v[cy * cw + cx] = quantize(128.0 + 224.0 * cr / n);
        }
    }
    Yuv420 { w, h, y, u, v }
}

/// The colour a stream declares, for [`yuv_to_rgb`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Colour {
    pub matrix: Matrix,
    /// Full range (0..=max) rather than limited (16..=235 luma, 16..=240 chroma at 8 bits).
    pub full_range: bool,
}

/// One plane's samples, widened to `u16`, and its row stride in samples.
pub type Plane<'a> = (&'a [u16], usize);

/// One decoded picture's planes at `bits` per sample (8, 10 or 12), for
/// [`yuv_to_rgb`]. `chroma_shift` is (x, y): (1, 1) for 4:2:0, (1, 0) for
/// 4:2:2, (0, 0) for 4:4:4. `uv` is `None` for a monochrome stream.
pub struct Planes<'a> {
    pub w: usize,
    pub h: usize,
    pub bits: u32,
    pub y: Plane<'a>,
    pub uv: Option<(Plane<'a>, Plane<'a>)>,
    pub chroma_shift: (u32, u32),
}

pub fn yuv_to_rgb(p: &Planes<'_>, colour: Colour) -> Vec<u8> {
    let Matrix { kr, kb } = colour.matrix;
    let kg = 1.0 - kr - kb;
    let max = ((1u32 << p.bits) - 1) as f32;
    let scale = max / 255.0;
    // Sample -> unit luma / centred chroma.
    let (y_off, y_span, c_span) = if colour.full_range { (0.0, max, max) } else { (16.0 * scale, 219.0 * scale, 224.0 * scale) };
    let c_mid = (1u32 << (p.bits - 1)) as f32;
    let mut out = vec![0u8; p.w * p.h * 3];
    for row in 0..p.h {
        for col in 0..p.w {
            let l = (p.y.0[row * p.y.1 + col] as f32 - y_off) / y_span;
            let (cb, cr) = match p.uv {
                Some(((u, us), (v, vs))) => {
                    let (cx, cy) = (col >> p.chroma_shift.0, row >> p.chroma_shift.1);
                    ((u[cy * us + cx] as f32 - c_mid) / c_span, (v[cy * vs + cx] as f32 - c_mid) / c_span)
                }
                None => (0.0, 0.0),
            };
            let r = l + 2.0 * (1.0 - kr) * cr;
            let b = l + 2.0 * (1.0 - kb) * cb;
            let g = (l - kr * r - kb * b) / kg;
            let o = &mut out[(row * p.w + col) * 3..][..3];
            o[0] = quantize(r * 255.0);
            o[1] = quantize(g * 255.0);
            o[2] = quantize(b * 255.0);
        }
    }
    out
}

fn quantize(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widen(v: &[u8]) -> Vec<u16> {
        v.iter().map(|&x| x as u16).collect()
    }

    fn round_trip(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
        let yuv = rgb_to_yuv420(rgb, w, h);
        let (cw, _) = Yuv420::chroma_size(w, h);
        let (y, u, v) = (widen(&yuv.y), widen(&yuv.u), widen(&yuv.v));
        let planes = Planes { w, h, bits: 8, y: (&y, w), uv: Some(((&u, cw), (&v, cw))), chroma_shift: (1, 1) };
        yuv_to_rgb(&planes, Colour { matrix: Matrix::BT709, full_range: false })
    }

    /// The matrix and ranges invert each other: a flat colour, which loses
    /// nothing to chroma subsampling, comes back within rounding.
    #[test]
    fn flat_colours_come_back_within_rounding() {
        for c in [[0u8, 0, 0], [255, 255, 255], [200, 30, 40], [20, 180, 90], [60, 70, 230], [128, 128, 128]] {
            let rgb: Vec<u8> = c.repeat(6 * 4);
            let back = round_trip(&rgb, 6, 4);
            let worst = rgb.iter().zip(&back).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(worst <= 2, "{c:?} came back as {:?}", &back[..3]);
        }
    }

    /// A smooth image loses only what 4:2:0 shares across each 2x2 block,
    /// most where colour changes fastest (here about 5 levels per pixel).
    #[test]
    fn a_smooth_image_keeps_its_detail() {
        let (w, h) = (37, 21);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            [(40.0 + 180.0 * x) as u8, (30.0 + 150.0 * y) as u8, (200.0 - 120.0 * x * y) as u8]
        }).collect();
        let back = round_trip(&rgb, w, h);
        let mse = rgb.iter().zip(&back).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>() / rgb.len() as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / mse).log10();
        assert!(psnr > 35.0, "{psnr:.1} dB");
    }

    /// The fixed points of limited range: black is luma 16, white 235, grey has no colour difference.
    #[test]
    fn black_white_and_grey_land_on_the_limited_range_levels() {
        let yuv = rgb_to_yuv420(&[0, 0, 0, 255, 255, 255], 2, 1);
        assert_eq!(yuv.y, [16, 235]);
        assert_eq!((yuv.u[0], yuv.v[0]), (128, 128));
    }
}
