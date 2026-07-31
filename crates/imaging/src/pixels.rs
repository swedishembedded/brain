// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pixel containers, rectangles, and the layout / value-range conversions.
//!
//! Everything here is a **permutation or a cast** — index bookkeeping, not
//! arithmetic over the image. Per-pixel arithmetic (scale, shift, normalise)
//! belongs on the device; see [`crate::Ctx::affine`] and [`crate::Normalization`].
//!
//! The layout functions are generic over the element type on purpose. The
//! workspace had `chw_to_hwc` five times — once generic over `c` in
//! `cli::image_io`, twice with `c = 3` hard-coded in `crates/npu`, twice more in
//! yolo's tests — plus a byte-typed twin in `wm-display::record` that exists only
//! because the f32 versions divide by 255. One generic function with the value
//! conversion kept **separate** covers all of them.

/// An interleaved 8-bit RGB image — what a decoder produces and an encoder
/// consumes.
///
/// Deliberately the only owning pixel container in this crate: everything else
/// is a borrowed slice plus its dimensions, so nothing is copied to call it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb8 {
    pub w: u32,
    pub h: u32,
    /// `w * h * 3` bytes, row-major, R,G,B interleaved.
    pub px: Vec<u8>,
}

impl Rgb8 {
    /// Wrap `px`, checking it is exactly `w * h * 3` bytes.
    pub fn new(w: u32, h: u32, px: Vec<u8>) -> Result<Rgb8, String> {
        let need = w as usize * h as usize * 3;
        if px.len() != need {
            return Err(format!("Rgb8: {w}x{h} needs {need} bytes, got {}", px.len()));
        }
        Ok(Rgb8 { w, h, px })
    }

    /// Interleaved HWC f32 in `[0, 1]` — the form every model preprocessor
    /// starts from.
    pub fn to_hwc_unit(&self) -> Vec<f32> {
        u8_to_unit(&self.px)
    }

    /// Full-image rectangle.
    pub fn rect(&self) -> Rect {
        Rect { x: 0, y: 0, w: self.w, h: self.h }
    }
}

/// An axis-aligned pixel rectangle, `x`/`y` = top-left, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Rect {
        Rect { x, y, w, h }
    }
    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }
    /// Exclusive right / bottom edges.
    pub fn right(&self) -> u32 {
        self.x + self.w
    }
    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }
    /// The overlap of two rectangles, or `None` when they do not touch.
    pub fn intersect(&self, o: &Rect) -> Option<Rect> {
        let x = self.x.max(o.x);
        let y = self.y.max(o.y);
        let r = self.right().min(o.right());
        let b = self.bottom().min(o.bottom());
        if r <= x || b <= y {
            return None;
        }
        Some(Rect { x, y, w: r - x, h: b - y })
    }
    /// Clip to a `w x h` frame, or `None` when entirely outside it.
    pub fn clip(&self, w: u32, h: u32) -> Option<Rect> {
        self.intersect(&Rect { x: 0, y: 0, w, h })
    }
}

/// What to do when a buffer has fewer than three channels but RGB8 is wanted.
///
/// Both behaviours exist in the workspace today and neither is obviously right:
/// `cli::caps_cli::save_blob` replicates channel 0 (so a depth map or a mask
/// saves as a visible grey image), while `wm-display::chw_to_rgb8` requires
/// `c >= 3`. Making it a parameter is the point — a caller that picks
/// [`ChannelPolicy::RequireRgb`] gets an error instead of a silently grey PNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChannelPolicy {
    /// Fewer than 3 channels is an error. The default: a single-channel buffer
    /// rendered as grey is usually a bug upstream.
    #[default]
    RequireRgb,
    /// Replicate channel 0 into R, G and B.
    ReplicateFirst,
}

/// Planar CHW `[c, h, w]` -> interleaved HWC `[h, w, c]`.
///
/// Generic over the element type: `f32` for activations, `u8` for the
/// episode-dataset frames in `wm-display::record`. The `/255` that the f32
/// call sites want is [`u8_to_unit`], applied separately — fusing it into the
/// permutation is how `wm-display` ended up needing a second function.
pub fn chw_to_hwc<T: Copy + Default>(chw: &[T], c: usize, h: usize, w: usize) -> Vec<T> {
    assert_eq!(chw.len(), c * h * w, "chw_to_hwc: buffer is not c*h*w");
    let mut out = vec![T::default(); c * h * w];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ch] = chw[ch * h * w + y * w + x];
            }
        }
    }
    out
}

/// Interleaved HWC `[h, w, c]` -> planar CHW `[c, h, w]`. Inverse of
/// [`chw_to_hwc`].
pub fn hwc_to_chw<T: Copy + Default>(hwc: &[T], c: usize, h: usize, w: usize) -> Vec<T> {
    assert_eq!(hwc.len(), c * h * w, "hwc_to_chw: buffer is not h*w*c");
    let mut out = vec![T::default(); c * h * w];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[ch * h * w + y * w + x] = hwc[(y * w + x) * c + ch];
            }
        }
    }
    out
}

/// `u8` -> f32 in `[0, 1]`. Layout-agnostic (it is a per-element cast).
pub fn u8_to_unit(px: &[u8]) -> Vec<f32> {
    px.iter().map(|&b| b as f32 / 255.0).collect()
}

/// One element of [`unit_to_u8`]: clamp to `[0,1]`, then round half **up**.
///
/// Private and `#[inline]` so the quantisation rule has exactly one spelling in
/// this crate. [`hwc_to_rgb8`] needs it per element (it gathers and replicates
/// channels rather than walking a slice), and writing `* 255.0 + 0.5` a second
/// time there is how the tie-break drifts from [`unit_to_u8`] at the next edit.
#[inline]
fn quantize_unit(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// f32 in `[0, 1]` -> `u8`, clamped, round-half-up (`*255 + 0.5`).
///
/// The rounding matches every existing writer in the workspace
/// (`wm-display::chw_to_rgb8`, `cli::caps_cli::save_blob`,
/// `cli::splat_cli::write_ppm`); `f32::round` would differ on ties.
pub fn unit_to_u8(v: &[f32]) -> Vec<u8> {
    v.iter().map(|&x| quantize_unit(x)).collect()
}

/// Interleaved HWC `[h, w, c]` in `[0, 1]` -> [`Rgb8`].
///
/// Channels beyond the third are dropped (an RGBA buffer saves as RGB); fewer
/// than three are handled per [`ChannelPolicy`].
pub fn hwc_to_rgb8(hwc: &[f32], w: u32, h: u32, c: usize, policy: ChannelPolicy) -> Result<Rgb8, String> {
    let n = w as usize * h as usize;
    if hwc.len() != n * c {
        return Err(format!("hwc_to_rgb8: {w}x{h}x{c} needs {} values, got {}", n * c, hwc.len()));
    }
    if c == 0 {
        return Err("hwc_to_rgb8: zero channels".to_string());
    }
    if c < 3 && policy == ChannelPolicy::RequireRgb {
        return Err(format!(
            "hwc_to_rgb8: {c} channel(s) with ChannelPolicy::RequireRgb — pass ReplicateFirst to render it as grey"
        ));
    }
    let mut px = vec![0u8; n * 3];
    for i in 0..n {
        for ch in 0..3 {
            let src = if c >= 3 { hwc[i * c + ch] } else { hwc[i * c] };
            px[i * 3 + ch] = quantize_unit(src);
        }
    }
    Rgb8::new(w, h, px)
}

/// Planar CHW `[c, h, w]` in `[0, 1]` -> [`Rgb8`]. Composition of
/// [`chw_to_hwc`] and [`hwc_to_rgb8`], spelled out so the common case is one
/// call and one allocation of the intermediate.
pub fn chw_to_rgb8(chw: &[f32], w: u32, h: u32, c: usize, policy: ChannelPolicy) -> Result<Rgb8, String> {
    let hwc = chw_to_hwc(chw, c, h as usize, w as usize);
    hwc_to_rgb8(&hwc, w, h, c, policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chw_hwc_round_trips_for_f32_and_u8() {
        let (c, h, w) = (3usize, 2usize, 4usize);
        let chw: Vec<f32> = (0..(c * h * w)).map(|i| i as f32).collect();
        let back = hwc_to_chw(&chw_to_hwc(&chw, c, h, w), c, h, w);
        assert_eq!(back, chw, "f32 round trip must be bitwise identity");

        let bytes: Vec<u8> = (0..(c * h * w) as u8).collect();
        let back_u8 = hwc_to_chw(&chw_to_hwc(&bytes, c, h, w), c, h, w);
        assert_eq!(back_u8, bytes, "the same code path serves byte-typed frames");
    }

    #[test]
    fn chw_to_hwc_interleaves_in_the_documented_order() {
        // 1x2 image, 3 channels: R=[1,2] G=[3,4] B=[5,6].
        let chw = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(chw_to_hwc(&chw, 3, 1, 2), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn unit_to_u8_rounds_half_up_and_clamps() {
        assert_eq!(unit_to_u8(&[0.0, 1.0, -1.0, 2.0]), vec![0, 255, 0, 255]);
        // 0.5 * 255 = 127.5 -> 128 under round-half-up, 127 under truncation.
        assert_eq!(unit_to_u8(&[0.5]), vec![128]);
    }

    #[test]
    fn u8_unit_round_trip_is_exact() {
        let bytes: Vec<u8> = (0..=255u8).collect();
        assert_eq!(unit_to_u8(&u8_to_unit(&bytes)), bytes);
    }

    #[test]
    fn single_channel_needs_an_explicit_policy() {
        let gray = [0.25f32, 0.5, 0.75, 1.0];
        assert!(hwc_to_rgb8(&gray, 2, 2, 1, ChannelPolicy::RequireRgb).is_err());
        let img = hwc_to_rgb8(&gray, 2, 2, 1, ChannelPolicy::ReplicateFirst).unwrap();
        assert_eq!(&img.px[..3], &[64, 64, 64]);
        assert_eq!(&img.px[3..6], &[128, 128, 128]);
    }

    #[test]
    fn extra_channels_are_dropped_not_rejected() {
        // RGBA in, RGB out.
        let rgba = [1.0f32, 0.0, 0.0, 0.5];
        let img = hwc_to_rgb8(&rgba, 1, 1, 4, ChannelPolicy::RequireRgb).unwrap();
        assert_eq!(img.px, vec![255, 0, 0]);
    }

    #[test]
    fn rect_intersect_and_clip() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 10, 10);
        assert_eq!(a.intersect(&b), Some(Rect::new(5, 5, 5, 5)));
        assert_eq!(a.intersect(&Rect::new(20, 20, 1, 1)), None);
        assert_eq!(Rect::new(8, 8, 10, 10).clip(10, 10), Some(Rect::new(8, 8, 2, 2)));
        assert_eq!(a.area(), 100);
    }
}
