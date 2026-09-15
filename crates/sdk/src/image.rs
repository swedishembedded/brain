// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`Image`]: the SDK's normalized wrapper over a generated picture.
//!
//! Wraps [`imaging::Rgb8`] rather than reinventing pixel storage, and
//! [`Image::save`] delegates to [`imaging::save`]'s extension-dispatched
//! codec (`crates/imaging/src/codec.rs`) rather than this crate writing its
//! own PNG/JPEG encoder.

use crate::{Error, Result};

/// One generated image: interleaved 8-bit RGB pixels plus their size.
///
/// Produced by [`crate::ImagePipeline::generate`]/`generate_with`, which
/// normalize each backend's own generation output into this ONE type --
/// flux2's `(Vec<u8>, u32, u32)` via [`Image::from_rgb8`], s3dit's float HWC
/// `[0,1]` `s3dit::pipeline::Image` via [`Image::from_hwc_unit`] -- so a
/// caller never sees which backend actually ran.
#[derive(Clone, Debug, PartialEq)]
pub struct Image(imaging::Rgb8);

impl Image {
    /// Wrap an already RGB8, `w*h*3`-byte pixel buffer (flux2's own
    /// `Pipeline::generate` output shape).
    pub(crate) fn from_rgb8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Image> {
        Ok(Image(imaging::Rgb8::new(width, height, pixels).map_err(Error::Backend)?))
    }

    /// Normalize an interleaved HWC `f32` buffer in `[0,1]` (s3dit's own
    /// `s3dit::pipeline::Image { hwc, w, h }` shape) into RGB8 -- via
    /// [`imaging::hwc_to_rgb8`], the SAME clamp-and-quantize helper every
    /// other f32-HWC-to-u8 writer in this workspace already uses
    /// (`crates/imaging/src/pixels.rs`), not a second copy of that rounding
    /// rule written here.
    pub(crate) fn from_hwc_unit(width: u32, height: u32, hwc: &[f32]) -> Result<Image> {
        Ok(Image(imaging::pixels::hwc_to_rgb8(hwc, width, height, 3, imaging::ChannelPolicy::RequireRgb).map_err(Error::Backend)?))
    }

    pub fn width(&self) -> u32 {
        self.0.w
    }

    pub fn height(&self) -> u32 {
        self.0.h
    }

    /// Interleaved RGB8 pixels, row-major, `width() * height() * 3` bytes.
    pub fn pixels(&self) -> &[u8] {
        &self.0.px
    }

    /// Write this image to `path`. The format is chosen from `path`'s
    /// extension (`.png`, `.jpg`/`.jpeg`, `.ppm`, or no extension at all --
    /// see [`imaging::save`]); any other extension is a clean error, never a
    /// silent reinterpretation.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        imaging::save(path, &self.0).map_err(Error::Backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_gradient(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h * 3)).map(|i| (i % 256) as u8).collect()
    }

    /// [`Image::save`] round-trips through the real `imaging` codec (never a
    /// hand-rolled encoder) -- a `.png` written here must be readable back.
    #[test]
    fn save_writes_a_real_png_via_the_shared_codec() {
        let img = Image::from_rgb8(4, 3, tiny_gradient(4, 3)).unwrap();
        let out = std::env::temp_dir().join(format!("brain-sdk-image-save-{}.png", std::process::id()));
        img.save(&out).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "must be a real PNG signature");
        std::fs::remove_file(&out).ok();
    }

    /// A mismatched pixel buffer is a clean [`Error::Backend`], never a
    /// panic -- the same contract [`imaging::Rgb8::new`] itself carries.
    #[test]
    fn from_rgb8_refuses_a_mismatched_buffer_length_cleanly() {
        let err = Image::from_rgb8(4, 3, vec![0u8; 5]).unwrap_err();
        assert!(matches!(err, Error::Backend(_)), "{err:?}");
    }

    /// An unsupported extension refuses to write rather than silently
    /// picking a format the filename does not say.
    #[test]
    fn save_refuses_an_unsupported_extension() {
        let img = Image::from_rgb8(2, 2, tiny_gradient(2, 2)).unwrap();
        let out = std::env::temp_dir().join(format!("brain-sdk-image-save-{}.webp", std::process::id()));
        let err = img.save(&out).unwrap_err();
        assert!(matches!(err, Error::Backend(_)), "{err:?}");
        assert!(!out.exists());
    }

    /// [`Image::from_hwc_unit`]: s3dit's own float-HWC `[0,1]` shape
    /// normalizes to the SAME RGB8 a caller would get from the flux2
    /// backend -- clamped and quantized via `imaging::pixels::hwc_to_rgb8`,
    /// not a second, hand-rolled rounding rule.
    #[test]
    fn from_hwc_unit_normalizes_float_hwc_to_rgb8() {
        // 1x2 pixels, RGB: black and (over-range, clamped) white.
        let hwc = [0.0f32, 0.0, 0.0, 2.0, -1.0, 0.5];
        let img = Image::from_hwc_unit(2, 1, &hwc).unwrap();
        assert_eq!(img.pixels(), &[0, 0, 0, 255, 0, 128]);
    }

    /// A mismatched HWC buffer is a clean [`Error::Backend`], never a panic.
    #[test]
    fn from_hwc_unit_refuses_a_mismatched_buffer_length_cleanly() {
        let err = Image::from_hwc_unit(2, 2, &[0.0f32; 3]).unwrap_err();
        assert!(matches!(err, Error::Backend(_)), "{err:?}");
    }
}
