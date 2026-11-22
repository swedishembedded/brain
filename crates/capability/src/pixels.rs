// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The 8-bit pixel value types a decoder produces and a serving front end consumes ([`Rgb8`], [`Rgba8`],
//! [`Decoded`]) and the byte <-> unit-float casts they use.
//!
//! They live here, in the contracts layer beside the image blob codec, so a serving crate that only turns request
//! blobs into pixels does not link the imaging pipeline (and with it vision, video codecs and forecasting).
//! `imaging::pixels` and `imaging::Decoded` re-export them: there is one definition.
//!
//! Swedish Embedded AB implements image serving front ends for its clients. If your team needs expertise in
//! inference serving then you can procure our services by sending an email to info@swedishembedded.com.

/// An interleaved 8-bit RGB image - what a decoder produces and an encoder
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

    /// Interleaved HWC f32 in `[0, 1]` - the form every model preprocessor
    /// starts from.
    pub fn to_hwc_unit(&self) -> Vec<f32> {
        u8_to_unit(&self.px)
    }

    /// The 8-bit image of interleaved HWC f32 in `[0, 1]` (the inverse of [`Self::to_hwc_unit`]: a value that came
    /// from a byte returns to it exactly). Errors when the buffer is not `w * h * 3` floats.
    pub fn from_hwc_unit(hwc: &[f32], w: u32, h: u32) -> Result<Rgb8, String> {
        let px = hwc.iter().map(|&v| quantize_unit(v)).collect();
        Rgb8::new(w, h, px)
    }
}

/// An interleaved 8-bit RGBA image, straight (non-premultiplied) alpha: what a decoder produces for a source that
/// carries transparency, kept apart from [`Rgb8`] so a caller that must composite it cannot forget to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8 {
    pub w: u32,
    pub h: u32,
    /// `w * h * 4` bytes, row-major, R,G,B,A interleaved.
    pub px: Vec<u8>,
}

impl Rgba8 {
    /// Wrap `px`, checking it is exactly `w * h * 4` bytes.
    pub fn new(w: u32, h: u32, px: Vec<u8>) -> Result<Rgba8, String> {
        let need = w as usize * h as usize * 4;
        if px.len() != need {
            return Err(format!("Rgba8: {w}x{h} needs {need} bytes, got {}", px.len()));
        }
        Ok(Rgba8 { w, h, px })
    }

    /// Interleaved HWC f32 in `[0, 1]`, alpha the fourth channel.
    pub fn to_hwc_unit(&self) -> Vec<f32> {
        u8_to_unit(&self.px)
    }

    /// The 8-bit image of interleaved RGBA f32 in `[0, 1]` (the inverse of [`Self::to_hwc_unit`]). Errors when the
    /// buffer is not `w * h * 4` floats.
    pub fn from_hwc_unit(hwc: &[f32], w: u32, h: u32) -> Result<Rgba8, String> {
        let px = hwc.iter().map(|&v| quantize_unit(v)).collect();
        Rgba8::new(w, h, px)
    }
}

/// A decoded image with its transparency kept: [`Decoded::Rgba`] for a source whose colour type has an alpha channel
/// (PNG RGBA, grey+alpha, palette with transparency), [`Decoded::Rgb`] for every other. 8-bit sources are exact; a
/// 16-bit source goes through `image`'s 16-to-8-bit conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decoded {
    Rgb(Rgb8),
    Rgba(Rgba8),
}

impl Decoded {
    pub fn size(&self) -> (u32, u32) {
        match self {
            Decoded::Rgb(i) => (i.w, i.h),
            Decoded::Rgba(i) => (i.w, i.h),
        }
    }
}

/// `u8` -> f32 in `[0, 1]`. Layout-agnostic (it is a per-element cast).
pub fn u8_to_unit(px: &[u8]) -> Vec<f32> {
    px.iter().map(|&b| b as f32 / 255.0).collect()
}

/// One element of `imaging::pixels::unit_to_u8`: clamp to `[0,1]`, then round half **up**.
///
/// `#[inline]` so the quantisation rule has exactly one spelling in the
/// workspace. `imaging::pixels::hwc_to_rgb8` needs it per element (it gathers and replicates
/// channels rather than walking a slice), and writing `* 255.0 + 0.5` a second
/// time there is how the tie-break drifts from `imaging::pixels::unit_to_u8` at the next edit.
#[inline]
pub fn quantize_unit(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
