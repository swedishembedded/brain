// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pad-and-tile preprocessing: an image of any size becomes a grid of fixed-size square regions, one pixel row per
//! region, with no resampling.
//!
//! The recipe a tile-per-token vision tower reads:
//!
//! 1. transparency is composited onto an 8-pixel checkerboard of two greys with the integer formula
//!    `(c * a + bg * (255 - a) + 127) / 255` ([`composite_over_checkerboard`]), so a transparent image never
//!    reaches the model as the black or white a plain drop of the alpha channel would give;
//! 2. the image is padded at the right and the bottom to a whole number of regions with a constant on the 0-255
//!    scale ([`RegionTiling::pad_fill`]);
//! 3. every value is rescaled and normalised in the order and with the fused constants of the reference processor:
//!    `(v - mean / rescale) / (std / rescale)` in f32, which for a source byte equals `(v / 255 - mean) / std` only up
//!    to rounding, so the form is kept exactly;
//! 4. the padded image is cut into `region x region` cells, row-major over the grid, each cell stored as
//!    `[y][x][channel]`.
//!
//! Nothing here is arithmetic over a model-sized buffer, and the layout is the reference processor's own, so it is
//! host glue (see the crate header's line between this crate and a kernel); a test compares it byte for byte with the
//! reference's dumped `pixel_values`.
//!
//! Input is untrusted: an empty image, a side over [`RegionTiling::max_side`] and a source with more pixels than that
//! bound allows are refused by name, before any pixel is decoded or allocated for the output.

use crate::codec::{decode_alpha_bounded, Decoded};
use crate::pixels::{Rgb8, Rgba8};

/// The two greys of the compositing checkerboard and its cell size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkerboard {
    /// Side of one square cell, in pixels.
    pub cell: u32,
    /// The grey of cells whose `(row / cell + col / cell)` is even.
    pub light: u8,
    /// The grey of the other cells.
    pub dark: u8,
}

/// How an image is padded, normalised and cut into regions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionTiling {
    /// Side of one square region, in pixels.
    pub region: u32,
    /// The checkerboard transparency is composited onto.
    pub checkerboard: Checkerboard,
    /// The constant the padding takes, on the 0-255 scale and before normalisation.
    pub pad_fill: f32,
    /// The reference's `rescale_factor` (`1 / 255`), which only enters through the fused constants.
    pub rescale_factor: f64,
    /// Per-channel mean on the rescaled scale.
    pub mean: [f32; 3],
    /// Per-channel standard deviation on the rescaled scale.
    pub std: [f32; 3],
    /// The longest side, in pixels, an input may have.
    pub max_side: u32,
}

/// One image as a grid of regions.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionImage {
    /// Region rows (image height over the region side, rounded up).
    pub rows: u32,
    /// Region columns.
    pub cols: u32,
    /// `[rows * cols, region * region * 3]` f32, regions row-major over the grid, each `[y][x][channel]`.
    pub pixels: Vec<f32>,
}

impl RegionImage {
    /// `[1, rows, cols]`: temporal, height and width of the grid in regions (the reference's `image_grid_thw`).
    pub fn grid_thw(&self) -> [u32; 3] {
        [1, self.rows, self.cols]
    }

    /// The number of regions, one language-model token each.
    pub fn regions(&self) -> usize {
        self.rows as usize * self.cols as usize
    }
}

/// The straight-alpha image composited onto the checkerboard, as RGB.
pub fn composite_over_checkerboard(img: &Rgba8, board: Checkerboard) -> Rgb8 {
    let mut px = Vec::with_capacity(img.w as usize * img.h as usize * 3);
    for (i, p) in img.px.as_chunks::<4>().0.iter().enumerate() {
        let (y, x) = (i as u32 / img.w, i as u32 % img.w);
        let even = (y / board.cell + x / board.cell).is_multiple_of(2);
        let bg = i32::from(if even { board.light } else { board.dark });
        let a = i32::from(p[3]);
        for &c in &p[..3] {
            px.push(((i32::from(c) * a + bg * (255 - a) + 127) / 255) as u8);
        }
    }
    Rgb8 { w: img.w, h: img.h, px }
}

impl RegionTiling {
    fn check(&self) -> Result<(), String> {
        if self.region == 0 || self.max_side == 0 || self.checkerboard.cell == 0 {
            return Err("RegionTiling: region, max_side and the checkerboard cell must be positive".to_string());
        }
        if self.std.iter().any(|&s| s == 0.0 || !s.is_finite()) || !(self.rescale_factor.is_finite() && self.rescale_factor > 0.0) {
            return Err("RegionTiling: std and rescale_factor must be finite and non-zero".to_string());
        }
        Ok(())
    }

    /// Refuse an empty image or a side over [`Self::max_side`], naming the size.
    pub fn check_size(&self, w: u32, h: u32) -> Result<(), String> {
        if w == 0 || h == 0 {
            return Err(format!("image of {w}x{h} pixels is empty"));
        }
        if w > self.max_side || h > self.max_side {
            return Err(format!("image of {w}x{h} pixels has a side over the limit of {} pixels", self.max_side));
        }
        Ok(())
    }

    /// Decode PNG, JPEG or P6 `bytes` (keeping alpha) and cut the image into regions.
    pub fn preprocess_bytes(&self, bytes: &[u8]) -> Result<RegionImage, String> {
        self.check()?;
        let max_pixels = u64::from(self.max_side) * u64::from(self.max_side);
        self.preprocess(&decode_alpha_bounded(bytes, max_pixels)?)
    }

    /// Cut a decoded image into regions: composite, pad, normalise, tile.
    pub fn preprocess(&self, img: &Decoded) -> Result<RegionImage, String> {
        self.check()?;
        let (w, h) = img.size();
        self.check_size(w, h)?;
        let composited;
        let rgb = match img {
            Decoded::Rgb(i) => i,
            Decoded::Rgba(i) => {
                composited = composite_over_checkerboard(i, self.checkerboard);
                &composited
            }
        };
        let r = self.region as usize;
        let (rows, cols) = (h.div_ceil(self.region), w.div_ceil(self.region));
        let inv = (1.0 / self.rescale_factor) as f32;
        let mean = self.mean.map(|m| m * inv);
        let std = self.std.map(|s| s * inv);
        let norm = |v: f32, c: usize| (v - mean[c]) / std[c];
        let pad = [norm(self.pad_fill, 0), norm(self.pad_fill, 1), norm(self.pad_fill, 2)];
        let cell = r * r * 3;
        let mut pixels = vec![0f32; rows as usize * cols as usize * cell];
        for gy in 0..rows as usize {
            for gx in 0..cols as usize {
                let out = &mut pixels[(gy * cols as usize + gx) * cell..][..cell];
                for y in 0..r {
                    for x in 0..r {
                        let (sy, sx) = (gy * r + y, gx * r + x);
                        let o = (y * r + x) * 3;
                        if sy < h as usize && sx < w as usize {
                            let s = (sy * w as usize + sx) * 3;
                            for c in 0..3 {
                                out[o + c] = norm(f32::from(rgb.px[s + c]), c);
                            }
                        } else {
                            out[o..o + 3].copy_from_slice(&pad);
                        }
                    }
                }
            }
        }
        Ok(RegionImage { rows, cols, pixels })
    }
}
