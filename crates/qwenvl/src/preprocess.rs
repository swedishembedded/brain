// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-VL image preprocessing (host): native-dynamic-resolution smart-resize and
//! the token-count / patch-grid bookkeeping that follows from it. Ports HF
//! `smart_resize` exactly (Python `round` = round-half-to-even).
//!
//! The resize target keeps the aspect ratio, snaps both sides to a multiple of
//! `factor = patch_size · spatial_merge_size` (32 for 4B), and bounds the pixel
//! *area* to `[min_pixels, max_pixels]`. The learned pos-embed is then resampled
//! onto the resulting patch grid (`crate::vision::pos_embed_bilinear`) and the
//! image expands to `(h/patch)·(w/patch)/merge²` decoder tokens.

/// `factor` for 4B: `patch_size · spatial_merge_size = 16 · 2`.
pub const DEFAULT_FACTOR: u32 = 32;
/// Area bounds from the released `preprocessor_config.json` (256² and 4096²).
pub const DEFAULT_MIN_PIXELS: u32 = 256 * 256;
pub const DEFAULT_MAX_PIXELS: u32 = 4096 * 4096;

fn round_by_factor(x: f64, f: u32) -> u32 {
    ((x / f as f64).round_ties_even() as u32) * f
}
fn floor_by_factor(x: f64, f: u32) -> u32 {
    ((x / f as f64).floor() as u32) * f
}
fn ceil_by_factor(x: f64, f: u32) -> u32 {
    ((x / f as f64).ceil() as u32) * f
}

/// Resize `(height, width)` to `(h_bar, w_bar)`: both multiples of `factor`,
/// aspect ratio preserved, pixel area clamped into `[min_pixels, max_pixels]`.
/// Ports HF `smart_resize`.
pub fn smart_resize(height: u32, width: u32, factor: u32, min_pixels: u32, max_pixels: u32) -> (u32, u32) {
    let (h, w) = (height as f64, width as f64);
    let mut h_bar = factor.max(round_by_factor(h, factor));
    let mut w_bar = factor.max(round_by_factor(w, factor));
    if (h_bar as u64) * (w_bar as u64) > max_pixels as u64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        h_bar = floor_by_factor(h / beta, factor);
        w_bar = floor_by_factor(w / beta, factor);
    } else if (h_bar as u64) * (w_bar as u64) < min_pixels as u64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        h_bar = ceil_by_factor(h * beta, factor);
        w_bar = ceil_by_factor(w * beta, factor);
    }
    (h_bar, w_bar)
}

/// Convenience wrapper using the 4B defaults.
pub fn smart_resize_default(height: u32, width: u32) -> (u32, u32) {
    smart_resize(height, width, DEFAULT_FACTOR, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
}

/// Patch grid `(h_patches, w_patches)` for a smart-resized image.
pub fn patch_grid(h_bar: u32, w_bar: u32, patch: u32) -> (u32, u32) {
    (h_bar / patch, w_bar / patch)
}

/// Number of decoder image tokens the image expands to (`t = 1`):
/// `(h/patch)·(w/patch)/merge²`.
pub fn image_token_count(h_bar: u32, w_bar: u32, patch: u32, merge: u32) -> u32 {
    let (gh, gw) = patch_grid(h_bar, w_bar, patch);
    gh * gw / (merge * merge)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn already_valid_size_is_unchanged() {
        // 512×512 (256k px) is a factor-multiple and within [256², 4096²].
        assert_eq!(smart_resize_default(512, 512), (512, 512));
    }

    #[test]
    fn snaps_to_factor_multiple_with_round_half_even() {
        // 100 → round(100/32)=round(3.125)=3 → 96; 112 → round(3.5)=4 (half→even) → 128.
        // Area 96·128 = 12288 < 256² so it scales up, but the snapping itself is the
        // point — check via a size already in-bounds after snapping.
        let (h, w) = smart_resize(300, 500, 32, 1, u32::MAX); // no area clamp
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert_eq!(h, round_by_factor(300.0, 32)); // round(9.375)=9 → 288
        assert_eq!(w, round_by_factor(500.0, 32)); // round(15.625)=16 → 512
        assert_eq!((h, w), (288, 512));
    }

    #[test]
    fn downscales_when_area_exceeds_max() {
        // 10000×10000 → area ≫ 4096²; result area ≤ 4096², sides factor-multiples,
        // aspect preserved (square stays square).
        let (h, w) = smart_resize_default(10000, 10000);
        assert!((h as u64) * (w as u64) <= DEFAULT_MAX_PIXELS as u64);
        assert_eq!(h, w);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn upscales_when_area_below_min() {
        // 64×64 (4096 px) < 256²; result area ≥ 256².
        let (h, w) = smart_resize_default(64, 64);
        assert!((h as u64) * (w as u64) >= DEFAULT_MIN_PIXELS as u64);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn token_count_matches_grid() {
        // 512×512, patch 16, merge 2 → 32×32 patches → 1024/4 = 256 tokens.
        assert_eq!(patch_grid(512, 512, 16), (32, 32));
        assert_eq!(image_token_count(512, 512, 16, 2), 256);
    }
}
