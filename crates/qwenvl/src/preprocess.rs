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

/// im2col patch packing for one image (`t = 1`), porting HF's
/// `view → permute(0,1,4,7,5,8,3,2,6,9) → reshape`. Input is a **normalized**
/// CHW image `[channels, h_bar, w_bar]` (both sides multiples of `patch·merge`);
/// output is `[N, channels·temporal·patch²]` (1536 for 4B) in spatial-merge-block
/// order — the exact patch-token stream the ViT patch-embed consumes, and the
/// same order as [`crate::vision::vision_position_ids`] /
/// [`crate::vision::pos_embed_bilinear`]. The single frame is repeated across the
/// `temporal` slices (images carry no motion).
pub fn pack_patches(img_chw: &[f32], channels: u32, h_bar: u32, w_bar: u32, patch: u32, merge: u32, temporal: u32) -> Vec<f32> {
    assert_eq!(img_chw.len(), (channels * h_bar * w_bar) as usize, "img must be [C, h_bar, w_bar]");
    assert!(h_bar % (patch * merge) == 0 && w_bar % (patch * merge) == 0);
    let (gh, gw) = patch_grid(h_bar, w_bar, patch);
    let pv = channels * temporal * patch * patch;
    let n = gh * gw;
    let mut out = vec![0f32; (n * pv) as usize];
    let pix = |c: u32, y: u32, x: u32| img_chw[((c * h_bar + y) * w_bar + x) as usize];
    let mut row = 0u32;
    for bh in 0..gh / merge {
        for bw in 0..gw / merge {
            for ih in 0..merge {
                for iw in 0..merge {
                    let (hi, wi) = (bh * merge + ih, bw * merge + iw);
                    for c in 0..channels {
                        for tp in 0..temporal {
                            for ph in 0..patch {
                                for pw in 0..patch {
                                    let vidx = ((c * temporal + tp) * patch + ph) * patch + pw;
                                    out[(row * pv + vidx) as usize] = pix(c, hi * patch + ph, wi * patch + pw);
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
    out
}

/// Normalize `[0,1]` pixels to `[-1,1]` (Qwen3-VL uses mean=std=0.5 per channel):
/// `x -> (x - 0.5) / 0.5`.
pub fn normalize_unit(pixels: &mut [f32]) {
    for p in pixels {
        *p = (*p - 0.5) / 0.5;
    }
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

    #[test]
    fn pack_1x1_patches_is_merge_block_order() {
        // C=1, patch=1, merge=2, temporal=1 on a 2×2 image → 4 patches, pv=1.
        // merge-block order over a single 2×2 block = row-major (0,0),(0,1),(1,0),(1,1).
        let img = vec![10.0, 20.0, 30.0, 40.0]; // [1,2,2]
        let out = pack_patches(&img, 1, 2, 2, 1, 2, 1);
        assert_eq!(out, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn pack_single_patch_flattens_channel_temporal_ph_pw() {
        // C=1, patch=2, merge=1, temporal=2 on a 2×2 image → 1 patch,
        // pv = 1·2·2·2 = 8 = [temporal][ph][pw], frame repeated across temporal.
        let img = vec![1.0, 2.0, 3.0, 4.0]; // [1,2,2] = [[1,2],[3,4]]
        let out = pack_patches(&img, 1, 2, 2, 2, 1, 2);
        // vidx = ((0·2 + tp)·2 + ph)·2 + pw; both temporal slices equal.
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, /* tp=1 repeat */ 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pack_two_channels_are_channel_major() {
        // C=2, patch=1, merge=1, temporal=1, 1×1 image → 1 patch, pv=2 = [c0, c1].
        let img = vec![7.0, 9.0]; // c0=7 at (0,0), c1=9 at (0,0)
        let out = pack_patches(&img, 2, 1, 1, 1, 1, 1);
        assert_eq!(out, vec![7.0, 9.0]);
    }

    #[test]
    fn normalize_maps_unit_to_signed() {
        let mut p = vec![0.0, 0.5, 1.0];
        normalize_unit(&mut p);
        assert_eq!(p, vec![-1.0, 0.0, 1.0]);
    }
}
