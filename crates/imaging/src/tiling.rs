// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tiling for images too large for one pass.
//!
//! Net-new: pixel-space tiling exists nowhere in the workspace today.
//! `moondream::preprocess::{select_tiling, reconstruct_from_crops}` looks
//! adjacent but works in **patch** units against one model's crop policy, and
//! `moondream`'s own header records that the pixel-space `overlap_crop_image` is
//! missing. This is that.
//!
//! ## Halo tiles with disjoint cores — why there is no blending
//!
//! Each tile is a **core** rectangle grown by a `halo` of context. Tiles overlap
//! on their *input* so a convolutional model sees the neighbourhood of every
//! pixel it produces, but each tile contributes only its core to the output, and
//! the cores tile the image exactly: disjoint, and covering.
//!
//! That removes the blend entirely, and with it the weight table, the seam
//! artefacts, and the "which tile wins" question. Recomposition is therefore two
//! kernels ([`crate::Ctx::crop`] to take the core out of a tile result,
//! [`crate::Ctx::add_region`] to place it into a zeroed canvas) and no host loop
//! over pixels.
//!
//! Feathered overlap-blending would need a per-pixel weight image and a weighted
//! accumulate; brain has no kernel for the latter, and inventing a host loop for
//! it is exactly the trap this crate exists to avoid. If a model genuinely needs
//! blended overlap (a diffusion tail, where each tile's *content* differs rather
//! than just its border error), that arrives as a `blend_acc.wgsl` plus a plan
//! variant — not as a host fallback here.
//!
//! ## Choosing `halo`
//!
//! `halo` must cover the model's receptive-field radius; anything less shows as
//! a visible grid. An **interior** tile costs `((tile + 2*halo)/tile)²` in
//! compute, so `tile = 512, halo = 32` is `(576/512)² = 1.27x` while
//! `tile = 128, halo = 32` is `(192/128)² = 2.25x` — the halo is a *fixed* ring,
//! so its relative cost grows as the square of `1 + 2*halo/tile`, not as its
//! area.
//!
//! That formula is the ceiling, not the answer: border tiles have their halo
//! clipped, so a whole plan always costs less. Do not quote it for a specific
//! image — ask [`TilePlan::overhead`], which sums the plan's actual `src` areas.
//! For 1024x1024 at `tile = 512, halo = 32` the ceiling is 1.27x and the real
//! figure is **1.129x**, because every one of the four tiles is clipped on two
//! sides (pinned in `overhead_is_the_measured_read_amplification`).

use crate::pixels::Rect;

/// Tiling parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSpec {
    /// Core edge length in pixels — the region each tile contributes to the
    /// output. The last row/column of cores is clipped, so `w`/`h` need not be
    /// multiples of it.
    pub tile: u32,
    /// Context grown around the core on every side, clipped at the image
    /// border. Must cover the consumer's receptive-field radius.
    pub halo: u32,
}

impl TileSpec {
    pub fn new(tile: u32, halo: u32) -> TileSpec {
        assert!(tile > 0, "TileSpec: tile edge must be non-zero");
        TileSpec { tile, halo }
    }
}

/// One tile of a [`TilePlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// What to crop out of the source image (core + halo, clipped to the image).
    pub src: Rect,
    /// The core **within `src`'s own coordinate frame** — what to crop out of
    /// the tile's result. Always the same size as [`Tile::dst`].
    pub keep: Rect,
    /// Where that core goes in the output image.
    pub dst: Rect,
}

/// A complete cover of a `w x h` image by halo tiles with disjoint cores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TilePlan {
    pub w: u32,
    pub h: u32,
    pub spec: TileSpec,
    pub tiles: Vec<Tile>,
}

impl TilePlan {
    /// Build the cover. A `w x h` that fits in one core yields exactly one tile
    /// whose `src`, `keep` and `dst` are the whole image, so a caller need not
    /// special-case small images.
    pub fn new(w: u32, h: u32, spec: TileSpec) -> TilePlan {
        assert!(w > 0 && h > 0, "TilePlan: image must be non-empty");
        let mut tiles = Vec::new();
        let mut y = 0u32;
        while y < h {
            let ch = spec.tile.min(h - y);
            let mut x = 0u32;
            while x < w {
                let cw = spec.tile.min(w - x);
                // Core, then the haloed source clipped to the image.
                let sx = x.saturating_sub(spec.halo);
                let sy = y.saturating_sub(spec.halo);
                let sr = (x + cw + spec.halo).min(w);
                let sb = (y + ch + spec.halo).min(h);
                let src = Rect::new(sx, sy, sr - sx, sb - sy);
                tiles.push(Tile {
                    src,
                    // The core's offset inside the (possibly clipped) source.
                    keep: Rect::new(x - sx, y - sy, cw, ch),
                    dst: Rect::new(x, y, cw, ch),
                });
                x += spec.tile;
            }
            y += spec.tile;
        }
        TilePlan { w, h, spec, tiles }
    }

    /// Number of tiles.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Always false — a plan over a non-empty image has at least one tile.
    /// Present because clippy asks for it next to `len`, and because a caller
    /// that guards on it is telling the reader the plan may be degenerate; it
    /// cannot be.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Tiles per row / per column.
    pub fn grid(&self) -> (u32, u32) {
        (self.w.div_ceil(self.spec.tile), self.h.div_ceil(self.spec.tile))
    }

    /// Total pixels read across all tiles, over the image's own pixel count —
    /// the compute overhead the halo costs. `1.0` means no overlap.
    pub fn overhead(&self) -> f32 {
        let read: u64 = self.tiles.iter().map(|t| t.src.area()).sum();
        read as f32 / (self.w as u64 * self.h as u64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The invariant the whole design rests on: cores are disjoint and cover
    /// every pixel exactly once. Checked by painting.
    fn assert_exact_cover(plan: &TilePlan) {
        let mut hits = vec![0u32; (plan.w * plan.h) as usize];
        for t in &plan.tiles {
            assert_eq!((t.keep.w, t.keep.h), (t.dst.w, t.dst.h), "keep and dst must match in size");
            // `keep` must lie inside `src`.
            assert!(t.keep.right() <= t.src.w && t.keep.bottom() <= t.src.h);
            // and map back to `dst` in image coords.
            assert_eq!(t.src.x + t.keep.x, t.dst.x);
            assert_eq!(t.src.y + t.keep.y, t.dst.y);
            for y in t.dst.y..t.dst.bottom() {
                for x in t.dst.x..t.dst.right() {
                    hits[(y * plan.w + x) as usize] += 1;
                }
            }
        }
        assert!(hits.iter().all(|&c| c == 1), "cores must cover every pixel exactly once");
    }

    #[test]
    fn small_image_is_a_single_whole_image_tile() {
        let p = TilePlan::new(100, 60, TileSpec::new(256, 16));
        assert_eq!(p.len(), 1);
        let t = p.tiles[0];
        assert_eq!(t.src, Rect::new(0, 0, 100, 60));
        assert_eq!(t.keep, Rect::new(0, 0, 100, 60));
        assert_eq!(t.dst, Rect::new(0, 0, 100, 60));
        assert!((p.overhead() - 1.0).abs() < 1e-6, "no halo work on a single tile");
        assert!(!p.is_empty());
    }

    #[test]
    fn cores_tile_the_image_exactly_including_ragged_edges() {
        for (w, h, tile, halo) in [(64, 64, 32, 8), (100, 70, 32, 8), (5, 5, 2, 3), (257, 129, 64, 0)] {
            let p = TilePlan::new(w, h, TileSpec::new(tile, halo));
            assert_exact_cover(&p);
            assert_eq!(p.grid(), (w.div_ceil(tile), h.div_ceil(tile)));
            assert_eq!(p.len(), (p.grid().0 * p.grid().1) as usize);
        }
    }

    #[test]
    fn halo_is_clipped_at_the_image_border() {
        let p = TilePlan::new(64, 64, TileSpec::new(32, 8));
        // Top-left tile: no halo above/left, 8 px below/right.
        assert_eq!(p.tiles[0].src, Rect::new(0, 0, 40, 40));
        assert_eq!(p.tiles[0].keep, Rect::new(0, 0, 32, 32));
        // Bottom-right tile: halo above/left, clipped below/right.
        let last = *p.tiles.last().unwrap();
        assert_eq!(last.src, Rect::new(24, 24, 40, 40));
        assert_eq!(last.keep, Rect::new(8, 8, 32, 32));
        assert_eq!(last.dst, Rect::new(32, 32, 32, 32));
    }

    #[test]
    fn zero_halo_makes_src_equal_dst() {
        let p = TilePlan::new(96, 96, TileSpec::new(32, 0));
        assert_eq!(p.len(), 9);
        for t in &p.tiles {
            assert_eq!(t.src, t.dst);
            assert_eq!(t.keep, Rect::new(0, 0, t.dst.w, t.dst.h));
        }
        assert!((p.overhead() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overhead_is_the_measured_read_amplification() {
        // 512-core, 32-halo over a 1024x1024 image: interior tiles read
        // 576x576 but border sides are clipped, so it is below the naive
        // (576/512)^2 = 1.266 and must still exceed 1.
        let p = TilePlan::new(1024, 1024, TileSpec::new(512, 32));
        assert_eq!(p.len(), 4);
        // Every tile is 544x544 (one side clipped each way), so the exact figure
        // is 4*544^2 / 1024^2 = 1.128906..., NOT the (576/512)^2 = 1.2656
        // interior ceiling. Pinned to the value, not to a band: the band would
        // still pass if the halo silently stopped being clipped at the border.
        assert!(p.tiles.iter().all(|t| t.src.w == 544 && t.src.h == 544));
        let o = p.overhead();
        let want = 4.0 * 544.0 * 544.0 / (1024.0 * 1024.0);
        assert!((o - want).abs() < 1e-6, "overhead {o}, expected {want}");
        assert!((o - 1.128_906_3).abs() < 1e-6, "overhead {o}");
        assert!(o < (576.0f32 / 512.0).powi(2), "must beat the interior ceiling");
    }
}
