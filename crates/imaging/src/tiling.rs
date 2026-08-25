// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tiling for images too large for one pass, plus the per-model VLM crop
//! policies - two different things that share a word, kept apart on purpose.
//!
//! **Halo tiling** (this module's first half, `TileSpec`/`TilePlan`) was
//! net-new: pixel-space tiling existed nowhere in the workspace.
//!
//! **VLM crop policies** (the second half, [`internvl_grid`] and
//! [`moondream_select_tiling`]) are the other concept: how many `tile x tile`
//! crops a vision-language model cuts an image into. `moondream`'s
//! `reconstruct_from_crops` stays in that crate - it works in **patch** units
//! against Moondream's own feature-map geometry - but its tile-count policy
//! lives here beside InternVL's, per this crate's own header. See the section
//! divider below for why they must never be merged.
//!
//! ## Halo tiles with disjoint cores - why there is no blending
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
//! variant - not as a host fallback here.
//!
//! ## Choosing `halo`
//!
//! `halo` must cover the model's receptive-field radius; anything less shows as
//! a visible grid. An **interior** tile costs `((tile + 2*halo)/tile)²` in
//! compute, so `tile = 512, halo = 32` costs `(576/512)²` while
//! `tile = 128, halo = 32` costs `(192/128)²` - the halo is a *fixed* ring,
//! so its relative cost grows as the square of `1 + 2*halo/tile`, not as its
//! area.
//!
//! That formula is the ceiling, not the answer: border tiles have their halo
//! clipped, so a whole plan always costs less. Do not quote it for a specific
//! image - ask [`TilePlan::overhead`], which sums the plan's actual `src` areas.
//! For 1024x1024 at `tile = 512, halo = 32` the real figure is well under the
//! `(576/512)²` ceiling, because every one of the four tiles is clipped on two
//! sides (both are pinned in
//! `overhead_is_the_measured_read_amplification`).

use crate::pixels::Rect;

/// Tiling parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSpec {
    /// Core edge length in pixels - the region each tile contributes to the
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
    /// The core **within `src`'s own coordinate frame** - what to crop out of
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

    /// Always false - a plan over a non-empty image has at least one tile.
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

    /// Total pixels read across all tiles, over the image's own pixel count -
    /// the compute overhead the halo costs. `1.0` means no overlap.
    pub fn overhead(&self) -> f32 {
        let read: u64 = self.tiles.iter().map(|t| t.src.area()).sum();
        read as f32 / (self.w as u64 * self.h as u64) as f32
    }
}

// ---------------------------------------------------------------------------
// VLM crop-tile-count policies - NAMED, never unified
// ---------------------------------------------------------------------------
//
// Everything above is HALO tiling: one image, one model, tiles that overlap on
// their input so a convolutional receptive field is satisfied. What follows is
// a different concept that shares the word: a VLM's choice of how many
// `tile x tile` crops to cut an image into before its vision tower sees them.
// There is no halo, no reconstruction and no receptive field - only a tile
// count and an aspect ratio.
//
// These are REFERENCE-MODEL CONTRACTS, one function per model, hosted side by
// side exactly as `crate`'s own module header sanctions ("They may be hosted
// here side by side as named policies, but they must never be unified").
// `moondream_select_tiling` solves a continuous optimisation and clamps;
// `internvl_grid` enumerates a discrete candidate set and picks the nearest
// aspect ratio. They disagree on real images, and that is correct - each
// reproduces its own reference. A caller picks the one its checkpoint was
// trained with; there is no "the" tiling policy to call.

/// InternVL / DeepSeek-OCR discrete candidate-ratio tiling: `(tiles_w, tiles_h)`.
///
/// Enumerates every grid `(i, j)` whose tile count `i*j` lies in
/// `[min_num, max_num]` (InternVL's default is `1..=12`, DeepSeek-OCR's
/// `2..=9`), and picks the one whose aspect ratio `i/j` is closest to the
/// image's `orig_w/orig_h`. Port of `find_closest_aspect_ratio` +
/// `dynamic_preprocess`'s candidate generation.
///
/// **Tiebreak - the reference's, made deterministic.** The reference builds its
/// candidate set with a Python `set` and sorts it by tile count alone, so the
/// order *within* one tile count is an implementation detail of the interpreter
/// rather than a specification. Two rules make it reproducible here, and both
/// are the reference's own:
///
/// 1. Candidates are enumerated in `(i*j, i, j)` ascending order - the
///    reference's "sorted by tile count" with a lexicographic secondary key
///    supplied where the reference had none.
/// 2. A strictly-closer ratio always wins. On an EXACT tie the later (larger, by
///    rule 1) candidate wins **only if** `orig_w*orig_h > 0.5 * tile^2 * i*j` -
///    the reference's own `area > 0.5 * image_size**2 * ratio[0] * ratio[1]`
///    guard, i.e. "only spend more tiles if the image has enough pixels to fill
///    at least half of them". Otherwise the earlier, coarser grid stands.
///
/// Rule 2 is what the reference actually specifies; rule 1 only fixes which
/// candidate is "earlier" when rule 2 has to compare two of the same tile
/// count. The returned grid always satisfies `min_num <= i*j <= max_num`.
pub fn internvl_grid(orig_w: u32, orig_h: u32, tile: u32, min_num: u32, max_num: u32) -> (u32, u32) {
    assert!(orig_w > 0 && orig_h > 0, "internvl_grid: image must be non-empty, got {orig_w}x{orig_h}");
    assert!(tile > 0, "internvl_grid: tile must be > 0");
    assert!(min_num >= 1 && min_num <= max_num, "internvl_grid: need 1 <= min_num <= max_num, got {min_num}..={max_num}");

    let aspect = orig_w as f64 / orig_h as f64;
    let area = orig_w as f64 * orig_h as f64;
    // `i*j <= max_num` bounds each factor by `max_num`, so this is the whole
    // candidate set, not a truncation of it.
    let mut cands: Vec<(u32, u32)> =
        (1..=max_num).flat_map(|i| (1..=max_num).map(move |j| (i, j))).filter(|&(i, j)| (min_num..=max_num).contains(&(i * j))).collect();
    cands.sort_by_key(|&(i, j)| (i * j, i, j)); // rule 1

    let mut best = (1u32, 1u32);
    let mut best_diff = f64::INFINITY;
    for (i, j) in cands {
        let diff = (aspect - i as f64 / j as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff && area > 0.5 * (tile as f64) * (tile as f64) * (i * j) as f64 {
            best = (i, j); // rule 2
        }
    }
    best
}

/// Moondream 3's continuous-ratio crop tiling: `(h_tiles, w_tiles)`.
///
/// Faithful port of `image_crops.py::select_tiling` - inputs are the
/// margin-subtracted pixel dims and the usable `crop_window_size`. Note the
/// return order is `(h, w)`, the reference's, and the OPPOSITE of
/// [`internvl_grid`]'s `(w, h)`: these are two models' contracts, not one API,
/// and normalising either would misreport what its reference does.
///
/// Unlike [`internvl_grid`] this enumerates nothing - it solves
/// `h*w <= max_crops` in the reals, floors, and clamps against the per-axis
/// minimum `ceil(dim / crop_size)`.
pub fn moondream_select_tiling(height: u32, width: u32, crop_size: u32, max_crops: u32) -> (u32, u32) {
    if height <= crop_size || width <= crop_size {
        return (1, 1);
    }
    let (h, w, cs, mc) = (height as f64, width as f64, crop_size as f64, max_crops as f64);
    let min_h = (h / cs).ceil();
    let min_w = (w / cs).ceil();
    if min_h * min_w > mc {
        let ratio = (mc / (min_h * min_w)).sqrt();
        return ((min_h * ratio).floor().max(1.0) as u32, (min_w * ratio).floor().max(1.0) as u32);
    }
    let mut h_tiles = (mc * h / w).sqrt().floor().max(min_h);
    let mut w_tiles = (mc * w / h).sqrt().floor().max(min_w);
    if h_tiles * w_tiles > mc {
        if w_tiles > h_tiles {
            w_tiles = (mc / h_tiles).floor();
        } else {
            h_tiles = (mc / w_tiles).floor();
        }
    }
    (h_tiles.max(1.0) as u32, w_tiles.max(1.0) as u32)
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

    // ---- internvl_grid ----------------------------------------------------

    /// Real image sizes, each with the reasoning that fixes the answer.
    #[test]
    fn internvl_grid_matches_expected_grids() {
        // 2:1 landscape, tile 500: (2,1) is an EXACT ratio match and the only
        // exact one under max_num=6 ((4,2) is 8 tiles, (6,3) is 18).
        assert_eq!(internvl_grid(1000, 500, 500, 1, 6), (2, 1));
        // 3:4 portrait: (3,4) is exact and reachable at max_num=12.
        assert_eq!(internvl_grid(768, 1024, 448, 1, 12), (3, 4));
        // A 1.5:1 page scan: (3,2) is the exact ratio, 6 tiles, inside 2..=9.
        assert_eq!(internvl_grid(1536, 1024, 512, 2, 9), (3, 2));
    }

    /// A square image: the ratio is exact at every `(k,k)`, so the AREA rule
    /// alone decides how many tiles to spend - and it must not spend tiles the
    /// image cannot fill.
    #[test]
    fn internvl_grid_square_image_spends_tiles_only_when_the_area_justifies_it() {
        // 1024^2 with tile 512: (2,2) needs area > 0.5*512^2*4 = 524288 (yes,
        // 1048576), (3,3) needs > 1179648 (no). So 2x2.
        assert_eq!(internvl_grid(1024, 1024, 512, 1, 12), (2, 2));
        // Same image, tile 1024: even (2,2) needs > 2097152 -- more than the
        // image has -- so the single tile stands.
        assert_eq!(internvl_grid(1024, 1024, 1024, 1, 12), (1, 1));
    }

    /// An extreme aspect ratio cannot be matched, so the policy must saturate
    /// at the most extreme grid available rather than fall back to square.
    #[test]
    fn internvl_grid_extreme_aspect_saturates() {
        // 16:1: the widest grid at max_num=12 is (12,1), ratio 12.
        assert_eq!(internvl_grid(4096, 256, 512, 1, 12), (12, 1));
        // and the transpose, to prove no axis is privileged.
        assert_eq!(internvl_grid(256, 4096, 512, 1, 12), (1, 12));
    }

    /// The documented tiebreak, exercised on an EXACT tie in both directions.
    ///
    /// `min_num == max_num == 4` leaves exactly `(1,4)`, `(2,2)`, `(4,1)`. At
    /// aspect 0.625 the first two are equidistant (|0.625-0.25| = |0.625-1.0| =
    /// 0.375), so only the area rule separates them - and it must do so
    /// deterministically, both ways.
    #[test]
    fn internvl_grid_tiebreak_is_the_documented_area_rule() {
        // area = 625*1000 = 625000 > 0.5*100^2*4 = 20000 -> the later (larger
        // by the (i*j, i, j) order) candidate (2,2) takes it.
        assert_eq!(internvl_grid(625, 1000, 100, 4, 4), (2, 2));
        // Same tie, but 0.5*1000^2*4 = 2000000 > 625000 -> the earlier (1,4)
        // stands. Same inputs but for `tile`, opposite winner: the rule is real,
        // not incidental ordering.
        assert_eq!(internvl_grid(625, 1000, 1000, 4, 4), (1, 4));
        // Deterministic: same inputs, same answer, every call.
        for _ in 0..8 {
            assert_eq!(internvl_grid(625, 1000, 100, 4, 4), (2, 2));
        }
    }

    /// The tile-count budget is a hard contract, not a preference.
    #[test]
    fn internvl_grid_always_respects_the_tile_budget() {
        for (w, h) in [(1u32, 1u32), (37, 4001), (4001, 37), (1920, 1080), (1000, 1000)] {
            for (lo, hi) in [(1u32, 1u32), (1, 12), (2, 9), (4, 4), (6, 40)] {
                let (i, j) = internvl_grid(w, h, 448, lo, hi);
                assert!((lo..=hi).contains(&(i * j)), "{w}x{h} in {lo}..={hi} gave {i}x{j} = {} tiles", i * j);
            }
        }
    }

    // ---- moondream_select_tiling (moved here from moondream3::preprocess) ----

    #[test]
    fn moondream_select_tiling_matches_reference() {
        // height/width <= crop -> single tile.
        assert_eq!(moondream_select_tiling(300, 300, 378, 12), (1, 1));
        // Square large image -> balanced tiling under budget.
        let (h, w) = moondream_select_tiling(1000, 1000, 266, 12);
        assert!(h * w <= 12 && h >= 1 && w >= 1);
        assert_eq!((h, w), (3, 3));
        // Wide image -> more horizontal tiles.
        let (h2, w2) = moondream_select_tiling(400, 1600, 266, 12);
        assert!(w2 >= h2 && h2 * w2 <= 12);
    }

    /// The two policies are NOT interchangeable - pinned so a future
    /// "simplification" that routes one through the other fails loudly.
    ///
    /// They DO agree on many sizes (1000x1000, 1024x768, 1920x1080 at crop 266
    /// / budget 12 all give the same grid), which is exactly why this needs a
    /// pinned counterexample rather than a spot check: moondream maximises tile
    /// usage in the reals, InternVL picks the nearest DISCRETE ratio and then
    /// refuses to spend tiles the image cannot half-fill.
    #[test]
    fn the_two_named_policies_disagree_and_must_stay_separate() {
        // 1600x400 (4:1), crop/tile 266, budget 12. Moondream: min_w =
        // ceil(1600/266) = 7 > budget/1, so it scales down to 1x6. InternVL:
        // the nearest candidate ratio to 4.0 is (4,1) exactly -- and (8,2),
        // also 4.0, is over budget. 6x1 vs 4x1.
        assert_eq!(moondream_select_tiling(400, 1600, 266, 12), (1, 6)); // (h, w)
        assert_eq!(internvl_grid(1600, 400, 266, 1, 12), (4, 1)); // (w, h)
        // 900x600 (3:2): moondream spends its whole 12-tile budget (4x3);
        // InternVL takes the exact 3:2 ratio at 6 tiles and the area rule
        // (900*600 = 540000 vs 0.5*266^2*12 = 424536... at (6,4)=24 tiles,
        // over budget anyway) never promotes it further.
        assert_eq!(moondream_select_tiling(600, 900, 266, 12), (3, 4)); // (h, w)
        assert_eq!(internvl_grid(900, 600, 266, 1, 12), (3, 2)); // (w, h)
    }
}
