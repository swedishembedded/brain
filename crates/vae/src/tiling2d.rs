// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overlapping-tile geometry for the **2-D image autoencoder** - the host-side
//! splits, output mappings and trapezoidal blend masks a tiled `[C, H, W]`
//! encode/decode needs so an image larger than one card's VRAM can be
//! processed one tile at a time.
//!
//! Swedish Embedded AB implements memory-bounded tiled inference for image
//! autoencoders for its clients. If your team needs expertise in fitting large
//! generative image models onto fixed edge hardware, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # This module is a projection of [`crate::tiling3d`], not a second tiler
//!
//! Every formula this needs - the trapezoidal mask, the interval split, the
//! per-axis outer-product divisor, the weighted accumulation - already exists
//! next door for the video autoencoder, transcribed there from the LTX-2
//! reference and gated by its own unit tests. So the plan here IS a
//! [`crate::tiling3d::TilePlan3d`] whose temporal axis is a single cell, and
//! the blender IS [`crate::tiling3d::Blender`] over an `F = 1` volume, where
//! `[C, 1, H, W]` and `[C, H, W]` are the same bytes in the same order. What
//! this module adds is the two things a 2-D image VAE genuinely has and a
//! video decoder does not:
//!
//! * a **down-scaling** axis mapping ([`map_spatial_down`]). A decode's tile
//!   is split on the latent grid and lands on the pixel grid (`x scale`,
//!   which is [`crate::tiling3d::map_spatial`]); an ENCODE's tile is the
//!   mirror - it reads a pixel range and lands on the latent grid, so the
//!   blend mask has to be built at the latent resolution the output is
//!   accumulated at. Reusing the up-scaling mapping for both would blend at
//!   the wrong resolution and only show up as a seam.
//! * the two-axis cartesian product and the 2-D accessors, so a caller writes
//!   `tile.h` / `tile.w` rather than carrying a degenerate frame index
//!   through every loop.
//!
//! # Why the blend, rather than `imaging::tiling`'s halo-and-crop
//!
//! Same reason [`crate::tiling3d`] gives: a halo large enough to cover this
//! conv stack's receptive field, plus a mid-block self-attention that is
//! global over whatever extent it is handed, is not a halo that still saves
//! memory. A tile's interior is exact and its seam is a weighted average of
//! two tiles that saw different context; the trapezoidal ramp makes that a
//! gradient rather than an edge. Upstream (`diffusers`' own `tiled_decode`)
//! blends for exactly this reason.

use crate::tiling3d::{split_by_size, trapezoidal_mask_1d, AxisPlan, AxisTile, Blender, Interval, Tile3d, TilePlan3d};

/// Map an interval of the **output** (latent) grid to the pixel range an
/// ENCODE tile must read, with the blend mask at the OUTPUT resolution.
///
/// The mirror of [`crate::tiling3d::map_spatial`]: there the split is on the
/// small grid and the result lands on the large one, here the split is on the
/// small grid and the INPUT is the large one. `src` is therefore
/// `interval x scale` pixels and `dst` the interval itself, and the mask is
/// `interval`-long because that is what gets accumulated.
pub fn map_spatial_down(iv: Interval, scale: usize) -> AxisTile {
    let mask = trapezoidal_mask_1d(iv.len(), iv.left_ramp, iv.right_ramp, false);
    AxisTile { src: (iv.start * scale, iv.end * scale), dst: (iv.start, iv.end), mask }
}

/// The degenerate frame axis that makes a 2-D plan a [`TilePlan3d`]: one cell,
/// one tile, weight exactly 1, so it multiplies through every mask and divisor
/// without changing a value.
fn one_frame() -> AxisPlan {
    AxisPlan::new(vec![AxisTile { src: (0, 1), dst: (0, 1), mask: vec![1.0] }], 1)
}

/// A complete overlapping-tile cover of a 2-D `[C, H, W]` plane and the plane
/// it maps to.
///
/// Built either way round: [`TilePlan2d::decode`] splits the latent grid and
/// lands on pixels, [`TilePlan2d::encode`] splits the latent grid and reads
/// pixels. Both split on the LATENT grid, because that is the grid a VAE's
/// geometry is quantised to - a tile whose pixel extent is not a whole number
/// of latent cells cannot be run by either graph.
#[derive(Clone, Debug, PartialEq)]
pub struct TilePlan2d {
    inner: TilePlan3d,
}

/// One tile of a [`TilePlan2d`] - a borrow of the two axis tiles whose product
/// it is. `src` is the range to read on the input grid, `dst` the range it
/// lands on, and `mask` the blend weights over `dst`.
#[derive(Clone, Copy, Debug)]
pub struct Tile2d<'a> {
    pub h: &'a AxisTile,
    pub w: &'a AxisTile,
    /// The plan's degenerate frame axis, so [`Blender2d::add`] can hand this
    /// tile to the 3-D blender unchanged.
    t: &'a AxisTile,
}

impl TilePlan2d {
    /// The DECODE cover: split a `lh x lw` latent into tiles of at most
    /// `tile` cells sharing `overlap`, each landing on `x scale` pixels.
    pub fn decode(lh: usize, lw: usize, tile: usize, overlap: usize, scale: usize) -> TilePlan2d {
        TilePlan2d {
            inner: TilePlan3d {
                t: one_frame(),
                h: AxisPlan::spatial(lh, tile, overlap, scale),
                w: AxisPlan::spatial(lw, tile, overlap, scale),
            },
        }
    }

    /// The ENCODE cover: the same split of the `lh x lw` LATENT grid the
    /// encode produces, with each tile reading `x scale` pixels of the image
    /// and blending at latent resolution.
    pub fn encode(lh: usize, lw: usize, tile: usize, overlap: usize, scale: usize) -> TilePlan2d {
        let axis = |len: usize| {
            let tiles: Vec<AxisTile> =
                split_by_size(tile, overlap, len).into_iter().map(|iv| map_spatial_down(iv, scale)).collect();
            AxisPlan::new(tiles, len)
        };
        TilePlan2d { inner: TilePlan3d { t: one_frame(), h: axis(lh), w: axis(lw) } }
    }

    pub fn h(&self) -> &AxisPlan {
        &self.inner.h
    }

    pub fn w(&self) -> &AxisPlan {
        &self.inner.w
    }

    /// Every tile, row-major (height axis slowest).
    pub fn tiles(&self) -> Vec<Tile2d<'_>> {
        let t = &self.inner.t.tiles[0];
        let mut out = Vec::with_capacity(self.inner.h.len() * self.inner.w.len());
        for h in &self.inner.h.tiles {
            for w in &self.inner.w.tiles {
                out.push(Tile2d { h, w, t });
            }
        }
        out
    }

    /// Output plane `(height, width)`.
    pub fn out_shape(&self) -> (usize, usize) {
        (self.inner.h.out_len, self.inner.w.out_len)
    }

    /// `processed / unique` output area - the redundant work the overlap
    /// costs, `>= 1`. `1.0` means no overlap at all.
    pub fn overlap_waste(&self) -> f64 {
        self.inner.overlap_waste()
    }

    /// True when both axes' masks partition unity to within `1e-5`.
    /// Informational: the blend divides by the accumulated weight regardless.
    pub fn masks_are_complementary(&self) -> bool {
        self.inner.masks_are_complementary()
    }

    /// The distinct INPUT shapes this cover contains, each with the tiles that
    /// have it, in a deterministic order.
    ///
    /// A caller builds one device graph per SHAPE, not per tile: a
    /// `split_by_size` cover has at most four of them (interior, short last
    /// row, short last column, and their corner) however many tiles it has.
    /// `BTreeMap` rather than a hash map because a decode that reorders its
    /// own float accumulation is a decode whose output is not reproducible.
    pub fn by_src_shape(&self) -> std::collections::BTreeMap<(usize, usize), Vec<usize>> {
        let mut out: std::collections::BTreeMap<(usize, usize), Vec<usize>> = std::collections::BTreeMap::new();
        for (i, t) in self.tiles().iter().enumerate() {
            out.entry((t.h.src_len(), t.w.src_len())).or_default().push(i);
        }
        out
    }

    /// The largest input tile shape in the cover - what a peak-memory estimate
    /// for this plan is priced from.
    pub fn max_src_shape(&self) -> (usize, usize) {
        let tiles = self.tiles();
        let h = tiles.iter().map(|t| t.h.src_len()).max().unwrap_or(0);
        let w = tiles.iter().map(|t| t.w.src_len()).max().unwrap_or(0);
        (h, w)
    }
}

/// Accumulates masked output tiles into one `[C, H, W]` plane and divides by
/// the separable blend weights on [`Blender2d::finish`].
///
/// Host-side by construction: the whole point of tiling is that the full plane
/// does not fit on the device, so the accumulator lives in RAM and each tile's
/// device resources are released before the next tile's are created.
pub struct Blender2d(Blender);

impl Blender2d {
    pub fn new(plan: &TilePlan2d, channels: usize) -> Blender2d {
        Blender2d(Blender::new(&plan.inner, channels))
    }

    /// Add one tile's result, laid out `[C, th, tw]` in the same row-major
    /// order the accumulator uses, scaled by the tile's separable mask.
    pub fn add(&mut self, tile: Tile2d<'_>, values: &[f32]) {
        self.0.add(Tile3d { t: tile.t, h: tile.h, w: tile.w }, values);
    }

    /// Divide out the accumulated blend weight and take the result.
    pub fn finish(self) -> Vec<f32> {
        self.0.finish()
    }
}

/// Cut one tile's `[C, th, tw]` sub-plane out of a full `[C, h, w]` plane,
/// reading the tile's INPUT range.
pub fn slice_src(plane: &[f32], channels: usize, (h, w): (usize, usize), tile: Tile2d<'_>) -> Vec<f32> {
    assert_eq!(plane.len(), channels * h * w, "slice_src: plane has {} values, expected {}", plane.len(), channels * h * w);
    let (h0, h1) = tile.h.src;
    let (w0, w1) = tile.w.src;
    let (th, tw) = (h1 - h0, w1 - w0);
    let mut out = vec![0.0f32; channels * th * tw];
    for ci in 0..channels {
        for hi in 0..th {
            let src = (ci * h + h0 + hi) * w + w0;
            let dst = (ci * th + hi) * tw;
            out[dst..dst + tw].copy_from_slice(&plane[src..src + tw]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blend's own correctness, isolated from any autoencoder: cut a known
    /// plane into the plan's tiles, feed the pieces back through
    /// [`Blender2d`], and require the stitched result to equal the original.
    ///
    /// This is the gate a VAE-based comparison CANNOT be, because a real conv
    /// stack's receptive field is wider than the overlap and so its tiles
    /// genuinely disagree in the seam. Here the "model" is the identity, so
    /// any deviation is a mask, slice or divisor bug and nothing else.
    #[test]
    fn the_blend_reconstructs_a_known_plane_exactly() {
        for plan in [TilePlan2d::decode(34, 60, 14, 2, 2), TilePlan2d::encode(34, 60, 14, 2, 8)] {
            assert!(plan.tiles().len() >= 9, "expected a genuinely split cover");
            assert!(plan.masks_are_complementary());

            let (h, w) = plan.out_shape();
            let c = 2usize;
            // Structure on every axis, so a swapped or off-by-one slice cannot
            // cancel out.
            let val = |ci: usize, hi: usize, wi: usize| ((ci * 7 + hi) as f32 * 0.37).sin() + (wi as f32 * 0.011).cos();
            let mut whole = vec![0.0f32; c * h * w];
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        whole[(ci * h + hi) * w + wi] = val(ci, hi, wi);
                    }
                }
            }

            let mut b = Blender2d::new(&plan, c);
            for tile in plan.tiles() {
                let (th, tw) = (tile.h.dst_len(), tile.w.dst_len());
                let mut piece = vec![0.0f32; c * th * tw];
                for ci in 0..c {
                    for hi in 0..th {
                        for wi in 0..tw {
                            piece[(ci * th + hi) * tw + wi] = val(ci, tile.h.dst.0 + hi, tile.w.dst.0 + wi);
                        }
                    }
                }
                b.add(tile, &piece);
            }
            let got = b.finish();
            let worst = got.iter().zip(&whole).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 1e-5, "blend reconstruction worst |delta| = {worst}");
        }
    }

    /// A cover that does not split is the identity, to the bit - the property
    /// the automatic size threshold rests on.
    #[test]
    fn a_single_tile_plan_blends_to_the_identity() {
        let plan = TilePlan2d::decode(8, 8, 64, 16, 8);
        assert_eq!(plan.tiles().len(), 1);
        assert_eq!(plan.overlap_waste(), 1.0);
        assert_eq!(plan.max_src_shape(), (8, 8));
        let (h, w) = plan.out_shape();
        let src: Vec<f32> = (0..3 * h * w).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut b = Blender2d::new(&plan, 3);
        b.add(plan.tiles()[0], &src);
        assert_eq!(b.finish(), src, "an untiled plan must be bit-identical");
    }

    /// The two directions are mirrors: a decode tile reads latent and writes
    /// pixels, an encode tile reads pixels and writes latent, and the encode
    /// blend happens at LATENT resolution (the mask is as long as `dst`).
    #[test]
    fn the_encode_cover_reads_pixels_and_blends_on_the_latent_grid() {
        let dec = TilePlan2d::decode(32, 32, 16, 4, 8);
        let enc = TilePlan2d::encode(32, 32, 16, 4, 8);
        assert_eq!(dec.out_shape(), (256, 256));
        assert_eq!(enc.out_shape(), (32, 32));
        let (d, e) = (dec.tiles(), enc.tiles());
        assert_eq!(d.len(), e.len(), "the same split, both directions");
        for (d, e) in d.iter().zip(&e) {
            // The decode's latent read range is the encode's latent write
            // range, and the decode's pixel write range is the encode's pixel
            // read range.
            assert_eq!(d.h.src, e.h.dst);
            assert_eq!(d.h.dst, e.h.src);
            assert_eq!(e.h.mask.len(), e.h.dst_len());
        }
        assert!(enc.masks_are_complementary());
    }

    /// Peak memory is what tiling is for, so the shape a graph gets built at
    /// must be the tile's, never the image's - and a cover has a handful of
    /// distinct shapes however many tiles it has.
    #[test]
    fn a_cover_has_at_most_four_distinct_input_shapes() {
        let plan = TilePlan2d::decode(256, 256, 64, 16, 8);
        assert!(plan.tiles().len() >= 25, "expected a large cover, got {}", plan.tiles().len());
        assert!(plan.by_src_shape().len() <= 4, "shapes: {:?}", plan.by_src_shape().keys().collect::<Vec<_>>());
        assert_eq!(plan.max_src_shape(), (64, 64));
        assert_eq!(plan.by_src_shape().values().map(Vec::len).sum::<usize>(), plan.tiles().len());
    }
}
