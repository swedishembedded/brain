// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Overlapping-tile geometry for **3D causal video autoencoders** - the
//! host-side splits, output mappings and trapezoidal blend masks a tiled
//! `[C, T, H, W]` encode/decode needs so a clip larger than one card's VRAM
//! can be processed one tile at a time.
//!
//! Swedish Embedded AB implements memory-bounded tiled inference for video
//! autoencoders for its clients. If your team needs expertise in fitting
//! large generative video models onto fixed edge hardware, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! # This is a port, not an invention
//!
//! Every formula below is transcribed from the LTX-2 reference
//! (`resources/ltxv/source/packages/ltx-core/src/ltx_core/`):
//! [`trapezoidal_mask_1d`] from `tiling.py::compute_trapezoidal_mask_1d`,
//! [`split_by_size`] / [`split_temporal_causal`] from the like-named
//! functions there, and [`map_spatial`] / [`map_temporal`] from
//! `model/video_vae/video_vae.py::map_spatial_slice` / `map_temporal_slice`.
//! The blend is **linear-ramp trapezoidal**, and the temporal ramp uses a
//! *different* start convention from the spatial one - reading that off the
//! reference rather than assuming one shape for both is the whole point of
//! this module (`left_starts_from_0=True` for time, `False` for space; see
//! [`trapezoidal_mask_1d`]).
//!
//! # Why not `imaging::tiling`
//!
//! `imaging::tiling` is the workspace's other tiler and it is a genuinely
//! different concept, not a 2D version of this one. It covers an image with
//! **disjoint cores grown by a halo**, and its own module doc says why it does
//! not blend: each output pixel comes from exactly one tile, so there is no
//! partition-of-unity question and no double-counting. That works when the
//! halo can cover the consumer's receptive field. It cannot here - this
//! decoder's spatial receptive field is ~15 latent cells wide and a 1080p
//! latent is only 34 cells tall, so a halo large enough to be exact is larger
//! than the image. Blended overlap is what the reference ships for exactly
//! that reason, and blending is a different accumulation contract (weighted
//! sum over a divisor), not a parameter of the halo one.
//!
//! # The tiling is separable, and so is its divisor
//!
//! Tiles are the full cartesian product of the per-axis splits, and each
//! tile's mask is the outer product of its three 1-D axis masks. That makes
//! the accumulated weight at any output cell factor exactly:
//!
//! ```text
//! W(t,h,w) = sum_tiles mt(t)*mh(h)*mw(w) = Wt(t) * Wh(h) * Ww(w)
//! ```
//!
//! so the blend divisor is three 1-D vectors ([`AxisPlan::weights`]), never a
//! dense `[T,H,W]` buffer. The reference has both forms - a per-axis
//! complementarity check (`masks_are_complementary`) that lets it *skip* the
//! divisor, and a dense `compute_summed_weights` for when it cannot. This
//! module always divides: when the masks do partition unity the divisor is
//! exactly `1.0` and the division is a no-op on the bit pattern, and when
//! they do not (a short final tile clamps its own ramp, which the reference
//! permits) dividing is the correct answer rather than a silently unnormalised
//! seam. One path, no branch to get wrong.

/// One tile's extent along one axis of the **input** (latent) grid, with the
/// ramp lengths it shares with its neighbours.
///
/// `left_ramp` / `right_ramp` are the overlap widths with the previous / next
/// interval. The first interval has `left_ramp == 0` and the last has
/// `right_ramp == 0`; an untiled axis has both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub start: usize,
    pub end: usize,
    pub left_ramp: usize,
    pub right_ramp: usize,
}

impl Interval {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// A 1-D trapezoidal blend mask with linear ramps
/// (`ltx_core.tiling.compute_trapezoidal_mask_1d`).
///
/// `left_starts_from_0` picks which of two ramp conventions the fade-in uses,
/// and the two are NOT interchangeable:
///
/// * `false` (spatial axes) - the ramp is `i/(r+1)` for `i` in `1..=r`, so it
///   never reaches 0. Paired with the previous tile's fade-out
///   `(r+1-i)/(r+1)` over the same `r` cells, the two sum to exactly 1.
/// * `true` (the temporal axis) - the ramp is `i/r` for `i` in `0..r`, so it
///   *does* start at 0. This is the "sacrificial first sample" a causal
///   temporal split needs: [`split_temporal_causal`] gives the later tile one
///   extra leading cell whose value is discarded, and the previous tile's
///   fade-out is one cell shorter to match. The pair still sums to 1, but
///   only because both halves of that asymmetry are applied together.
///
/// Ramps are clamped to `length` and applied MULTIPLICATIVELY, so a tile
/// shorter than `left_ramp + right_ramp` gets the product of both ramps
/// rather than one overwriting the other - the reference's own `*=`.
pub fn trapezoidal_mask_1d(length: usize, ramp_left: usize, ramp_right: usize, left_starts_from_0: bool) -> Vec<f32> {
    assert!(length > 0, "trapezoidal_mask_1d: length must be positive");
    let ramp_left = ramp_left.min(length);
    let ramp_right = ramp_right.min(length);
    let mut mask = vec![1.0f32; length];

    if ramp_left > 0 {
        // `linspace(0, 1, n)[:-1]`, then `[1:]` unless the ramp starts at 0.
        let n = if left_starts_from_0 { ramp_left + 1 } else { ramp_left + 2 };
        let skip = usize::from(!left_starts_from_0);
        for (i, m) in mask.iter_mut().enumerate().take(ramp_left) {
            *m *= (i + skip) as f32 / (n - 1) as f32;
        }
    }
    if ramp_right > 0 {
        // `linspace(1, 0, ramp_right + 2)[1:-1]`.
        let n = ramp_right + 2;
        for j in 0..ramp_right {
            mask[length - ramp_right + j] *= 1.0 - (j + 1) as f32 / (n - 1) as f32;
        }
    }
    for v in &mut mask {
        *v = v.clamp(0.0, 1.0);
    }
    mask
}

/// Split one axis of length `length` into overlapping intervals of `size`
/// sharing `overlap` cells (`ltx_core.tiling.split_by_size`).
///
/// The last interval may be shorter than `size` when the axis does not divide
/// evenly; the reference's optional `min_tile_size` last-tile growth is NOT
/// ported (no call path in this workspace passes it, and a defaulted
/// almost-never-taken branch is a silent opt-out - lesson #30).
pub fn split_by_size(size: usize, overlap: usize, length: usize) -> Vec<Interval> {
    assert!(size > 0, "split_by_size: size must be > 0");
    assert!(overlap < size, "split_by_size: overlap {overlap} must be < size {size}");
    if length <= size {
        return vec![Interval { start: 0, end: length, left_ramp: 0, right_ramp: 0 }];
    }
    let stride = size - overlap;
    let amount = (length + size - 2 * overlap - 1) / stride;
    let mut out = Vec::with_capacity(amount);
    out.push(Interval { start: 0, end: size, left_ramp: 0, right_ramp: overlap });
    for i in 1..amount - 1 {
        out.push(Interval { start: i * stride, end: i * stride + size, left_ramp: overlap, right_ramp: overlap });
    }
    out.push(Interval { start: (amount - 1) * stride, end: length, left_ramp: overlap, right_ramp: 0 });
    out
}

/// [`split_by_size`] with the causal temporal adjustment
/// (`ltx_core.tiling.split_temporal_causal`): every interval after the first
/// starts one cell EARLIER and carries one more cell of left ramp.
///
/// That extra cell is what makes the pixel-space masks complementary once
/// [`map_temporal`] has expanded them: a temporal interval maps to
/// `1 + (len-1)*scale` pixel frames rather than `len*scale`, so without the
/// shift the previous tile's fade-out and the next tile's fade-in would be
/// one frame apart.
pub fn split_temporal_causal(size: usize, overlap: usize, length: usize) -> Vec<Interval> {
    if length <= size {
        return vec![Interval { start: 0, end: length, left_ramp: 0, right_ramp: 0 }];
    }
    let mut out = split_by_size(size, overlap, length);
    for iv in out.iter_mut().skip(1) {
        iv.start -= 1;
        iv.left_ramp += 1;
    }
    out
}

/// One axis tile: where it is read from on the input grid, where its result
/// lands on the output grid, and the 1-D blend mask over that output range.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisTile {
    /// Half-open range on the INPUT (latent) axis.
    pub src: (usize, usize),
    /// Half-open range on the OUTPUT (pixel) axis.
    pub dst: (usize, usize),
    /// Blend weights over `dst`, one per output cell.
    pub mask: Vec<f32>,
}

impl AxisTile {
    pub fn src_len(&self) -> usize {
        self.src.1 - self.src.0
    }

    pub fn dst_len(&self) -> usize {
        self.dst.1 - self.dst.0
    }
}

/// Map a latent interval to its pixel range on a SPATIAL axis
/// (`video_vae.py::map_spatial_slice`): a plain `x scale` on every quantity.
pub fn map_spatial(iv: Interval, scale: usize) -> AxisTile {
    let dst = (iv.start * scale, iv.end * scale);
    let mask = trapezoidal_mask_1d(dst.1 - dst.0, iv.left_ramp * scale, iv.right_ramp * scale, false);
    AxisTile { src: (iv.start, iv.end), dst, mask }
}

/// Map a latent interval to its pixel range on the **temporal** axis
/// (`video_vae.py::map_temporal_slice`).
///
/// Not a plain `x scale`: this VAE's frame rule is `F = 1 + scale*(T-1)`, so
/// the end maps to `1 + (end-1)*scale` and the left ramp to
/// `1 + (left_ramp-1)*scale` (the `+1` is the causal cell
/// [`split_temporal_causal`] added). The right ramp IS a plain `x scale`, one
/// cell shorter than the overlap it faces - which is why the temporal mask
/// starts its fade-in at exactly 0 (`left_starts_from_0 = true`).
pub fn map_temporal(iv: Interval, scale: usize) -> AxisTile {
    let dst = (iv.start * scale, 1 + (iv.end - 1) * scale);
    let left = if iv.left_ramp == 0 { 0 } else { 1 + (iv.left_ramp - 1) * scale };
    let mask = trapezoidal_mask_1d(dst.1 - dst.0, left, iv.right_ramp * scale, true);
    AxisTile { src: (iv.start, iv.end), dst, mask }
}

/// Every tile along one axis, plus the accumulated blend weight per output
/// cell - the divisor's factor for this axis.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisPlan {
    pub tiles: Vec<AxisTile>,
    /// Length of the output axis.
    pub out_len: usize,
}

impl AxisPlan {
    /// Build a plan from already-mapped tiles.
    pub fn new(tiles: Vec<AxisTile>, out_len: usize) -> AxisPlan {
        assert!(!tiles.is_empty(), "AxisPlan: at least one tile");
        assert_eq!(tiles.last().expect("non-empty").dst.1, out_len, "AxisPlan: tiles must cover the output axis");
        AxisPlan { tiles, out_len }
    }

    /// A spatial axis: `split_by_size` on the latent grid, `map_spatial` out.
    pub fn spatial(lat_len: usize, tile: usize, overlap: usize, scale: usize) -> AxisPlan {
        let tiles: Vec<AxisTile> = split_by_size(tile, overlap, lat_len).into_iter().map(|iv| map_spatial(iv, scale)).collect();
        AxisPlan::new(tiles, lat_len * scale)
    }

    /// The temporal axis: `split_temporal_causal` in, `map_temporal` out.
    pub fn temporal(lat_len: usize, tile: usize, overlap: usize, scale: usize) -> AxisPlan {
        let tiles: Vec<AxisTile> = split_temporal_causal(tile, overlap, lat_len).into_iter().map(|iv| map_temporal(iv, scale)).collect();
        AxisPlan::new(tiles, 1 + (lat_len - 1) * scale)
    }

    /// This axis's factor of the separable blend divisor: the sum of every
    /// tile's mask, laid out over the output axis. Exactly `1.0` everywhere
    /// when the masks partition unity.
    pub fn weights(&self) -> Vec<f32> {
        let mut w = vec![0.0f32; self.out_len];
        for t in &self.tiles {
            for (i, m) in t.mask.iter().enumerate() {
                w[t.dst.0 + i] += *m;
            }
        }
        w
    }

    /// Largest deviation of [`AxisPlan::weights`] from 1 - zero when the
    /// masks are a partition of unity. The reference's
    /// `masks_are_complementary` is this compared against `1e-5`.
    pub fn unity_error(&self) -> f32 {
        self.weights().iter().map(|w| (w - 1.0).abs()).fold(0.0, f32::max)
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

/// A complete overlapping-tile cover of a `[T, H, W]` latent volume and the
/// `[F, H*, W*]` pixel volume it decodes to.
///
/// Tiles are the cartesian product of the three axis plans; iterate with
/// [`TilePlan3d::tiles`].
#[derive(Clone, Debug, PartialEq)]
pub struct TilePlan3d {
    pub t: AxisPlan,
    pub h: AxisPlan,
    pub w: AxisPlan,
}

/// One tile of a [`TilePlan3d`] - a borrow of the three axis tiles whose
/// product it is.
#[derive(Clone, Copy, Debug)]
pub struct Tile3d<'a> {
    pub t: &'a AxisTile,
    pub h: &'a AxisTile,
    pub w: &'a AxisTile,
}

impl TilePlan3d {
    /// Every tile, temporal axis slowest (the order the reference's own
    /// `itertools.product` produces, and the order a temporal-streaming
    /// consumer wants).
    pub fn tiles(&self) -> Vec<Tile3d<'_>> {
        let mut out = Vec::with_capacity(self.t.len() * self.h.len() * self.w.len());
        for t in &self.t.tiles {
            for h in &self.h.tiles {
                for w in &self.w.tiles {
                    out.push(Tile3d { t, h, w });
                }
            }
        }
        out
    }

    /// Output volume `(frames, height, width)`.
    pub fn out_shape(&self) -> (usize, usize, usize) {
        (self.t.out_len, self.h.out_len, self.w.out_len)
    }

    /// `processed / unique` output volume - the redundant work overlap costs,
    /// `>= 1` (`diffusion_tiling.py::volumetric_overlap_waste`). `1.0` means
    /// no overlap at all.
    pub fn overlap_waste(&self) -> f64 {
        let processed: usize = self.tiles().iter().map(|t| t.t.dst_len() * t.h.dst_len() * t.w.dst_len()).sum();
        let (f, h, w) = self.out_shape();
        processed as f64 / (f * h * w).max(1) as f64
    }

    /// True when every axis's masks partition unity to within `1e-5` - the
    /// reference's `masks_are_complementary`. Informational here: the blend
    /// divides by [`AxisPlan::weights`] regardless.
    pub fn masks_are_complementary(&self) -> bool {
        self.t.unity_error() <= 1e-5 && self.h.unity_error() <= 1e-5 && self.w.unity_error() <= 1e-5
    }
}

/// Accumulates masked pixel tiles into one `[C, F, H, W]` volume and divides
/// by the separable blend weights on [`Blender::finish`].
///
/// Host-side by construction: the whole point of tiling is that the full
/// pixel volume does not fit on the device, so the accumulator lives in RAM
/// and each tile's device resources are released before the next tile's are
/// created.
pub struct Blender {
    c: usize,
    f: usize,
    h: usize,
    w: usize,
    acc: Vec<f32>,
    wt: Vec<f32>,
    wh: Vec<f32>,
    ww: Vec<f32>,
}

impl Blender {
    pub fn new(plan: &TilePlan3d, channels: usize) -> Blender {
        let (f, h, w) = plan.out_shape();
        Blender {
            c: channels,
            f,
            h,
            w,
            acc: vec![0.0f32; channels * f * h * w],
            wt: plan.t.weights(),
            wh: plan.h.weights(),
            ww: plan.w.weights(),
        }
    }

    /// Add one decoded tile, laid out `[C, tf, th, tw]` in the same row-major
    /// order the accumulator uses, scaled by the tile's separable mask.
    pub fn add(&mut self, tile: Tile3d<'_>, pixels: &[f32]) {
        let (tf, th, tw) = (tile.t.dst_len(), tile.h.dst_len(), tile.w.dst_len());
        assert_eq!(pixels.len(), self.c * tf * th * tw, "Blender::add: tile has {} values, expected {}", pixels.len(), self.c * tf * th * tw);
        let (f0, h0, w0) = (tile.t.dst.0, tile.h.dst.0, tile.w.dst.0);
        for ci in 0..self.c {
            for fi in 0..tf {
                let mf = tile.t.mask[fi];
                for hi in 0..th {
                    let mfh = mf * tile.h.mask[hi];
                    let src = ((ci * tf + fi) * th + hi) * tw;
                    let dst = ((ci * self.f + f0 + fi) * self.h + h0 + hi) * self.w + w0;
                    for wi in 0..tw {
                        self.acc[dst + wi] += pixels[src + wi] * mfh * tile.w.mask[wi];
                    }
                }
            }
        }
    }

    /// Divide out the accumulated blend weight and take the result.
    ///
    /// The divisor is floored at `1e-8` exactly as the reference's
    /// `compute_summed_weights` does, so an output cell no tile covered (which
    /// a correct plan never produces - the axis plans are asserted to cover
    /// their axis) yields 0 rather than a NaN.
    pub fn finish(mut self) -> Vec<f32> {
        for ci in 0..self.c {
            for fi in 0..self.f {
                for hi in 0..self.h {
                    let d = (self.wt[fi] * self.wh[hi]).max(1e-8);
                    let row = ((ci * self.f + fi) * self.h + hi) * self.w;
                    for wi in 0..self.w {
                        self.acc[row + wi] /= (d * self.ww[wi]).max(1e-8);
                    }
                }
            }
        }
        self.acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------- masks

    /// Pinned against the reference formula evaluated by hand:
    /// `linspace(0,1,r+2)[:-1][1:]` for the spatial fade-in and
    /// `linspace(1,0,r+2)[1:-1]` for the fade-out.
    #[test]
    fn the_spatial_trapezoid_never_reaches_zero_and_its_ramps_are_linear() {
        let m = trapezoidal_mask_1d(8, 3, 3, false);
        assert_eq!(&m[..3], &[0.25, 0.5, 0.75]);
        assert_eq!(&m[3..5], &[1.0, 1.0]);
        assert_eq!(&m[5..], &[0.75, 0.5, 0.25]);
    }

    /// The temporal convention differs: the fade-in DOES start at 0, and the
    /// step is `1/r` rather than `1/(r+1)`. A port that used one convention
    /// for both axes would pass a "looks like a ramp" eyeball and produce a
    /// seam every temporal tile boundary.
    #[test]
    fn the_temporal_trapezoid_starts_at_zero_with_a_different_step() {
        let m = trapezoidal_mask_1d(8, 4, 3, true);
        assert_eq!(&m[..4], &[0.0, 0.25, 0.5, 0.75]);
        assert_eq!(&m[4..5], &[1.0]);
        assert_eq!(&m[5..], &[0.75, 0.5, 0.25]);
    }

    /// Ramps are applied multiplicatively (`*=`), so an over-long pair
    /// multiplies rather than the later one overwriting the earlier.
    #[test]
    fn overlapping_ramps_multiply_rather_than_overwrite() {
        let m = trapezoidal_mask_1d(4, 4, 4, false);
        // fade_in  = [.2 .4 .6 .8], fade_out = [.8 .6 .4 .2]
        for (got, want) in m.iter().zip([0.2 * 0.8, 0.4 * 0.6, 0.6 * 0.4, 0.8 * 0.2]) {
            assert!((got - want).abs() < 1e-6, "{m:?}");
        }
    }

    // ------------------------------------------------------------ splits

    /// The reference's own conv-VAE 1080p width layout: latent extent 60,
    /// tile 24 (768 px / 32), overlap 2 (64 px / 32).
    #[test]
    fn split_by_size_matches_the_reference_interval_arithmetic() {
        let ivs = split_by_size(24, 2, 60);
        assert_eq!(
            ivs,
            vec![
                Interval { start: 0, end: 24, left_ramp: 0, right_ramp: 2 },
                Interval { start: 22, end: 46, left_ramp: 2, right_ramp: 2 },
                Interval { start: 44, end: 60, left_ramp: 2, right_ramp: 0 },
            ]
        );
    }

    #[test]
    fn an_axis_that_fits_one_tile_is_a_single_ramp_free_interval() {
        assert_eq!(split_by_size(24, 2, 24), vec![Interval { start: 0, end: 24, left_ramp: 0, right_ramp: 0 }]);
        assert_eq!(split_temporal_causal(10, 3, 4), vec![Interval { start: 0, end: 4, left_ramp: 0, right_ramp: 0 }]);
    }

    #[test]
    fn the_causal_temporal_split_shifts_every_later_tile_back_by_one() {
        let plain = split_by_size(10, 3, 25);
        let causal = split_temporal_causal(10, 3, 25);
        assert_eq!(causal[0], plain[0]);
        for (c, p) in causal.iter().zip(&plain).skip(1) {
            assert_eq!(c.start, p.start - 1);
            assert_eq!(c.left_ramp, p.left_ramp + 1);
            assert_eq!(c.end, p.end);
        }
    }

    // ---------------------------------------------------------- mappings

    #[test]
    fn the_temporal_mapping_follows_the_one_plus_8k_frame_rule() {
        // Whole axis, untiled: 4 latent frames -> 25 pixel frames.
        let a = map_temporal(Interval { start: 0, end: 4, left_ramp: 0, right_ramp: 0 }, 8);
        assert_eq!(a.dst, (0, 25));
        assert!(a.mask.iter().all(|m| *m == 1.0));
    }

    /// The property the whole causal split exists for: after mapping, the
    /// per-axis masks sum to exactly 1 at every output frame. Checked on a
    /// long clip that really is split (11 latent frames, tile 5, overlap 2).
    #[test]
    fn the_temporal_masks_partition_unity_after_the_causal_shift() {
        let plan = AxisPlan::temporal(11, 5, 2, 8);
        assert!(plan.len() >= 3, "expected a genuinely multi-tile split, got {}", plan.len());
        assert!(plan.unity_error() < 1e-6, "temporal weights deviate by {}", plan.unity_error());
        assert_eq!(plan.out_len, 1 + 10 * 8);
    }

    #[test]
    fn the_spatial_masks_partition_unity() {
        for (lat, tile, overlap) in [(60usize, 24usize, 2usize), (34, 14, 2), (45, 24, 2), (17, 8, 3)] {
            let plan = AxisPlan::spatial(lat, tile, overlap, 32);
            assert!(plan.unity_error() < 1e-6, "lat={lat} tile={tile} overlap={overlap}: deviation {}", plan.unity_error());
            assert_eq!(plan.out_len, lat * 32);
        }
    }

    // ------------------------------------------------------------- blend

    /// The blend's own correctness, isolated from any decoder: cut a known
    /// pixel volume into the plan's tiles, feed the pieces back through
    /// [`Blender`], and require the stitched result to equal the original.
    ///
    /// This is the gate that a decoder-based comparison CANNOT be, because a
    /// real decoder's receptive field is wider than the overlap and so its
    /// tiles genuinely disagree in the seam. Here the "decoder" is the
    /// identity, so any deviation is a mask, slice or divisor bug and nothing
    /// else.
    ///
    /// Run at REDUCED scale factors (2 instead of 8/32) on purpose: the scale
    /// is a parameter of every function under test, so the same arithmetic
    /// runs, while the real `(8, 32, 32)` factors would make this one test
    /// materialise a 1.4 GB volume and dominate the suite's runtime. The real
    /// factors are covered by the mask/partition tests above, which are
    /// `O(axis length)` rather than `O(volume)`.
    #[test]
    fn the_blend_reconstructs_a_known_volume_exactly() {
        let plan = TilePlan3d { t: AxisPlan::temporal(11, 5, 2, 2), h: AxisPlan::spatial(34, 14, 2, 2), w: AxisPlan::spatial(60, 24, 2, 2) };
        assert!(plan.tiles().len() > 8, "expected a genuinely 3D-split plan, got {}", plan.tiles().len());
        assert!(plan.masks_are_complementary());

        let (f, h, w) = plan.out_shape();
        let c = 2usize;
        // A volume with real structure on every axis, so a swapped or
        // off-by-one slice cannot cancel out.
        let val = |ci: usize, fi: usize, hi: usize, wi: usize| ((ci * 7 + fi) as f32 * 0.37).sin() + (hi as f32 * 0.011).cos() * (wi as f32 * 0.007).sin();
        let mut whole = vec![0.0f32; c * f * h * w];
        for ci in 0..c {
            for fi in 0..f {
                for hi in 0..h {
                    for wi in 0..w {
                        whole[((ci * f + fi) * h + hi) * w + wi] = val(ci, fi, hi, wi);
                    }
                }
            }
        }

        let mut b = Blender::new(&plan, c);
        for tile in plan.tiles() {
            let (tf, th, tw) = (tile.t.dst_len(), tile.h.dst_len(), tile.w.dst_len());
            let mut piece = vec![0.0f32; c * tf * th * tw];
            for ci in 0..c {
                for fi in 0..tf {
                    for hi in 0..th {
                        for wi in 0..tw {
                            piece[((ci * tf + fi) * th + hi) * tw + wi] = val(ci, tile.t.dst.0 + fi, tile.h.dst.0 + hi, tile.w.dst.0 + wi);
                        }
                    }
                }
            }
            b.add(tile, &piece);
        }
        let got = b.finish();
        let worst = got.iter().zip(&whole).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "blend reconstruction worst |delta| = {worst}");
    }

    #[test]
    fn a_single_tile_plan_blends_to_the_identity() {
        let plan = TilePlan3d { t: AxisPlan::temporal(4, 10, 3, 8), h: AxisPlan::spatial(4, 24, 2, 32), w: AxisPlan::spatial(4, 24, 2, 32) };
        assert_eq!(plan.tiles().len(), 1);
        assert_eq!(plan.overlap_waste(), 1.0);
        let n = 3 * plan.t.out_len * plan.h.out_len * plan.w.out_len;
        let src: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut b = Blender::new(&plan, 3);
        b.add(plan.tiles()[0], &src);
        assert_eq!(b.finish(), src, "an untiled plan must be bit-identical");
    }

    #[test]
    fn overlap_waste_counts_the_redundant_volume() {
        let plan = TilePlan3d { t: AxisPlan::temporal(4, 10, 3, 8), h: AxisPlan::spatial(34, 14, 2, 32), w: AxisPlan::spatial(60, 24, 2, 32) };
        // 3 x 3 spatial tiles, temporal untiled.
        assert_eq!(plan.tiles().len(), 9);
        let waste = plan.overlap_waste();
        assert!(waste > 1.0 && waste < 1.4, "waste {waste}");
    }
}
