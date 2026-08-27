// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spatial preservation masks for the FLUX.2 denoise loop - **blended latent
//! diffusion**.
//!
//! `--strength` is a *global* dial: it decides how far down the schedule the
//! whole canvas starts, so every pixel is preserved or redrawn to the same
//! degree. Some edits need a *spatial* dial instead - redraw the middle of the
//! room, keep the walls and windows exactly. That is what a mask is.
//!
//! ## The rule
//!
//! A mask is one greyscale weight per output pixel, **white = regenerate,
//! black = preserve**. It is area-averaged down to the latent grid, and after
//! every Euler step the latent is recombined with the *source* latent renoised
//! to that step's own sigma:
//!
//! ```text
//! x = m·x_denoised + (1 − m)·((1 − σ)·x₀ + σ·ε)
//! ```
//!
//! `(1 − σ)·x₀ + σ·ε` is the rectified-flow forward process - the same one
//! `strength` uses to build its init latent, and the same one the trainer's
//! `modelgrad::make_flow_batch` draws from - so the preserved region is always
//! a legal point on the source's own trajectory, not an out-of-distribution
//! paste. At the final sigma of 0 it is the source latent exactly. Preserved
//! regions therefore **track the source at every step** rather than being
//! softly guided toward it, which is the whole difference from `strength`.
//!
//! ## Two properties this module guarantees exactly, not approximately
//!
//! * An **all-white** mask is a bit-for-bit no-op: masking costs nothing when
//!   it is not used, so it cannot silently perturb an unmasked generation.
//! * An **all-black** mask reproduces the source latent bit-for-bit.
//!
//! Both fall out of [`blend`]'s hard-0/hard-1 short circuits *and* of the
//! resampler's integer area weights ([`Mask::to_latent`]), which map a mask
//! that is constant over a latent cell to exactly that constant. A weighted
//! mean that merely *rounded* to 1.0 would not do: it would leave every
//! unmasked run one blend away from its old output.
//!
//! Intermediate greys are used verbatim as blend weights - see
//! [`Mask::from_hwc`] for the normalisation rule and [`Mask::to_latent`] for
//! the resampling rule. Soft edges are the point: a hard latent-cell boundary
//! between "kept" and "redrawn" shows up as a seam.
//!
//! Swedish Embedded AB implements masked/region-controlled diffusion editing
//! for its clients. If your team needs expertise in latent-space image editing
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.

/// A greyscale preservation mask over the **output canvas**: one weight per
/// pixel, row-major, `1.0` = regenerate, `0.0` = preserve.
///
/// Constructed already normalised - see [`Mask::from_hwc`] - so the denoise
/// loop never has to ask what range it is holding.
#[derive(Clone, PartialEq)]
pub struct Mask {
    w: u32,
    h: u32,
    values: Vec<f32>,
}

impl std::fmt::Debug for Mask {
    /// Dimensions and coverage, never the pixels: a mask is up to millions of
    /// floats and [`crate::GenOpts`] is `Debug`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let regen = self.values.iter().sum::<f32>() / self.values.len().max(1) as f32;
        write!(f, "Mask({}x{}, {:.1}% regenerate)", self.w, self.h, 100.0 * regen)
    }
}

impl Mask {
    /// Wrap already-normalised `[0,1]` weights. `values` is `h*w` row-major.
    pub fn new(values: Vec<f32>, w: u32, h: u32) -> Result<Mask, String> {
        if w == 0 || h == 0 {
            return Err(format!("mask {w}x{h} is empty"));
        }
        if values.len() != (w as usize) * (h as usize) {
            return Err(format!("mask {w}x{h} needs {} weights, got {}", w as usize * h as usize, values.len()));
        }
        // A NaN weight is a broken file, not a blend instruction, and it would
        // propagate silently through `clamp` into the latent. Refuse it here.
        if let Some(i) = values.iter().position(|v| v.is_nan()) {
            return Err(format!("mask {w}x{h} has a NaN weight at index {i}"));
        }
        let values = values.into_iter().map(|v| v.clamp(0.0, 1.0)).collect();
        Ok(Mask { w, h, values })
    }

    /// Build from a decoded image in the CLI/capability wire format:
    /// interleaved HWC RGB in `[0,1]`.
    ///
    /// **Normalisation rule**, stated because it is a choice and the obvious
    /// alternative is wrong:
    ///
    /// * the three channels are averaged, so a genuinely greyscale file (where
    ///   `R == G == B`) round-trips exactly and a colour one degrades to its
    ///   mean rather than being rejected;
    /// * values are **clamped** into `[0,1]`, never min/max *stretched*. A
    ///   stretch would turn a uniformly white mask into a division by zero and
    ///   a uniformly mid-grey one into all-black or all-white - i.e. it would
    ///   destroy exactly the two masks whose meaning must be unambiguous;
    /// * everything strictly between 0 and 1 is kept **verbatim** as a linear
    ///   blend weight. There is no threshold: 50% grey means half the source
    ///   and half the generation, which is what makes a feathered edge blend
    ///   instead of cut.
    pub fn from_hwc(hwc: &[f32], w: u32, h: u32) -> Result<Mask, String> {
        let n = (w as usize) * (h as usize);
        if hwc.len() != n * 3 {
            return Err(format!("mask {w}x{h} needs {} RGB samples, got {}", n * 3, hwc.len()));
        }
        let values = (0..n)
            .map(|i| {
                let (r, g, b) = (hwc[i * 3], hwc[i * 3 + 1], hwc[i * 3 + 2]);
                // A grey pixel is returned verbatim rather than run through
                // `(r+g+b)/3`, which does not round back to its own input for
                // every f32 - and the whole point of a mask is that a white
                // pixel is EXACTLY 1 and a black one EXACTLY 0.
                if r == g && g == b {
                    r
                } else {
                    (r + g + b) / 3.0
                }
            })
            .collect();
        Mask::new(values, w, h)
    }

    pub fn width(&self) -> u32 {
        self.w
    }

    pub fn height(&self) -> u32 {
        self.h
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Resample to the `lh × lw` latent token grid: one weight per latent
    /// token, row-major, matching `position_ids`' token order.
    ///
    /// **Resampling rule**: an exact **area average** (box filter). Latent cell
    /// `(y, x)` covers the rectangle `[x/lw, (x+1)/lw) × [y/lh, (y+1)/lh)` of
    /// the mask in *normalised* coordinates, and its weight is the mean of the
    /// mask over that rectangle, each source pixel counted by how much of it
    /// the rectangle actually covers. The two axes are resampled independently,
    /// so a **non-square** canvas (the 4:3 real-estate frames this was built
    /// for) is not a special case: `lw/w` and `lh/h` are simply different
    /// ratios. Upsampling a mask smaller than the latent grid works by the same
    /// rule, degrading to nearest-neighbour when a cell falls inside one pixel.
    ///
    /// The overlaps are accumulated as **integers** (both axes scaled by the
    /// destination extent, which makes every overlap and the per-cell total
    /// exact) and divided once at the end. That is what makes a mask which is
    /// *constant over a cell* resample to exactly that constant - the property
    /// the all-white no-op and all-black passthrough guarantees rest on.
    pub fn to_latent(&self, lh: usize, lw: usize) -> Vec<f32> {
        let (mw, mh) = (self.w as usize, self.h as usize);
        let rows = axis_overlaps(mh, lh);
        let cols = axis_overlaps(mw, lw);
        // Each cell's overlaps sum to exactly `mh * mw` (whole numbers, and far
        // below 2^53), so a cell covering a constant region accumulates
        // `c * total` exactly and divides back to `c`.
        let total = (mh as f64) * (mw as f64);
        let mut out = vec![0.0f32; lh * lw];
        for (y, ry) in rows.iter().enumerate() {
            for (x, rx) in cols.iter().enumerate() {
                let mut acc = 0.0f64;
                for &(sy, wy) in ry {
                    for &(sx, wx) in rx {
                        acc += (wy * wx) as f64 * self.values[sy * mw + sx] as f64;
                    }
                }
                out[y * lw + x] = (acc / total) as f32;
            }
        }
        out
    }
}

/// Per-destination-cell source overlaps for one axis, as exact integers.
///
/// Both extents are scaled by the *other* one (destination cell `j` spans
/// `[j·src, (j+1)·src)`, source pixel `i` spans `[i·dst, (i+1)·dst)`), so every
/// overlap is a whole number and each cell's overlaps sum to exactly `src`.
fn axis_overlaps(src: usize, dst: usize) -> Vec<Vec<(usize, u64)>> {
    let (s, d) = (src as u64, dst as u64);
    (0..dst)
        .map(|j| {
            let (lo, hi) = (j as u64 * s, (j as u64 + 1) * s);
            // Source pixels the cell can touch: `lo/d ..= (hi-1)/d`.
            let first = (lo / d) as usize;
            let last = ((hi - 1) / d) as usize;
            (first..=last.min(src - 1))
                .map(|i| {
                    let (plo, phi) = (i as u64 * d, (i as u64 + 1) * d);
                    (i, hi.min(phi) - lo.max(plo))
                })
                .collect()
        })
        .collect()
}

/// One masked-blend step: recombine `lat` with the source latent renoised to
/// `sigma`, in place.
///
/// `mask` is one weight per token (`n` of them), `src` and `noise` are
/// `n * ch` latent values in the same token-major layout as `lat`. `sigma` is
/// the noise level `lat` now sits at - i.e. the schedule entry the step just
/// *arrived at*, not the one it left.
///
/// Hard 0 and hard 1 are short-circuited rather than multiplied through, which
/// is what makes them exact: `m = 1` leaves the bits of `lat` untouched, and
/// `m = 0` writes the renoised source with no arithmetic that could round.
pub fn blend(lat: &mut [f32], mask: &[f32], src: &[f32], noise: &[f32], sigma: f32, ch: usize) {
    debug_assert_eq!(lat.len(), mask.len() * ch);
    debug_assert_eq!(src.len(), lat.len());
    debug_assert_eq!(noise.len(), lat.len());
    for (t, &m) in mask.iter().enumerate() {
        if m >= 1.0 {
            continue; // regenerate: the step's own result stands, untouched
        }
        let span = t * ch..(t + 1) * ch;
        if m <= 0.0 {
            // Preserve: write the renoised source with no arithmetic on the
            // denoised value at all. At the terminal σ the renoise is copied
            // rather than computed, so the region lands on `src` bit-for-bit
            // (`x + 0.0·ε` differs from `x` for a negative zero).
            if sigma <= 0.0 {
                lat[span.clone()].copy_from_slice(&src[span]);
            } else {
                for i in span {
                    lat[i] = (1.0 - sigma) * src[i] + sigma * noise[i];
                }
            }
        } else {
            for i in span {
                let r = (1.0 - sigma) * src[i] + sigma * noise[i];
                lat[i] = m * lat[i] + (1.0 - m) * r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grey(v: f32, w: u32, h: u32) -> Mask {
        Mask::new(vec![v; (w * h) as usize], w, h).unwrap()
    }

    /// The two masks whose meaning must be exact stay exact through the
    /// resampler. A weighted mean that merely *rounded* to 1.0 would leave
    /// every unmasked generation one blend away from its previous output.
    #[test]
    fn a_uniform_mask_resamples_to_that_constant_exactly() {
        for v in [0.0f32, 1.0, 0.5, 0.25] {
            // 4:3, the aspect these masks are actually used at.
            let m = grey(v, 1024, 768);
            let l = m.to_latent(48, 64);
            assert_eq!(l.len(), 48 * 64);
            assert!(l.iter().all(|&x| x.to_bits() == v.to_bits()), "{v} did not survive resampling: {:?}", &l[..4]);
        }
    }

    /// The resampling rule is an area average, and the two axes carry
    /// independent ratios - the 4:3 case is not a square case in disguise.
    #[test]
    fn resampling_is_an_area_average_on_both_axes_independently() {
        // 8x4 mask -> 2x2 latent grid: each cell averages a 2(row)x4(col) box.
        // Rows: 0,0,1,1 ; columns: left half 1, right half 0.
        let (w, h) = (8u32, 4u32);
        let mut v = vec![0.0f32; (w * h) as usize];
        for y in 2..4usize {
            for x in 0..8usize {
                v[y * 8 + x] = if x < 4 { 1.0 } else { 0.0 };
            }
        }
        let l = Mask::new(v, w, h).unwrap().to_latent(2, 2);
        // top row of cells sees only zeros; bottom-left sees all ones.
        assert_eq!(l, vec![0.0, 0.0, 1.0, 0.0]);

        // A cell straddling the edge gets the exact area fraction: a 3-wide
        // mask into 2 cells splits the middle pixel 1/2 : 1/2.
        let l = Mask::new(vec![1.0, 0.0, 0.0], 3, 1).unwrap().to_latent(1, 2);
        assert!((l[0] - 2.0 / 3.0).abs() < 1e-6, "{l:?}");
        assert!((l[1] - 0.0).abs() < 1e-6, "{l:?}");
    }

    /// A hard-edged region resamples exactly; only the cells the edge actually
    /// crosses are soft. This is what keeps a preserved wall bit-identical
    /// while still feathering the seam.
    #[test]
    fn hard_regions_are_exact_and_only_the_seam_is_soft() {
        // 4:3 canvas, left 3/8 white, right 5/8 black, on a 48x64 latent grid.
        let (w, h) = (1024u32, 768u32);
        let mut v = vec![0.0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..(3 * w / 8) as usize {
                v[y * w as usize + x] = 1.0;
            }
        }
        let l = Mask::new(v, w, h).unwrap().to_latent(48, 64);
        // 3/8 of 64 cells = cell 24 exactly, so there is no partial cell.
        for y in 0..48 {
            for x in 0..64 {
                let want: f32 = if x < 24 { 1.0 } else { 0.0 };
                assert_eq!(l[y * 64 + x].to_bits(), want.to_bits(), "cell ({y},{x})");
            }
        }
    }

    /// Greys are blend weights, not a threshold, and the range rule is
    /// clamping rather than a min/max stretch.
    #[test]
    fn greys_are_kept_verbatim_and_out_of_range_values_clamp() {
        let hwc: Vec<f32> = vec![0.25, 0.25, 0.25, -3.0, -3.0, -3.0, 7.0, 7.0, 7.0, 0.6, 0.6, 0.6];
        let m = Mask::from_hwc(&hwc, 4, 1).unwrap();
        assert_eq!(m.values(), &[0.25, 0.0, 1.0, 0.6]);
        // A uniformly mid-grey mask is mid-grey, not stretched to black/white.
        let hwc: Vec<f32> = vec![0.5; 12];
        assert_eq!(Mask::from_hwc(&hwc, 4, 1).unwrap().values(), &[0.5; 4]);
    }

    /// White is a *bit-for-bit* no-op, black is a bit-for-bit passthrough of
    /// the renoised source, and the values between are the plain lerp.
    #[test]
    fn blend_is_exact_at_both_ends_and_a_lerp_between() {
        let src = vec![0.5f32, -0.25, 3.0, -7.0];
        let noise = vec![-1.0f32, 2.0, 0.125, 9.0];
        let base = vec![0.3f32, 0.7, -0.9, 1.1];
        let ch = 2; // 2 tokens x 2 channels

        let mut white = base.clone();
        blend(&mut white, &[1.0, 1.0], &src, &noise, 0.4, ch);
        assert!(white.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()), "white must not touch a single bit");

        let mut black = base.clone();
        blend(&mut black, &[0.0, 0.0], &src, &noise, 0.0, ch);
        assert!(black.iter().zip(&src).all(|(a, b)| a.to_bits() == b.to_bits()), "black at sigma 0 must BE the source");

        // Negative zero is the ONLY input that tells "skip the arithmetic"
        // apart from "do the arithmetic and get the same answer": `1·x + 0·r`
        // and `1·x₀ + 0·ε` both turn a -0.0 into a +0.0. Without these two
        // cases the short circuits in `blend` are code no test can distinguish
        // from its absence, and the module's bit-for-bit claims are one
        // unlucky value short of true.
        let (nz_src, nz_lat) = (vec![-0.0f32; 2], vec![-0.0f32; 2]);
        let mut white_nz = nz_lat.clone();
        blend(&mut white_nz, &[1.0], &src[..2], &noise[..2], 0.4, 2);
        assert!(white_nz.iter().all(|v| v.is_sign_negative()), "white must preserve a negative zero");
        let mut black_nz = vec![7.0f32; 2];
        blend(&mut black_nz, &[0.0], &nz_src, &noise[..2], 0.0, 2);
        assert!(black_nz.iter().all(|v| v.is_sign_negative()), "black at sigma 0 must preserve the source's negative zero");

        // Halfway, at a sigma where the renoise actually does something.
        let (s, m) = (0.25f32, 0.5f32);
        let mut mid = base.clone();
        blend(&mut mid, &[m, m], &src, &noise, s, ch);
        for i in 0..4 {
            let r = (1.0 - s) * src[i] + s * noise[i];
            assert!((mid[i] - (m * base[i] + (1.0 - m) * r)).abs() < 1e-6, "i={i}");
        }
    }

    /// The mask is per *token*, so all 128 channels of a preserved token move
    /// together - a per-channel mask would be meaningless (latent channels are
    /// not spatial).
    #[test]
    fn one_weight_covers_every_channel_of_its_token() {
        let ch = 3;
        let src = vec![1.0f32; 6];
        let noise = vec![0.0f32; 6];
        let mut lat = vec![9.0f32; 6];
        blend(&mut lat, &[0.0, 1.0], &src, &noise, 0.0, ch);
        assert_eq!(lat, vec![1.0, 1.0, 1.0, 9.0, 9.0, 9.0]);
    }
}
