// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single-Image Random-Dot Stereogram (SIRDS) — the "Magic Eye" autostereogram.
//!
//! Depth-only: no glasses, no color image. The algorithm is Thimbleby, Witkin &
//! Inglis ("Displaying 3D Images", IEEE Computer 1994). Each scanline is
//! independent. For every pixel, its depth sets a horizontal SEPARATION `s`: the
//! two screen points `x-s/2` and `x+s/2` are constrained to be the same colour, so
//! when the eyes diverge (or cross) by that separation the point fuses at the depth
//! `s` encodes. A repeating random-dot pattern colours everything not otherwise
//! constrained; the constraints then ripple that pattern into the depth surface.
//!
//! Nearer surfaces get a SMALLER separation (their linked dots sit closer
//! together), so they float toward the viewer. `invert` flips that.
//!
//! Pure and host-side, like the rest of `viz` — rendered and eyeballed to verify.

use crate::viz::Bounds;

/// Autostereogram parameters.
#[derive(Clone, Copy, Debug)]
pub struct StereoOpts {
    /// Eye separation in pixels — the far-plane pattern period is `eye_sep/2`. Larger
    /// = fewer, wider repeats (easier to fuse on a big screen); smaller = denser.
    pub eye_sep: u32,
    /// Depth of field (0..1): how much of the separation range depth may use. ~1/3
    /// is comfortable; higher exaggerates depth but strains fusion.
    pub mu: f32,
    /// If true, HIGH depth values are NEAR (our inverse-depth convention). Flip to
    /// swap pop-out for pop-in.
    pub near_is_high: bool,
}

impl StereoOpts {
    /// Sensible defaults for a frame `w` px wide: ~5–6 pattern repeats.
    pub fn for_width(w: u32) -> StereoOpts {
        StereoOpts { eye_sep: (w / 5).clamp(90, 260), mu: 0.33, near_is_high: true }
    }
}

/// The stereo separation (in px) for a normalized depth `z in [0,1]` (1 = nearest).
/// `(1 - μz)E / (2 - μz)`: E/2 at the far plane, shrinking toward the viewer.
fn separation(z: f32, e: f32, mu: f32) -> i32 {
    (((1.0 - mu * z) * e / (2.0 - mu * z)).round()) as i32
}

/// A deterministic, colourful seed dot for the pattern (no RNG state, so the output
/// is reproducible and testable).
fn seed_color(x: usize, y: usize) -> [u8; 3] {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    // Full-range, high-contrast dots: SIRDS free-viewing relies on sharp per-dot
    // contrast so the eye can lock the repeat. Pastel dots wash out.
    [(h & 0xff) as u8, ((h >> 8) & 0xff) as u8, ((h >> 16) & 0xff) as u8]
}

/// Render a `[h*w]` depth map as a `w×h` RGB8 autostereogram of RANDOM DOTS.
pub fn autostereogram(depth: &[f32], w: u32, h: u32, bounds: Bounds, opts: &StereoOpts) -> Vec<u8> {
    build(depth, w, h, bounds, opts, None)
}

/// A TEXTURED autostereogram: unconstrained pixels are seeded from `source`
/// (row-major RGB8, same `w×h`) instead of random dots, so the stereogram is made
/// of the camera image's own colours and local textures — free-view it and the
/// depth pops while the surface stays recognizably the photo. A single-image
/// stereogram can only show one pattern period repeated, so the image tiles/warps,
/// but disocclusions at depth edges reveal fresh image content where the geometry
/// changes.
pub fn autostereogram_textured(depth: &[f32], w: u32, h: u32, bounds: Bounds, opts: &StereoOpts, source: &[u8]) -> Vec<u8> {
    assert_eq!(source.len(), (w * h * 3) as usize, "source must be [h*w*3] RGB");
    build(depth, w, h, bounds, opts, Some(source))
}

fn build(depth: &[f32], w: u32, h: u32, bounds: Bounds, opts: &StereoOpts, source: Option<&[u8]>) -> Vec<u8> {
    assert_eq!(depth.len(), (w * h) as usize, "depth must be [h*w]");
    let (wi, hi) = (w as usize, h as usize);
    let e = opts.eye_sep as f32;
    let mu = opts.mu;
    let z_of = |d: f32| -> f32 {
        let n = bounds.norm(d);
        if opts.near_is_high {
            n
        } else {
            1.0 - n
        }
    };

    let mut out = vec![0u8; wi * hi * 3];
    let mut same = vec![0usize; wi];
    for y in 0..hi {
        for (x, s) in same.iter_mut().enumerate() {
            *s = x;
        }
        for x in 0..wi {
            let z = z_of(depth[y * wi + x]);
            let s = separation(z, e, mu);
            let left = x as i32 - s / 2;
            let right = left + s;
            if left < 0 || right as usize >= wi {
                continue;
            }
            let (left, right) = (left as usize, right as usize);

            // Hidden-surface test (Thimbleby et al): a constraint is only honoured if
            // both linked pixels are actually visible from this depth — otherwise a
            // nearer surface between them would occlude one eye, and linking creates
            // ghost echoes at depth edges.
            let mut visible = true;
            let mut t = 1i32;
            loop {
                let zt = z + 2.0 * (2.0 - mu * z) * (t as f32) / (mu * e);
                let (xl, xr) = (x as i32 - t, x as i32 + t);
                if xl < 0 || xr as usize >= wi || zt >= 1.0 {
                    break;
                }
                let zl = z_of(depth[y * wi + xl as usize]);
                let zr = z_of(depth[y * wi + xr as usize]);
                visible = zl < zt && zr < zt;
                if !visible {
                    break;
                }
                t += 1;
            }
            if visible {
                // `right` copies from `left`; the left-to-right fill below then
                // ripples the pattern into the surface.
                same[right] = left;
            }
        }
        // Assign colours left-to-right so each `same[x]` source is already set.
        for x in 0..wi {
            let color = if same[x] == x {
                match source {
                    Some(src) => [src[(y * wi + x) * 3], src[(y * wi + x) * 3 + 1], src[(y * wi + x) * 3 + 2]],
                    None => seed_color(x, y),
                }
            } else {
                let s = same[x];
                [out[(y * wi + s) * 3], out[(y * wi + s) * 3 + 1], out[(y * wi + s) * 3 + 2]]
            };
            out[(y * wi + x) * 3..(y * wi + x) * 3 + 3].copy_from_slice(&color);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A FLAT depth field must produce a strictly horizontally-periodic pattern with
    /// period = `separation(z)` — the defining property: with no depth variation the
    /// autostereogram is just a repeating wallpaper, and the period is exactly the
    /// separation. Any deviation means the linking is wrong.
    #[test]
    fn flat_depth_is_periodic_with_the_separation() {
        let (w, h) = (400u32, 8u32);
        let opts = StereoOpts { eye_sep: 120, mu: 0.33, near_is_high: true };
        // z = 0.5 everywhere.
        let bounds = Bounds { lo: 0.0, hi: 1.0 };
        let depth = vec![0.5f32; (w * h) as usize];
        let img = autostereogram(&depth, w, h, bounds, &opts);
        let s = separation(0.5, opts.eye_sep as f32, opts.mu) as usize;
        assert!(s > 4, "separation should be meaningful, got {s}");
        // Every pixel at least `s` from the left edge equals the one `s` to its left.
        for y in 0..h as usize {
            for x in s..w as usize {
                let a = &img[(y * w as usize + x) * 3..][..3];
                let b = &img[(y * w as usize + (x - s)) * 3..][..3];
                assert_eq!(a, b, "flat depth must repeat with period {s} at ({x},{y})");
            }
        }
    }

    /// A NEARER region uses a smaller separation than a farther one — the whole point
    /// of encoding depth. Pin the monotonicity of the separation function.
    #[test]
    fn nearer_depth_has_smaller_separation() {
        let (e, mu) = (120.0, 0.33);
        let far = separation(0.0, e, mu);
        let mid = separation(0.5, e, mu);
        let near = separation(1.0, e, mu);
        assert!(near < mid && mid < far, "separation must shrink with depth: {far} {mid} {near}");
    }

    /// The output is a full RGB image with no black (unwritten) pixels — every pixel
    /// gets either a seed dot or a copied one.
    #[test]
    fn every_pixel_is_written() {
        let (w, h) = (200u32, 40u32);
        let depth: Vec<f32> = (0..(w * h)).map(|i| ((i % 100) as f32) / 100.0).collect();
        let img = autostereogram(&depth, w, h, Bounds { lo: 0.0, hi: 1.0 }, &StereoOpts::for_width(w));
        assert_eq!(img.len(), (w * h * 3) as usize);
        // The fill must touch every column: no ROW may be entirely black (an
        // unwritten column would leave a black streak). Individual black dots are
        // fine now that seed colours span the full range.
        for y in 0..h as usize {
            let row_black = (0..w as usize).all(|x| img[(y * w as usize + x) * 3..][..3] == [0, 0, 0]);
            assert!(!row_black, "row {y} is entirely black — the fill missed it");
        }
    }

    /// A raised central square must actually shift the pattern there: the separation
    /// inside the square differs from the background, so the pattern is NOT globally
    /// periodic (unlike the flat case). This is the smoke test that depth reaches the
    /// image.
    #[test]
    fn a_raised_region_breaks_global_periodicity() {
        let (w, h) = (400u32, 60u32);
        let opts = StereoOpts { eye_sep: 120, mu: 0.5, near_is_high: true };
        let mut depth = vec![0.0f32; (w * h) as usize];
        for y in 20..40 {
            for x in 150..250 {
                depth[y * w as usize + x] = 1.0; // a near square on a far field
            }
        }
        let img = autostereogram(&depth, w, h, Bounds { lo: 0.0, hi: 1.0 }, &opts);
        let s_far = separation(0.0, opts.eye_sep as f32, opts.mu) as usize;
        // On a background row the pattern is periodic with s_far; on a row through the
        // square it must NOT be (the square's smaller separation perturbs it).
        let periodic = |y: usize, s: usize| -> bool {
            (s..w as usize).all(|x| img[(y * w as usize + x) * 3..][..3] == img[(y * w as usize + (x - s)) * 3..][..3])
        };
        assert!(periodic(5, s_far), "a pure-background row should be periodic with s_far");
        assert!(!periodic(30, s_far), "a row through the near square must break s_far periodicity");
    }
}
