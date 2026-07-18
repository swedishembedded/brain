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
    /// Defaults for a frame `w` px wide: 5 pattern repeats (so the textured tile is
    /// a wide, meaningful `w/5` slice and the centre stripe is the image's centre),
    /// and a stronger depth budget (`mu = 0.5`) so the relief reads clearly.
    pub fn for_width(w: u32) -> StereoOpts {
        StereoOpts::with_stripes(w, 5)
    }

    /// Choose the number of horizontal pattern repeats explicitly. Fewer stripes =
    /// wider tile (more image per repeat, and closer to the eyes' physical
    /// separation for comfortable free-viewing); more = denser. The far-plane tile
    /// is `w/stripes` wide, and `eye_sep = 2 * tile`.
    pub fn with_stripes(w: u32, stripes: u32) -> StereoOpts {
        let stripes = stripes.max(2);
        StereoOpts { eye_sep: (2 * w / stripes).clamp(60, 400), mu: 0.5, near_is_high: true }
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

/// A TEXTURED (wallpaper) autostereogram: instead of random dots, the pattern is a
/// PERIODIC TILE taken from the centre strip of `source` (row-major RGB8, same
/// `w×h`) and repeated with period = the far-plane separation. Because the tile is
/// truly periodic, the eyes lock the repeat and the depth warps it — so it fuses to
/// 3D while showing the camera's own colours and textures.
///
/// The tile is ONE separation-wide strip from the image's horizontal centre, and it
/// varies per scanline, so vertical image detail is preserved and the perceived
/// surface reads like the photo's centre band in relief. (Seeding per-pixel from the
/// whole image — an earlier attempt — destroys the periodicity the illusion needs,
/// so no depth appears; that is the bug this replaces.)
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

    // Wallpaper method, left to right: `out[x] = out[x - separation(depth[x])]`. This
    // applies depth at EVERY column — a centred subject is fully in 3D. The ONLY
    // depth-free region is the leftmost ~max-separation strip, whose partner is
    // off-screen; a single-image stereogram cannot avoid one such strip, so it goes
    // at the EDGE (usually background) rather than the centre (the subject). For the
    // textured variant that seed strip tiles the image's CENTRE band, so the whole
    // surface is textured with the photo — sampled `x % tile_w` from the centre strip
    // to stay exactly periodic.
    let tile_w = (separation(0.0, e, mu).max(1) as usize).min(wi);
    let tile_x0 = (wi - tile_w) / 2; // image centre strip
    let mut out = vec![0u8; wi * hi * 3];
    for y in 0..hi {
        let base = y * wi;
        for x in 0..wi {
            let s = separation(z_of(depth[base + x]), e, mu).max(1) as usize;
            let c = if x >= s {
                let sx = base + x - s;
                [out[sx * 3], out[sx * 3 + 1], out[sx * 3 + 2]]
            } else {
                match source {
                    Some(src) => {
                        let sx = base + tile_x0 + (x % tile_w);
                        [src[sx * 3], src[sx * 3 + 1], src[sx * 3 + 2]]
                    }
                    None => seed_color(x, y),
                }
            };
            out[(base + x) * 3..(base + x) * 3 + 3].copy_from_slice(&c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// DEPTH MUST BE APPLIED AT EVERY COLUMN, INCLUDING THE CENTRE. Two depth maps
    /// that differ ONLY in a patch at the exact centre must produce DIFFERENT output
    /// in that patch — otherwise a centred subject shows no 3D there ("two ears
    /// clipped instead of the full head"). A static/flat centre reference tile
    /// ignores its own depth and fails this.
    #[test]
    fn centre_output_depends_on_centre_depth() {
        let (w, h) = (400u32, 8u32);
        let opts = StereoOpts { eye_sep: 120, mu: 0.5, near_is_high: true };
        let cx = w as usize / 2;
        let far = vec![0.0f32; (w * h) as usize];
        let mut near_patch = far.clone();
        for y in 0..h as usize {
            for x in (cx - 15)..(cx + 15) {
                near_patch[y * w as usize + x] = 1.0; // a near bump at the exact centre
            }
        }
        let b = Bounds { lo: 0.0, hi: 1.0 };
        let a = autostereogram(&far, w, h, b, &opts);
        let c = autostereogram(&near_patch, w, h, b, &opts);
        // The centre patch must be encoded: the two images must differ there.
        let diff: usize = (0..h as usize)
            .flat_map(|y| ((cx - 15)..(cx + 15)).map(move |x| (y, x)))
            .filter(|&(y, x)| a[(y * w as usize + x) * 3..][..3] != c[(y * w as usize + x) * 3..][..3])
            .count();
        assert!(diff > 0, "the centre depth change was NOT encoded — the centre is a static tile");
    }

    /// On FLAT depth the pattern must be horizontally periodic with EXACTLY
    /// `separation(z)` at every column past the first period — the wavelength that
    /// encodes depth. The only non-periodic region is the leftmost seed strip
    /// (`x < separation`), whose partner is off-screen.
    #[test]
    fn flat_depth_is_periodic_with_the_separation() {
        let (w, h) = (400u32, 8u32);
        let opts = StereoOpts { eye_sep: 120, mu: 0.33, near_is_high: true };
        let depth = vec![0.5f32; (w * h) as usize];
        let img = autostereogram(&depth, w, h, Bounds { lo: 0.0, hi: 1.0 }, &opts);
        let s = separation(0.5, opts.eye_sep as f32, opts.mu) as usize;
        assert!(s > 4);
        for y in 0..h as usize {
            for x in s..w as usize {
                assert_eq!(&img[(y * w as usize + x) * 3..][..3], &img[(y * w as usize + (x - s)) * 3..][..3],
                    "flat depth must repeat with period {s} at ({x},{y})");
            }
        }
    }

    /// The TEXTURED variant is periodic on flat depth (that periodicity is what makes
    /// it fuse) AND its wallpaper is sampled from the image's CENTRE strip, so the
    /// surface shows the photo. The seed strip (first period) must equal the image's
    /// centre band, and the rest repeats with the separation.
    #[test]
    fn textured_is_periodic_and_seeded_from_the_image_centre() {
        let (w, h) = (400u32, 8u32);
        let opts = StereoOpts { eye_sep: 120, mu: 0.5, near_is_high: true };
        let depth = vec![0.5f32; (w * h) as usize];
        let src: Vec<u8> = (0..(w * h)).flat_map(|i| { let x = (i % w) as u8; [x, 255 - x, 128] }).collect();
        let img = autostereogram_textured(&depth, w, h, Bounds { lo: 0.0, hi: 1.0 }, &opts, &src);
        let s = separation(0.5, opts.eye_sep as f32, opts.mu) as usize;
        let tile = separation(0.0, opts.eye_sep as f32, opts.mu) as usize;
        let tx0 = (w as usize - tile) / 2;
        // The seed strip (x < s) tiles the image's centre band.
        for y in 0..h as usize {
            for x in 0..s {
                let sx = tx0 + (x % tile);
                assert_eq!(&img[(y * w as usize + x) * 3..][..3], &src[(y * w as usize + sx) * 3..][..3],
                    "seed strip must come from the image centre band at ({x},{y})");
            }
        }
        // The rest is periodic with the separation.
        for y in 0..h as usize {
            for x in s..w as usize {
                assert_eq!(&img[(y * w as usize + x) * 3..][..3], &img[(y * w as usize + (x - s)) * 3..][..3],
                    "textured must repeat with period {s} at ({x},{y})");
            }
        }
    }

    /// The DIBR stereo pair: output is 2w wide, fully filled (no black holes), and
    /// near objects are displaced OPPOSITELY in the two panes (the disparity that
    /// produces depth on fusion).
    #[test]
    fn stereo_pair_fills_and_displaces() {
        let (w, h) = (80u32, 8u32);
        // A near red bar on a GREY far field (grey so a black column = a real hole).
        let mut depth = vec![0.0f32; (w * h) as usize];
        let mut rgb = vec![100u8; (w * h * 3) as usize];
        for y in 0..h as usize { for x in 0..w as usize {
            if (38..42).contains(&x) { depth[y*w as usize+x] = 1.0; rgb[(y*w as usize+x)*3..][..3].copy_from_slice(&[255,0,0]); }
        }}
        let out = stereo_pair(&rgb, &depth, w, h, Bounds{lo:0.0,hi:1.0}, 16, true);
        assert_eq!(out.len(), (2*w*h*3) as usize);
        // No black hole column: every column has some non-black pixel (the grey bg
        // or the bar), so an all-black column would mean the fill missed it.
        let ow = 2*w as usize;
        for x in 0..ow {
            let col_black = (0..h as usize).all(|y| out[(y*ow+x)*3..][..3]==[0,0,0]);
            assert!(!col_black, "column {x} is an unfilled black streak");
        }
        // The red bar's centroid differs between the two panes (it was displaced).
        let red_centroid = |x0:usize,x1:usize| -> f32 {
            let (mut sum, mut n) = (0f32, 0f32);
            for y in 0..h as usize { for x in x0..x1 {
                if out[(y*ow+x)*3] > 200 && out[(y*ow+x)*3+1] < 80 { sum += (x-x0) as f32; n += 1.0; }
            }}
            if n>0.0 { sum/n } else { -1.0 }
        };
        let lc = red_centroid(0, w as usize);
        let rc = red_centroid(w as usize, ow);
        assert!(lc >= 0.0 && rc >= 0.0, "the red bar must appear in both panes ({lc},{rc})");
        assert!((lc - rc).abs() > 2.0, "near object must be displaced between panes: {lc} vs {rc}");
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

// ---------------------------------------------------------------------------
// Cross-eye stereo PAIR (depth-image-based rendering) — shows the actual image.
// ---------------------------------------------------------------------------

/// A cross-eye stereo pair (`left | right`, `2w × h`) synthesized from the image and
/// its depth by Depth-Image-Based Rendering: each pixel is displaced horizontally by
/// a disparity proportional to its (normalized) depth — nearer moves more — forward-
/// warped with a per-pixel z-test so nearer pixels win, and disocclusion holes are
/// filled from neighbours. Free-view CROSS-EYED (right eye on the LEFT image) and the
/// actual scene appears in 3D — unlike an autostereogram, you see the real photo.
///
/// `max_disparity` is the peak shift in px (a knob, since depth is relative). Larger
/// = more relief but bigger holes and harder fusion.
pub fn stereo_pair(rgb: &[u8], depth: &[f32], w: u32, h: u32, bounds: Bounds, max_disparity: u32, near_is_high: bool) -> Vec<u8> {
    assert_eq!(rgb.len(), (w * h * 3) as usize, "rgb must be [h*w*3]");
    let (wi, hi) = (w as usize, h as usize);
    let ow = wi * 2;
    let md = max_disparity as f32;
    let mut out = vec![0u8; ow * hi * 3];
    let (mut lrow, mut rrow) = (vec![0u8; wi * 3], vec![0u8; wi * 3]);
    let (mut lz, mut rz) = (vec![f32::NEG_INFINITY; wi], vec![f32::NEG_INFINITY; wi]);
    let (mut lf, mut rf) = (vec![false; wi], vec![false; wi]);
    for y in 0..hi {
        lz.iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
        rz.iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
        lf.iter_mut().for_each(|v| *v = false);
        rf.iter_mut().for_each(|v| *v = false);
        for x in 0..wi {
            let n = bounds.norm(depth[y * wi + x]);
            let z = if near_is_high { n } else { 1.0 - n };
            let d = (md * z * 0.5).round() as i32; // half-shift each side
            let px = [rgb[(y * wi + x) * 3], rgb[(y * wi + x) * 3 + 1], rgb[(y * wi + x) * 3 + 2]];
            // LEFT pane is the right-eye view (near shifts right); RIGHT pane the
            // left-eye view (near shifts left) — the cross-eye convention.
            let xl = x as i32 + d;
            let xr = x as i32 - d;
            if xl >= 0 && (xl as usize) < wi && z > lz[xl as usize] {
                lz[xl as usize] = z;
                lf[xl as usize] = true;
                lrow[(xl as usize) * 3..][..3].copy_from_slice(&px);
            }
            if xr >= 0 && (xr as usize) < wi && z > rz[xr as usize] {
                rz[xr as usize] = z;
                rf[xr as usize] = true;
                rrow[(xr as usize) * 3..][..3].copy_from_slice(&px);
            }
        }
        fill_holes(&mut lrow, &lf, wi);
        fill_holes(&mut rrow, &rf, wi);
        out[(y * ow) * 3..(y * ow + wi) * 3].copy_from_slice(&lrow);
        out[(y * ow + wi) * 3..(y * ow + 2 * wi) * 3].copy_from_slice(&rrow);
    }
    out
}

/// Fill unwritten (disoccluded) pixels in a row from the nearest written neighbour:
/// interior/trailing holes copy from the left (the background just uncovered), and
/// leading holes copy from the first written pixel.
fn fill_holes(row: &mut [u8], filled: &[bool], w: usize) {
    let get = |row: &[u8], x: usize| [row[x * 3], row[x * 3 + 1], row[x * 3 + 2]];
    let mut last: Option<[u8; 3]> = None;
    for x in 0..w {
        if filled[x] {
            last = Some(get(row, x));
        } else if let Some(c) = last {
            row[x * 3..][..3].copy_from_slice(&c);
        }
    }
    // Leading holes (before the first filled pixel) get the first filled colour.
    if let Some(fx) = (0..w).find(|&x| filled[x]) {
        let c = get(row, fx);
        for x in 0..fx {
            row[x * 3..][..3].copy_from_slice(&c);
        }
    }
}
