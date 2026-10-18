// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniCPM-V 4.6's image processor (`MiniCPMV4_6ImageProcessor`): an image
//! becomes a SOURCE view - the whole image resized to a patch grid that is a
//! multiple of 4 - plus, when it is large enough, a grid of SLICES cut from a
//! second, refine-size resize. Every view keeps its own aspect ratio.
//!
//! The arithmetic is the reference's, including its Python rounding: `round`
//! is ties-to-even, `int()` truncates, ratios are f64. The resize is torch's
//! uint8 antialiased bicubic ([`imaging::host::resize_bicubic_torch_u8`]),
//! byte for byte, and pixels are normalized as the reference's fused
//! rescale-and-normalize does it: `(x - 127.5) / 127.5` in f32.

use std::path::Path;

use imaging::Rgb8;
use serde_json::Value;

use crate::config::Downsample;
use crate::vision::View;

/// The processor's knobs, from `preprocessor_config.json` and the request.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageOptions {
    pub max_slices: u32,
    /// Target side of one slice, in pixels (448).
    pub scale_resolution: u32,
    pub slice_mode: bool,
    pub use_image_id: bool,
    pub downsample: Downsample,
    pub patch: u32,
}

impl ImageOptions {
    /// `dir/preprocessor_config.json`, the reference processor class's own
    /// defaults for keys it leaves out, and the model's downsample mode.
    pub fn from_dir(dir: &Path, patch: u32, downsample: Downsample) -> Result<ImageOptions, String> {
        let path = dir.join("preprocessor_config.json");
        let v: Value = match std::fs::read(&path) {
            Ok(b) => serde_json::from_slice(&b).map_err(|e| format!("{}: {e}", path.display()))?,
            Err(_) => Value::Null,
        };
        let u = |k: &str, d: u32| v.get(k).and_then(Value::as_u64).map_or(d, |x| x as u32);
        let b = |k: &str, d: bool| v.get(k).and_then(Value::as_bool).unwrap_or(d);
        let file_patch = u("patch_size", patch);
        if file_patch != patch {
            return Err(format!("{}: patch_size {file_patch} disagrees with the vision tower's {patch}", path.display()));
        }
        for (k, want) in [("image_mean", 0.5), ("image_std", 0.5)] {
            if let Some(a) = v.get(k).and_then(Value::as_array) {
                if a.iter().any(|x| x.as_f64() != Some(want)) {
                    return Err(format!("{}: {k} {a:?} - only the released 0.5 normalization is implemented", path.display()));
                }
            }
        }
        Ok(ImageOptions {
            max_slices: u("max_slice_nums", 9),
            scale_resolution: u("scale_resolution", 448),
            slice_mode: b("slice_mode", true),
            use_image_id: b("use_image_id", true),
            downsample,
            patch,
        })
    }
}

/// One image after processing.
#[derive(Clone, Debug)]
pub struct ProcessedImage {
    /// The source view first, then the slices row-major.
    pub views: Vec<View>,
    /// `[sum h*w, 3*p*p]` patch rows for [`crate::vision::Tower::encode`].
    pub patches: Vec<f32>,
    /// `(rows, cols)` of the slice grid; `None` when the image is not sliced.
    pub grid: Option<(u32, u32)>,
    /// The resized images (the source, then the refine image before it is
    /// cut), for parity checks.
    pub resized: Vec<Rgb8>,
}

/// Python's `max(round(length / divisor) * divisor, divisor)`.
fn ensure_divide(length: f64, divisor: u32) -> u32 {
    (((length / divisor as f64).round_ties_even() as i64) * divisor as i64).max(divisor as i64) as u32
}

/// The reference's `find_best_resize`, for an `(h, w)` that may be fractional.
fn find_best_resize(h: f64, w: f64, scale: u32, patch: u32, allow_upscale: bool) -> (u32, u32) {
    let (mut h, mut w) = (h, w);
    if h * w > (scale as f64) * (scale as f64) || allow_upscale {
        let aspect = w / h;
        let nh = (scale as f64 / aspect.sqrt()) as i64;
        let nw = (nh as f64 * aspect) as i64;
        h = nh as f64;
        w = nw as f64;
    }
    (ensure_divide(h, patch * 4), ensure_divide(w, patch * 4))
}

/// The reference's `get_refine_size`.
fn refine_size(h: u32, w: u32, grid: (u32, u32), scale: u32, patch: u32) -> (u32, u32) {
    let (gy, gx) = grid;
    let rw = ensure_divide(w as f64, gx);
    let rh = ensure_divide(h as f64, gy);
    let (bh, bw) = find_best_resize(rh as f64 / gy as f64, rw as f64 / gx as f64, scale, patch, true);
    (bh * gy, bw * gx)
}

/// The reference's `get_sliced_grid`: `(rows, cols)`, or `None` for an image
/// small enough to stay whole. Among slice counts from `multiple - 1` to
/// `multiple + 1` it picks the factorization whose aspect is closest to the
/// image's in log space, preferring more rows on a tie.
pub fn sliced_grid(h: u32, w: u32, max_slices: u32, scale: u32) -> Option<(u32, u32)> {
    let log_ratio = (w as f64 / h as f64).ln();
    let ratio = (w as f64 * h as f64) / (scale as f64 * scale as f64);
    let multiple = (ratio.ceil() as u32).min(max_slices);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1u32, 1u32);
    let mut min_error = f64::INFINITY;
    for slices in [multiple - 1, multiple, multiple + 1] {
        if slices == 1 || slices > max_slices {
            continue;
        }
        for rows in 1..=slices {
            if slices % rows == 0 {
                let cols = slices / rows;
                let error = (log_ratio - (cols as f64 / rows as f64).ln()).abs();
                if error < min_error {
                    best = (rows, cols);
                    min_error = error;
                } else if error == min_error && rows > best.0 {
                    best = (rows, cols);
                }
            }
        }
    }
    Some(best)
}

/// The `[3, h, w]` normalized planes of an RGB8 image.
fn normalize(img: &Rgb8) -> Vec<f32> {
    let n = (img.w * img.h) as usize;
    let mut out = vec![0f32; 3 * n];
    for (i, px) in img.px.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            out[c * n + i] = (px[c] as f32 - 127.5) / 127.5;
        }
    }
    out
}

/// A `ch x cw` crop of `img` at pixel `(y0, x0)`.
fn crop(img: &Rgb8, y0: u32, x0: u32, ch: u32, cw: u32) -> Rgb8 {
    let mut px = Vec::with_capacity((ch * cw * 3) as usize);
    for y in y0..y0 + ch {
        let row = ((y * img.w + x0) * 3) as usize;
        px.extend_from_slice(&img.px[row..row + (cw * 3) as usize]);
    }
    Rgb8 { w: cw, h: ch, px }
}

fn view_of(img: &Rgb8, patch: u32) -> View {
    View { h: img.h / patch, w: img.w / patch }
}

/// Process one image: the source view, then the slices of the refine image
/// row-major, as patch rows.
pub fn process_image(img: &Rgb8, opts: &ImageOptions) -> ProcessedImage {
    let (h, w) = (img.h, img.w);
    assert!(h > 0 && w > 0, "minicpmv: an empty image");
    let grid = if opts.slice_mode { sliced_grid(h, w, opts.max_slices, opts.scale_resolution) } else { None };
    let (sh, sw) = find_best_resize(h as f64, w as f64, opts.scale_resolution, opts.patch, grid.is_none());
    let source = imaging::host::resize_bicubic_torch_u8(img, sw as usize, sh as usize);
    let mut views = vec![view_of(&source, opts.patch)];
    let mut patches = crate::vision::patchify(&normalize(&source), views[0], opts.patch);
    let mut resized = vec![source];
    if let Some((rows, cols)) = grid {
        let (rh, rw) = refine_size(h, w, (rows, cols), opts.scale_resolution, opts.patch);
        let refine = imaging::host::resize_bicubic_torch_u8(img, rw as usize, rh as usize);
        let (ph, pw) = (rh / rows, rw / cols);
        for r in 0..rows {
            for c in 0..cols {
                let slice = crop(&refine, r * ph, c * pw, ph, pw);
                let view = view_of(&slice, opts.patch);
                patches.extend(crate::vision::patchify(&normalize(&slice), view, opts.patch));
                views.push(view);
            }
        }
        resized.push(refine);
    }
    ProcessedImage { views, patches, grid, resized }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_rounding_is_reproduced() {
        // round(2.5) == 2 and round(3.5) == 4 in Python.
        assert_eq!(ensure_divide(2.5 * 56.0, 56), 112);
        assert_eq!(ensure_divide(3.5 * 56.0, 56), 224);
        assert_eq!(ensure_divide(10.0, 56), 56, "never below one divisor");
    }

    #[test]
    fn the_grid_prefers_the_closest_aspect_and_more_rows_on_a_tie() {
        assert_eq!(sliced_grid(448, 448, 9, 448), None);
        assert_eq!(sliced_grid(768, 1024, 9, 448), Some((2, 2)));
        // 2 slices fit 1600x200 worse (1x2, error ln 4) than 3 do (1x3, error ln 8/3).
        assert_eq!(sliced_grid(200, 1600, 9, 448), Some((1, 3)));
    }
}
