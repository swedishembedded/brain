// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reference-exact image preprocessing.
//!
//! The upstream pipeline is: PIL `Image.resize(BICUBIC)` on uint8 RGB →
//! `ToTensor()` (`/255`) → center-crop → (inside the model) ImageNet
//! mean/std. PIL's 8-bit resampler is a **fixed-point** separable convolution
//! (PRECISION_BITS = 22, horizontal then vertical, antialiased on downscale),
//! which differs from any float bicubic by up to 1 LSB per pixel — so this
//! ports Pillow's `Resample.c` arithmetic exactly (T1 gates it against a PIL
//! golden bit-for-bit).

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Interleaved 8-bit RGB image.
pub struct RgbImage {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

/// Binary P6 PPM loader (maxval 255).
pub fn load_ppm(path: &str) -> Result<RgbImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut fields = Vec::new();
    let mut pos = 0usize;
    // header: P6, width, height, maxval — whitespace/comment separated
    while fields.len() < 4 && pos < bytes.len() {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        fields.push(std::str::from_utf8(&bytes[start..pos]).map_err(|_| "bad header")?.to_string());
    }
    pos += 1; // single whitespace after maxval
    if fields.len() != 4 || fields[0] != "P6" || fields[3] != "255" {
        return Err(format!("{path}: need binary P6 maxval 255, got {fields:?}"));
    }
    let w: usize = fields[1].parse().map_err(|_| "bad width")?;
    let h: usize = fields[2].parse().map_err(|_| "bad height")?;
    if bytes.len() < pos + w * h * 3 {
        return Err(format!("{path}: truncated pixel data"));
    }
    Ok(RgbImage { w, h, rgb: bytes[pos..pos + w * h * 3].to_vec() })
}

/// `_calculate_resize_dims` (crop strategy): longest side → `target`, the
/// other side scaled and rounded to a multiple of `patch`.
pub fn resize_dims(orig_w: usize, orig_h: usize, target: usize, patch: usize) -> (usize, usize) {
    if orig_w >= orig_h {
        let new_h = ((orig_h as f64 * (target as f64 / orig_w as f64) / patch as f64).round()
            as usize)
            * patch;
        (target, new_h)
    } else {
        let new_w = ((orig_w as f64 * (target as f64 / orig_h as f64) / patch as f64).round()
            as usize)
            * patch;
        (new_w, target)
    }
}

/// `compute_adaptive_target_size`: min(longest edge, cap) floored to /patch.
pub fn adaptive_target(orig_w: usize, orig_h: usize, cap: usize, patch: usize) -> usize {
    let effective = orig_w.max(orig_h).min(cap) / patch * patch;
    effective.max(patch * 2)
}

// ---- Pillow 8bpc resampling (Resample.c) ----

const PRECISION_BITS: i32 = 32 - 8 - 2; // 22

/// PIL bicubic filter, a = -0.5, support 2.0.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// `precompute_coeffs`: per output position, the source window `[xmin, xmin+n)`
/// and its normalized fixed-point weights (i32 << PRECISION_BITS).
fn coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    const SUPPORT: f64 = 2.0;
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let inv = 1.0 / filterscale;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support).floor().max(0.0)) as usize;
        let xmax = (((center + support).ceil()) as usize).min(in_size) - xmin;
        let mut w = vec![0.0f64; xmax];
        let mut ww = 0.0f64;
        for (x, wx) in w.iter_mut().enumerate() {
            *wx = bicubic((x as f64 + xmin as f64 - center + 0.5) * inv);
            ww += *wx;
        }
        for (x, &wx) in w.iter().enumerate() {
            let k = wx / ww;
            kk[xx * ksize + x] = if k < 0.0 {
                (-0.5 + k * (1i64 << PRECISION_BITS) as f64) as i32
            } else {
                (0.5 + k * (1i64 << PRECISION_BITS) as f64) as i32
            };
        }
        bounds.push((xmin, xmax));
    }
    (ksize, bounds, kk)
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// PIL-exact bicubic resize of interleaved RGB8 (horizontal pass, then
/// vertical, both in fixed point).
pub fn resize_bicubic(img: &RgbImage, nw: usize, nh: usize) -> RgbImage {
    let (w, h) = (img.w, img.h);
    // horizontal
    let horiz: Vec<u8> = if nw != w {
        let (ksize, bounds, kk) = coeffs(w, nw);
        let mut out = vec![0u8; nw * h * 3];
        for y in 0..h {
            for x in 0..nw {
                let (xmin, xmax) = bounds[x];
                for ch in 0..3 {
                    let mut ss = 1i32 << (PRECISION_BITS - 1);
                    for i in 0..xmax {
                        ss += img.rgb[(y * w + xmin + i) * 3 + ch] as i32 * kk[x * ksize + i];
                    }
                    out[(y * nw + x) * 3 + ch] = clip8(ss);
                }
            }
        }
        out
    } else {
        img.rgb.clone()
    };
    // vertical
    let vert: Vec<u8> = if nh != h {
        let (ksize, bounds, kk) = coeffs(h, nh);
        let mut out = vec![0u8; nw * nh * 3];
        for y in 0..nh {
            let (ymin, ymax) = bounds[y];
            for x in 0..nw {
                for ch in 0..3 {
                    let mut ss = 1i32 << (PRECISION_BITS - 1);
                    for i in 0..ymax {
                        ss += horiz[((ymin + i) * nw + x) * 3 + ch] as i32 * kk[y * ksize + i];
                    }
                    out[(y * nw + x) * 3 + ch] = clip8(ss);
                }
            }
        }
        out
    } else {
        horiz
    };
    RgbImage { w: nw, h: nh, rgb: vert }
}

/// Full reference preprocessing of one frame: resize (crop strategy) →
/// center-crop to ≤ target per axis → `/255` → ImageNet normalize → planar
/// CHW f32. Returns (chw, out_w, out_h).
pub fn preprocess(img: &RgbImage, target: usize, patch: usize) -> (Vec<f32>, usize, usize) {
    let (nw, nh) = resize_dims(img.w, img.h, target, patch);
    let resized = resize_bicubic(img, nw, nh);
    let (cw, ch_) = (nw.min(target), nh.min(target));
    let (x0, y0) = ((nw - cw) / 2, (nh - ch_) / 2);
    let mut chw = vec![0.0f32; 3 * cw * ch_];
    for c in 0..3 {
        for y in 0..ch_ {
            for x in 0..cw {
                let v = resized.rgb[((y0 + y) * nw + x0 + x) * 3 + c] as f32 / 255.0;
                chw[c * cw * ch_ + y * cw + x] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    (chw, cw, ch_)
}
