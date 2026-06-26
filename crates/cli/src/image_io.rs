// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared image loading for the YOLO / NPU CLIs. Accepts a binary PPM (P6) file
//! or a brain detection-dataset directory (image 0), returning interleaved-RGB
//! HWC f32 in `[0,1]`.

use std::path::Path;

use data::gen_detect::load_dataset;

/// Load an image as `(hwc, w, h)`. A directory is treated as a detection dataset
/// (image 0); a file must be a binary PPM (P6).
pub fn load_image(path: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let p = Path::new(path);
    if p.is_dir() {
        let data = load_dataset(p).map_err(|e| format!("loading dataset {path}: {e}"))?;
        if data.n == 0 {
            return Err("dataset has no images".into());
        }
        let stride = data.image_stride();
        let hwc = chw_to_hwc(&data.images[..stride], 3, data.h as usize, data.w as usize);
        return Ok((hwc, data.w, data.h));
    }
    let bytes = std::fs::read(p).map_err(|e| format!("reading {path}: {e}"))?;
    if bytes.starts_with(b"P6") {
        let (px, w, h) = events::ppm::decode_p6(&bytes)?;
        let hwc: Vec<f32> = px.iter().map(|&b| b as f32 / 255.0).collect();
        return Ok((hwc, w, h));
    }
    Err(format!("{path}: unsupported image (send a binary PPM 'P6' file or a detection dataset dir)"))
}

/// CHW `[c,h,w]` → interleaved HWC `[h,w,c]`.
pub fn chw_to_hwc(chw: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; c * h * w];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ch] = chw[ch * h * w + y * w + x];
            }
        }
    }
    out
}
