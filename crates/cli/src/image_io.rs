// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared image loading for the YOLO / NPU CLIs. Accepts an image file (PPM P6,
//! PNG or JPEG) or a brain detection-dataset directory (image 0), returning
//! interleaved-RGB HWC f32 in `[0,1]`.
//!
//! The decode and the layout permutation both come from `imaging`; the only
//! thing that lives here is the *policy* of what a bare path means to a CLI —
//! which is genuinely CLI-specific and has no second copy.

use std::path::Path;

use data::gen_detect::load_dataset;

/// Load an image as `(hwc, w, h)`. A directory is treated as a detection dataset
/// (image 0); a file is decoded by `imaging` (P6 / PNG / JPEG).
pub fn load_image(path: &str) -> Result<(Vec<f32>, u32, u32), String> {
    let p = Path::new(path);
    if p.is_dir() {
        let data = load_dataset(p).map_err(|e| format!("loading dataset {path}: {e}"))?;
        if data.n == 0 {
            return Err("dataset has no images".into());
        }
        let stride = data.image_stride();
        let hwc = imaging::pixels::chw_to_hwc(
            &data.images[..stride],
            3,
            data.h as usize,
            data.w as usize,
        );
        return Ok((hwc, data.w, data.h));
    }
    let img = imaging::load(p)?;
    Ok((img.to_hwc_unit(), img.w, img.h))
}
