// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! DeepSeek-VL's image preprocessing (`VLMImageProcessor` plus the hybrid
//! tower's own resize and per-branch normalization).
//!
//! 1. The image is scaled (Pillow bicubic, antialiased) so its long side is
//!    `image_size`, the short side floored and at least `min_size`.
//! 2. It is pasted centred on an `image_size` square filled with the
//!    background colour `int(image_mean * 255)`.
//! 3. The bytes are rescaled to `[0, 1]`. The processor does not normalize
//!    (`do_normalize: false`); each tower branch does, with its own mean and
//!    standard deviation.
//! 4. The low-resolution branch sees the square resized to its own side by
//!    torch's antialiased bilinear filter.

use imaging::host::{resize_aa_planar, resize_bicubic_pil, AaFilter};
use imaging::pixels::Rgb8;
use serde_json::Value;

/// `preprocessor_config.json`: the fields the processor reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageProcessor {
    pub image_size: u32,
    pub min_size: u32,
    /// The square's fill, `int(image_mean * 255)` per channel.
    pub background: [u8; 3],
    pub rescale_factor: f64,
}

impl ImageProcessor {
    pub fn from_json(v: &Value) -> Result<ImageProcessor, String> {
        let num = |k: &str| v.get(k).and_then(Value::as_f64).ok_or_else(|| format!("preprocessor_config.json: missing number '{k}'"));
        if v.get("do_normalize").and_then(Value::as_bool) != Some(false) {
            return Err("preprocessor_config.json: `do_normalize` must be false; the tower branches normalize".into());
        }
        let mean = v.get("image_mean").and_then(Value::as_array).filter(|a| a.len() == 3).ok_or("preprocessor_config.json: 'image_mean' is not three numbers")?;
        let mut background = [0u8; 3];
        for (b, m) in background.iter_mut().zip(mean) {
            let m = m.as_f64().ok_or("preprocessor_config.json: 'image_mean' is not three numbers")?;
            // Python's `int(x * 255)`: truncation toward zero.
            *b = (m * 255.0) as u8;
        }
        Ok(ImageProcessor { image_size: num("image_size")? as u32, min_size: num("min_size")? as u32, background, rescale_factor: num("rescale_factor")? })
    }

    pub fn from_dir(dir: &std::path::Path) -> Result<ImageProcessor, String> {
        let path = dir.join("preprocessor_config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        ImageProcessor::from_json(&serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?)
    }

    /// The processor's `pixel_values` for one image: planar `[3, S, S]` in
    /// `[0, 1]`, `S = image_size`.
    pub fn pixel_values(&self, img: &Rgb8) -> Result<Vec<f32>, String> {
        let (w, h) = (img.w as usize, img.h as usize);
        if w == 0 || h == 0 {
            return Err(format!("image of {w}x{h} pixels"));
        }
        let side = self.image_size as usize;
        let long = w.max(h) as f64;
        let scaled = |n: usize| ((n as f64 / long * side as f64) as usize).max(self.min_size as usize);
        let (nw, nh) = (scaled(w), scaled(h));
        let resized = resize_bicubic_pil(img, nw, nh);
        // `expand2square`: centred on the long axis, the offset floored.
        let (ox, oy) = ((side - nw.min(side)) / 2, (side - nh.min(side)) / 2);
        let plane = side * side;
        let mut out = vec![0f32; 3 * plane];
        let scale = |v: u8| (v as f64 * self.rescale_factor) as f32;
        for c in 0..3 {
            out[c * plane..(c + 1) * plane].fill(scale(self.background[c]));
        }
        for y in 0..nh.min(side) {
            for x in 0..nw.min(side) {
                let src = (y * nw + x) * 3;
                let dst = (y + oy) * side + x + ox;
                for c in 0..3 {
                    out[c * plane + dst] = scale(resized.px[src + c]);
                }
            }
        }
        Ok(out)
    }
}

/// Normalize a planar `[3, H, W]` image in place: `(x - mean) / std`.
pub fn normalize(px: &mut [f32], mean: [f32; 3], std: [f32; 3]) {
    let plane = px.len() / 3;
    for (c, chunk) in px.chunks_mut(plane).enumerate() {
        for v in chunk {
            *v = (*v - mean[c]) / std[c];
        }
    }
}

/// The hybrid tower's resize of the `[3, side, side]` square to the
/// low-resolution branch's `[3, low, low]` (torchvision `Resize(low,
/// antialias=True)`, bilinear).
pub fn low_resolution(px: &[f32], side: usize, low: usize) -> Vec<f32> {
    resize_aa_planar(px, 3, side, side, low, low, AaFilter::Bilinear)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> ImageProcessor {
        ImageProcessor::from_json(&serde_json::json!({
            "background_color": [122, 116, 104], "do_normalize": false, "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_size": 16, "min_size": 2, "rescale_factor": 0.00392156862745098
        }))
        .unwrap()
    }

    #[test]
    fn the_background_is_the_truncated_mean() {
        assert_eq!(processor().background, [122, 116, 104]);
    }

    #[test]
    fn a_wide_image_is_centred_vertically_on_the_background() {
        let p = processor();
        // 8x2 of white: scaled to 16x4, rows 6..10 of the 16x16 square.
        let img = Rgb8 { w: 8, h: 2, px: vec![255; 8 * 2 * 3] };
        let px = p.pixel_values(&img).unwrap();
        let row = |y: usize| px[y * 16..(y + 1) * 16].to_vec();
        assert!(row(6).iter().chain(&row(9)).all(|&v| v == 1.0), "the image rows are white");
        let bg = (122.0 * p.rescale_factor) as f32;
        assert!(row(5).iter().chain(&row(10)).all(|&v| v == bg), "the rows around it are the background");
    }

    #[test]
    fn normalizing_a_config_that_already_normalizes_is_refused() {
        let v = serde_json::json!({"do_normalize": true, "image_mean": [0.5, 0.5, 0.5], "image_size": 16, "min_size": 2, "rescale_factor": 0.1});
        assert!(ImageProcessor::from_json(&v).unwrap_err().contains("do_normalize"));
    }
}
