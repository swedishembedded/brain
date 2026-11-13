// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 calibration: run brain's own fp32 YOLO over representative images and
//! collect each quantized conv's input-activation range, then derive symmetric
//! per-tensor scales. Reuses the `yolov8::net::ActTap` seam.

use std::cell::RefCell;
use std::collections::HashMap;

use yolov8::net::ActTap;
use yolov8::Yolo;

use crate::quant::{symmetric_act_scale, Quant};

/// Collects per-conv `(min, max)` activation ranges across a calibration set.
#[derive(Default)]
pub struct RangeCollector {
    ranges: RefCell<HashMap<String, (f32, f32)>>,
}

impl ActTap for RangeCollector {
    fn tap(&self, name: &str, x: &mut [f32]) {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in x.iter() {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        let mut r = self.ranges.borrow_mut();
        let e = r.entry(name.to_string()).or_insert((f32::INFINITY, f32::NEG_INFINITY));
        e.0 = e.0.min(mn);
        e.1 = e.1.max(mx);
    }
}

impl RangeCollector {
    /// Convert the accumulated ranges into symmetric per-tensor activation scales.
    pub fn into_quant(self) -> Quant {
        let mut q = Quant::new();
        for (k, (mn, mx)) in self.ranges.into_inner() {
            q.act_scales.insert(k, symmetric_act_scale(mn, mx));
        }
        q
    }
}

/// Calibrate `model` (eval-mode) over the given letterboxed CHW inputs (each
/// `[3,input,input]` matching the model's input size), returning the INT8
/// activation scales.
pub fn calibrate(model: &Yolo, inputs: &[Vec<f32>]) -> Quant {
    assert!(!inputs.is_empty(), "calibration needs at least one image");
    model.set_eval(true);
    let collector = RangeCollector::default();
    for chw in inputs {
        model.set_image(chw);
        model.forward_net_tapped(&collector);
    }
    collector.into_quant()
}

/// Load the model from a checkpoint and calibrate over `inputs`.
/// Panics on an unreadable checkpoint or no inputs; [`try_calibrate_from_weights`]
/// reports them.
pub fn calibrate_from_weights(weights_path: &str, inputs: &[Vec<f32>]) -> Quant {
    try_calibrate_from_weights(weights_path, inputs).unwrap_or_else(|e| panic!("{e}"))
}

/// [`calibrate_from_weights`] reporting an unreadable checkpoint or an empty
/// calibration set as an error.
pub fn try_calibrate_from_weights(weights_path: &str, inputs: &[Vec<f32>]) -> Result<Quant, String> {
    if inputs.is_empty() {
        return Err("calibration needs at least one image".into());
    }
    let model = Yolo::try_load_at(weights_path, 1, 0)?;
    Ok(calibrate(&model, inputs))
}

/// Load up to `max_n` calibration images from `dir`, each letterboxed to
/// `input×input` CHW (the model input layout). `dir` may be a brain detection
/// dataset (has `meta.json`) or a directory of images (PPM, PNG, JPEG), read in
/// name order; files that are not images are skipped.
pub fn load_calib_images(dir: &str, input: u32, max_n: usize) -> std::io::Result<Vec<Vec<f32>>> {
    let p = std::path::Path::new(dir);
    let mut out = Vec::new();
    if p.join("meta.json").exists() {
        // brain detection dataset: CHW [0,1] images.
        let ds = data::gen_detect::load_dataset(p)?;
        let stride = ds.image_stride();
        for i in 0..ds.n.min(max_n) {
            let chw = &ds.images[i * stride..(i + 1) * stride];
            let hwc = imaging::pixels::chw_to_hwc(chw, 3, ds.h as usize, ds.w as usize);
            let (lbchw, _) = yolov8::boxmath::letterbox_rgb(&hwc, ds.w, ds.h, input, 114.0 / 255.0);
            out.push(lbchw);
        }
    } else {
        let mut files: Vec<_> = std::fs::read_dir(p)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect();
        files.sort();
        for f in files {
            if out.len() >= max_n {
                break;
            }
            let Ok(img) = imaging::load(&f) else { continue };
            let (lbchw, _) = yolov8::boxmath::letterbox_rgb(&img.to_hwc_unit(), img.w, img.h, input, 114.0 / 255.0);
            out.push(lbchw);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: u32, h: u32) -> imaging::Rgb8 {
        imaging::Rgb8 { w, h, px: vec![120; (w * h * 3) as usize] }
    }

    /// A folder of mixed image formats is read in name order, a file that is no
    /// image is skipped, and every image comes out letterboxed to the model's
    /// input side.
    #[test]
    fn a_folder_of_images_is_letterboxed_to_the_model_input() {
        let dir = std::env::temp_dir().join(format!("npu-calib-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        imaging::save_png(dir.join("a.png"), &image(40, 20)).unwrap();
        imaging::save_ppm(dir.join("b.ppm"), &image(16, 16)).unwrap();
        std::fs::write(dir.join("c.txt"), b"not an image").unwrap();
        let all = load_calib_images(dir.to_str().unwrap(), 32, 10).unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.iter().all(|c| c.len() == 3 * 32 * 32));
        assert_eq!(load_calib_images(dir.to_str().unwrap(), 32, 1).unwrap().len(), 1, "max_n caps the set");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn calibration_without_images_or_a_checkpoint_is_an_error() {
        assert!(try_calibrate_from_weights("/nonexistent/w.safetensors", &[vec![0.0; 3]]).unwrap_err().contains("w.safetensors"));
        assert!(try_calibrate_from_weights("/nonexistent/w.safetensors", &[]).unwrap_err().contains("at least one"));
    }
}
