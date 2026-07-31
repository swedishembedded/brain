// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! INT8 calibration: run brain's own fp32 YOLO over representative images and
//! collect each quantized conv's input-activation range, then derive symmetric
//! per-tensor scales. Reuses the `yolo::net::ActTap` seam.

use std::cell::RefCell;
use std::collections::HashMap;

use yolo::net::ActTap;
use yolo::Yolo;

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
pub fn calibrate_from_weights(weights_path: &str, inputs: &[Vec<f32>]) -> Quant {
    let model = Yolo::load(weights_path, 1);
    calibrate(&model, inputs)
}

/// Load up to `max_n` calibration images from `dir`, each letterboxed to
/// `input×input` CHW (the model input layout). `dir` may be a brain detection
/// dataset (has `meta.json`) or a directory of binary PPM (P6) images.
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
            let (lbchw, _) = yolo::boxmath::letterbox_rgb(&hwc, ds.w, ds.h, input, 114.0 / 255.0);
            out.push(lbchw);
        }
    } else {
        // Directory of P6 PPM images.
        let mut files: Vec<_> = std::fs::read_dir(p)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "ppm" || e == "p6").unwrap_or(false))
            .collect();
        files.sort();
        for f in files.into_iter().take(max_n) {
            let bytes = std::fs::read(&f)?;
            if let Ok((px, w, h)) = events::ppm::decode_p6(&bytes) {
                let hwc: Vec<f32> = px.iter().map(|&b| b as f32 / 255.0).collect();
                let (lbchw, _) = yolo::boxmath::letterbox_rgb(&hwc, w, h, input, 114.0 / 255.0);
                out.push(lbchw);
            }
        }
    }
    Ok(out)
}
