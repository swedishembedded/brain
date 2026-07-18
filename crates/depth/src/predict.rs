// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single-image / single-frame depth inference, matching the reference's
//! preprocessing so brain's output matches the reference PyTorch's.
//!
//! The reference does NOT letterbox to a fixed square. It resizes the image
//! preserving aspect ratio so the SHORTER side is `input` (384), rounds both dims
//! to a multiple of 32, feeds that RECTANGULAR input to the (fully convolutional)
//! model, and resizes the depth back to the original resolution
//! (`zipdepth/inference/predictor.py`). Letterboxing to a padded square — which an
//! earlier version of this did — both downscales more (wasting resolution on the
//! pad) and feeds the network grey borders, visibly degrading the depth. The model
//! itself is exact against the reference (`tests/p3_reference_rect.rs`); getting the
//! preprocessing right is what makes the whole pipeline match.
//!
//! The model is rebuilt when the target size changes (cached otherwise), so a
//! fixed-resolution camera stream builds once and a still image builds once.

use std::cell::RefCell;

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::ParamStore;
use vision::Ctx;

use crate::config::ZipConfig;
use crate::model::ZipDepth;

/// Round `v` to the nearest multiple of `m`, at least `m` — the reference's
/// `make_divisible`.
fn make_divisible(v: f32, m: u32) -> u32 {
    let r = ((v / m as f32).round() as u32) * m;
    r.max(m)
}

/// The reference's target size: resize so the shorter side is `input`, both dims
/// rounded to a multiple of 32, aspect preserved.
pub fn target_size(w0: u32, h0: u32, input: u32) -> (u32, u32) {
    let scale = input as f32 / w0.min(h0) as f32;
    (make_divisible(h0 as f32 * scale, 32), make_divisible(w0 as f32 * scale, 32))
}

/// Bilinear resize of an interleaved-RGB HWC `[h0*w0*3]` image to `th × tw`,
/// `align_corners=false` (`half_pixel`), matching the reference's `cv2` /
/// `F.interpolate`.
fn resize_hwc(src: &[f32], w0: u32, h0: u32, tw: u32, th: u32) -> Vec<f32> {
    let mut out = vec![0f32; (tw * th * 3) as usize];
    let sx = w0 as f32 / tw as f32;
    let sy = h0 as f32 / th as f32;
    for y in 0..th {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, h0 as f32 - 1.0);
        let (y0, ty) = (fy.floor() as u32, fy - fy.floor());
        let y1 = (y0 + 1).min(h0 - 1);
        for x in 0..tw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, w0 as f32 - 1.0);
            let (x0, tx) = (fx.floor() as u32, fx - fx.floor());
            let x1 = (x0 + 1).min(w0 - 1);
            for c in 0..3u32 {
                let p = |xx: u32, yy: u32| src[((yy * w0 + xx) * 3 + c) as usize];
                let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
                let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
                out[((y * tw + x) * 3 + c) as usize] = top * (1.0 - ty) + bot * ty;
            }
        }
    }
    out
}

/// Bilinear resize of a single-channel `[h0*w0]` map to `th × tw`.
fn resize_map(src: &[f32], w0: u32, h0: u32, tw: u32, th: u32) -> Vec<f32> {
    let mut out = vec![0f32; (tw * th) as usize];
    let sx = w0 as f32 / tw as f32;
    let sy = h0 as f32 / th as f32;
    for y in 0..th {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, h0 as f32 - 1.0);
        let (y0, ty) = (fy.floor() as u32, fy - fy.floor());
        let y1 = (y0 + 1).min(h0 - 1);
        for x in 0..tw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, w0 as f32 - 1.0);
            let (x0, tx) = (fx.floor() as u32, fx - fx.floor());
            let x1 = (x0 + 1).min(w0 - 1);
            let p = |xx: u32, yy: u32| src[(yy * w0 + xx) as usize];
            let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
            let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
            out[(y * tw + x) as usize] = top * (1.0 - ty) + bot * ty;
        }
    }
    out
}

/// A model built for one target size, plus its input buffer.
struct Built {
    th: u32,
    tw: u32,
    model: ZipDepth,
    input: DeviceBuffer,
}

/// A ready-to-run ZipDepth predictor. The model is (re)built lazily when the target
/// input size changes, so a fixed-resolution stream compiles once.
pub struct Predictor<'g> {
    gpu: &'g Gpu,
    ps: ParamStore,
    cfg: ZipConfig,
    built: RefCell<Option<Built>>,
}

impl<'g> Predictor<'g> {
    pub fn new(gpu: &'g Gpu, cfg: ZipConfig, ps: ParamStore) -> Predictor<'g> {
        Predictor { gpu, ps, cfg, built: RefCell::new(None) }
    }

    pub fn input_size(&self) -> u32 {
        self.cfg.input
    }

    /// Predict depth for an interleaved-RGB HWC frame in `[0,1]`, returning a
    /// `[h0*w0]` inverse-depth map on the frame's own grid.
    ///
    /// Aspect-preserving resize to the reference target (shorter side = `input`,
    /// ×32) — NOT a letterboxed square — then the model forward, then a bilinear
    /// resize of the depth back to `w0 × h0`. The model normalizes internally, so
    /// the input stays `[0,1]`.
    pub fn predict(&self, hwc: &[f32], w0: u32, h0: u32) -> Vec<f32> {
        assert_eq!(hwc.len(), (w0 * h0 * 3) as usize, "hwc must be [h0*w0*3] RGB");
        let (th, tw) = target_size(w0, h0, self.cfg.input);

        // (Re)build the model if the target size changed.
        {
            let needs = self.built.borrow().as_ref().map(|b| (b.th, b.tw) != (th, tw)).unwrap_or(true);
            if needs {
                let ctx = Ctx::new(self.gpu, crate::net::ids());
                let model = ZipDepth::build_hw(&ctx, self.cfg.clone(), 1, th, tw, false);
                model.set_eval(true);
                let input = self.gpu.storage((3 * th * tw) as u64);
                *self.built.borrow_mut() = Some(Built { th, tw, model, input });
            }
        }

        // Resize to (th, tw), pack HWC -> CHW.
        let resized = resize_hwc(hwc, w0, h0, tw, th);
        let mut chw = vec![0f32; (3 * th * tw) as usize];
        let hw = (th * tw) as usize;
        for y in 0..th as usize {
            for x in 0..tw as usize {
                for c in 0..3 {
                    chw[c * hw + y * tw as usize + x] = resized[(y * tw as usize + x) * 3 + c];
                }
            }
        }

        let b = self.built.borrow();
        let b = b.as_ref().unwrap();
        self.gpu.write(&b.input, bytemuck::cast_slice(&chw));
        let ctx = Ctx::new(self.gpu, crate::net::ids());
        b.model.forward(&ctx, &self.ps, &b.input);
        let depth_t = self.gpu.read(b.model.out(), hw);

        // Resize the depth back to the original frame grid.
        resize_map(&depth_t, tw, th, w0, h0)
    }
}
