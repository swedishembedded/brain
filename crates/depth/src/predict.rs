// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Single-image / single-frame depth inference: letterbox an arbitrary-size RGB
//! frame into the model's square input, run the eval forward, and unwarp the depth
//! map back onto the frame's OWN pixel grid — exactly as `Yolo::detect` returns
//! boxes in frame coordinates.
//!
//! The letterbox math is a compact copy rather than a dependency on
//! `yolo::boxmath`: depth is a peer model, not a consumer of yolo, and the
//! transform is ~20 lines. (A shared `vision::letterbox` is the eventual home if a
//! third model needs it.)

use gpu_core::{DeviceBuffer, Gpu};
use paramstore::ParamStore;
use vision::Ctx;

use crate::config::ZipConfig;
use crate::model::ZipDepth;

/// Aspect-preserving resize + centre-pad transform from a `w0 x h0` frame to a
/// `size x size` square.
#[derive(Clone, Copy, Debug)]
struct Letterbox {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    size: u32,
}

impl Letterbox {
    fn compute(w0: u32, h0: u32, size: u32) -> Letterbox {
        let scale = (size as f32 / w0 as f32).min(size as f32 / h0 as f32);
        let new_w = (w0 as f32 * scale).round();
        let new_h = (h0 as f32 * scale).round();
        Letterbox { scale, pad_x: (size as f32 - new_w) * 0.5, pad_y: (size as f32 - new_h) * 0.5, size }
    }
}

/// A ready-to-run ZipDepth predictor: the eval-mode model plus its weights, sized
/// for one image at `cfg.input`.
pub struct Predictor<'g> {
    gpu: &'g Gpu,
    model: ZipDepth,
    ps: ParamStore,
    cfg: ZipConfig,
    input: DeviceBuffer,
}

impl<'g> Predictor<'g> {
    /// Build a predictor from a loaded ParamStore (see [`crate::import::load_into`]).
    pub fn new(gpu: &'g Gpu, cfg: ZipConfig, ps: ParamStore) -> Predictor<'g> {
        let ctx = Ctx::new(gpu, crate::net::ids());
        let model = ZipDepth::build(&ctx, cfg.clone(), 1, false);
        model.set_eval(true);
        let input = gpu.storage((3 * cfg.input * cfg.input) as u64);
        Predictor { gpu, model, ps, cfg, input }
    }

    pub fn input_size(&self) -> u32 {
        self.cfg.input
    }

    /// Predict depth for an interleaved-RGB HWC frame in `[0,1]`, returning a
    /// `[h0*w0]` inverse-depth map on the frame's own grid (row-major).
    ///
    /// The pipeline: letterbox `hwc` into the model's `[3,S,S]` CHW input (the model
    /// applies ImageNet normalize internally, so the input stays `[0,1]`), run the
    /// forward, then for each original pixel sample the `S x S` depth map at that
    /// pixel's letterboxed location (bilinear). Padding regions of the square are
    /// never sampled — the unwarp only reads inside the content box.
    pub fn predict(&self, hwc: &[f32], w0: u32, h0: u32) -> Vec<f32> {
        assert_eq!(hwc.len(), (w0 * h0 * 3) as usize, "hwc must be [h0*w0*3] RGB");
        let s = self.cfg.input;
        let lb = Letterbox::compute(w0, h0, s);

        // Letterbox -> CHW [3,S,S], grey pad. Nearest-neighbour resize (the model
        // is robust to it and it avoids a second bilinear pass on the host).
        let sz = s as usize;
        let mut chw = vec![0.5f32; 3 * sz * sz];
        let inv = 1.0 / lb.scale;
        let new_w = (w0 as f32 * lb.scale).round() as usize;
        let new_h = (h0 as f32 * lb.scale).round() as usize;
        for yi in 0..new_h {
            let sy = (((yi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, h0 as f32 - 1.0)) as usize;
            let dy = yi + lb.pad_y as usize;
            for xi in 0..new_w {
                let sx = (((xi as f32 + 0.5) * inv - 0.5).round().clamp(0.0, w0 as f32 - 1.0)) as usize;
                let dx = xi + lb.pad_x as usize;
                let sbase = (sy * w0 as usize + sx) * 3;
                for c in 0..3 {
                    chw[c * sz * sz + dy * sz + dx] = hwc[sbase + c];
                }
            }
        }

        self.gpu.write(&self.input, bytemuck::cast_slice(&chw));
        let ctx = Ctx::new(self.gpu, crate::net::ids());
        self.model.forward(&ctx, &self.ps, &self.input);
        let depth_sq = self.gpu.read(self.model.out(), (s * s) as usize);

        // Unwarp: sample the S x S depth at each original pixel's letterboxed spot,
        // bilinearly. The content box is [pad_x, pad_x+new_w) x [pad_y, ...).
        let mut out = vec![0f32; (w0 * h0) as usize];
        for y in 0..h0 {
            let fy = (y as f32 + 0.5) * lb.scale + lb.pad_y - 0.5;
            for x in 0..w0 {
                let fx = (x as f32 + 0.5) * lb.scale + lb.pad_x - 0.5;
                out[(y * w0 + x) as usize] = sample_bilinear(&depth_sq, s, s, fx, fy);
            }
        }
        out
    }
}

/// Bilinear sample of a `[h*w]` map at `(fx, fy)`, edge-clamped.
fn sample_bilinear(map: &[f32], w: u32, h: u32, fx: f32, fy: f32) -> f32 {
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let cx = |v: f32| (v as i32).clamp(0, w as i32 - 1) as usize;
    let cy = |v: f32| (v as i32).clamp(0, h as i32 - 1) as usize;
    let (x0, x1) = (cx(x0), cx(x0 + 1.0));
    let (y0, y1) = (cy(y0), cy(y0 + 1.0));
    let at = |xx: usize, yy: usize| map[yy * w as usize + xx];
    let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
    let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
    top * (1.0 - ty) + bot * ty
}
