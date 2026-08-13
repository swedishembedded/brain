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

/// The reference's preprocessing, as one function: aspect-preserving bilinear
/// resize so the shorter side is `input` with both dims rounded to a multiple of
/// 32, then HWC -> CHW. Returns the tensor and its `(th, tw)`.
///
/// Extracted so **calibration runs the transform inference actually uses**.
/// `brain depth calib` previously letterboxed to a padded square — a different
/// resampler, a different geometry and a grey fill the model never sees — so it
/// collected INT8 activation ranges from a distribution that does not occur at
/// inference. There is no second copy of this: [`Predictor::begin`] and
/// `zipdepth::quant` both call it.
pub fn preprocess_chw(hwc: &[f32], w0: u32, h0: u32, input: u32) -> (Vec<f32>, u32, u32) {
    let (th, tw) = target_size(w0, h0, input);
    let resized = imaging::resize_bilinear_hwc(hwc, 3, w0, h0, tw, th);
    let hw = (th * tw) as usize;
    let mut chw = vec![0f32; 3 * hw];
    // Channel-parallel: each channel plane strides through `resized` independently.
    backend_cpu::par::rows_mut(&mut chw, hw, |c, plane| {
        for (i, v) in plane.iter_mut().enumerate() {
            *v = resized[i * 3 + c];
        }
    });
    (chw, th, tw)
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
    /// The in-flight frame started by [`Predictor::begin`]: the source frame
    /// geometry `(w0, h0)` and the model grid `(tw, th)` the depth must be
    /// unwarped from. `None` when no frame is pending.
    pending: RefCell<Option<(u32, u32, u32, u32)>>,
}

impl<'g> Predictor<'g> {
    pub fn new(gpu: &'g Gpu, cfg: ZipConfig, ps: ParamStore) -> Predictor<'g> {
        Predictor { gpu, ps, cfg, built: RefCell::new(None), pending: RefCell::new(None) }
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
        self.begin(hwc, w0, h0);
        self.finish()
    }

    /// Start inference for a frame: preprocess (host), upload, record the
    /// forward and FLUSH it to the device — then return WITHOUT waiting. The
    /// caller overlaps its own host work (capture, colorize, render of the
    /// previous result) with the device compute and collects via
    /// [`Predictor::finish`]. A second `begin` before `finish` would overwrite
    /// the in-flight frame's input buffer, so it panics.
    pub fn begin(&self, hwc: &[f32], w0: u32, h0: u32) {
        assert_eq!(hwc.len(), (w0 * h0 * 3) as usize, "hwc must be [h0*w0*3] RGB");
        assert!(self.pending.borrow().is_none(), "begin() called with a frame already in flight");
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

        let (chw, _, _) = preprocess_chw(hwc, w0, h0, self.cfg.input);

        let b = self.built.borrow();
        let b = b.as_ref().unwrap();
        self.gpu.write(&b.input, bytemuck::cast_slice(&chw));
        let ctx = Ctx::new(self.gpu, crate::net::ids());
        b.model.forward(&ctx, &self.ps, &b.input);
        // Send the recorded forward to the device now (no wait) — this is what
        // buys the overlap; without it the work would only be submitted by the
        // blocking read in `finish`.
        self.gpu.flush();
        *self.pending.borrow_mut() = Some((w0, h0, tw, th));
    }

    /// Collect the frame started by [`Predictor::begin`]: block on the device,
    /// read the depth and unwarp it onto the frame's own grid.
    pub fn finish(&self) -> Vec<f32> {
        let (w0, h0, tw, th) =
            self.pending.borrow_mut().take().expect("finish() without a begin()");
        let b = self.built.borrow();
        let b = b.as_ref().unwrap();
        let depth_t = self.gpu.read(b.model.out(), (th * tw) as usize);
        imaging::resize_bilinear_hwc(&depth_t, 1, tw, th, w0, h0)
    }

    /// Whether a `begin` is awaiting its `finish`.
    pub fn in_flight(&self) -> bool {
        self.pending.borrow().is_some()
    }
}
