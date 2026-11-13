// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a window's depth comes from: brain's engine (the GPU or the CPU,
//! as `--device` places it) or the Intel NPU.
//!
//! The windows drive the model directly rather than through one `infer`
//! call per frame: a live camera keeps the engine busy on the next frame
//! while the last one is drawn, which a request/response call cannot.

use gpu_core::Gpu;
use zipdepth::{Predictor, ZipConfig};

/// The checkpoint, with an optional smaller model input.
pub fn config(weights: &str, input: Option<i64>) -> Result<ZipConfig, String> {
    let mut cfg = zipdepth::cfg_for_checkpoint(weights)?;
    if let Some(n) = input.filter(|&n| n > 0) {
        // fully convolutional: any side works, the predictor rounds it to 32
        cfg.input = n as u32;
    }
    Ok(cfg)
}

/// The engine's predictor for `weights`, on `gpu`.
pub fn engine<'g>(gpu: &'g Gpu, weights: &str, cfg: &ZipConfig) -> Result<Predictor<'g>, String> {
    let params = zipdepth::load_into(gpu, weights, cfg)?;
    Ok(Predictor::new(gpu, cfg.clone(), params))
}

/// ZipDepth compiled for `w x h` frames on the NPU.
pub fn npu(weights: &str, cfg: &ZipConfig, w: u32, h: u32) -> Result<npu::DepthSession, String> {
    let session = npu::DepthSession::compile(weights, cfg, w, h)?;
    eprintln!("depth: ZipDepth compiled for {w}x{h} on {}", session.device());
    Ok(session)
}
