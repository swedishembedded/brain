// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ELU activation Step-builders, plus a CPU reference oracle used as a test
//! oracle. Same shape as [`crate::conv`]/[`crate::snake`]: pure dispatch
//! assembly over `kernels::ELU`/`kernels::ELU_BWD` - shapes + buffers in,
//! `Step`s out. Added for CosyVoice's `ConvRNNF0Predictor` (5 conv layers
//! with ELU activation, despite the name no RNN), but kept general: any
//! future model that needs ELU dispatches through the same pair.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Kernel-pipeline indices a model supplies from its own kernel list.
#[derive(Clone, Copy)]
pub struct ActKernels {
    pub fwd: usize,
    pub bwd: usize,
}

/// `y = x` if `x > 0`, else `alpha*(exp(x)-1)` (PyTorch `nn.ELU()`; alpha =
/// 1.0 is that default).
pub fn elu_fwd(g: &Gpu, k: &ActKernels, x: &DeviceBuffer, y: &DeviceBuffer, total: u32, alpha: f32) -> Step {
    g.step(k.fwd, &[x, y], &[total, alpha.to_bits()], total)
}

/// Input gradient `dx` (overwritten).
pub fn elu_bwd(g: &Gpu, k: &ActKernels, x: &DeviceBuffer, dy: &DeviceBuffer, dx: &DeviceBuffer, total: u32, alpha: f32) -> Step {
    g.step(k.bwd, &[x, dy, dx], &[total, alpha.to_bits()], total)
}

/// CPU reference forward, matching `wgsl/elu.wgsl`.
pub fn elu_ref(x: &[f32], alpha: f32) -> Vec<f32> {
    x.iter().map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }).collect()
}
