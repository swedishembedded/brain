// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Snake activation Step-builders (the DAC-style codec vocoder's
//! single-parameter periodic activation - `y = x + (alpha+eps)^-1 *
//! sin(alpha*x)^2`, distinct from `kernels::SNAKE_BETA`'s two-parameter,
//! log-space BigVGAN v2 form), plus a CPU reference oracle used as a test
//! oracle. Same shape as [`crate::conv`]: pure dispatch assembly, shapes +
//! buffers in, `Step`s out.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Shape of one Snake activation call: `[rows, C, inner]` (NCL: `inner` =
/// length). Channel index `c = (idx / inner) % C`.
#[derive(Clone, Copy, Debug)]
pub struct Snake1d {
    pub rows: u32,
    pub c: u32,
    pub inner: u32,
    pub eps: f32,
}

impl Snake1d {
    pub fn total(&self) -> u32 {
        self.rows * self.c * self.inner
    }
    fn fwd_params(&self) -> [u32; 4] {
        [self.total(), self.c, self.inner, self.eps.to_bits()]
    }
}

/// Kernel-pipeline indices a model supplies from its own kernel list.
#[derive(Clone, Copy)]
pub struct SnakeKernels {
    pub fwd: usize,
    pub bwd_dx: usize,
    pub bwd_dalpha: usize,
}

/// `y = x + (alpha[c]+eps)^-1 * sin(alpha[c]*x)^2`.
pub fn snake1d_fwd(g: &Gpu, k: &SnakeKernels, c: &Snake1d, x: &DeviceBuffer, alpha: &DeviceBuffer, y: &DeviceBuffer) -> Step {
    g.step(k.fwd, &[x, alpha, y], &c.fwd_params(), c.total())
}

/// Input gradient `dx` (overwritten).
pub fn snake1d_bwd_dx(g: &Gpu, k: &SnakeKernels, c: &Snake1d, dy: &DeviceBuffer, x: &DeviceBuffer, alpha: &DeviceBuffer, dx: &DeviceBuffer) -> Step {
    g.step(k.bwd_dx, &[dy, x, alpha, dx], &c.fwd_params(), c.total())
}

/// Per-channel `alpha` gradient (written, not accumulated - one dispatch per
/// backward pass owns the whole buffer).
pub fn snake1d_bwd_dalpha(g: &Gpu, k: &SnakeKernels, c: &Snake1d, dy: &DeviceBuffer, x: &DeviceBuffer, alpha: &DeviceBuffer, dalpha: &DeviceBuffer) -> Step {
    let params = [c.rows, c.c, c.inner, c.eps.to_bits()];
    g.step(k.bwd_dalpha, &[dy, x, alpha, dalpha], &params, c.c)
}

/// CPU reference forward, matching `wgsl/snake1d.wgsl`.
pub fn snake1d_ref(c: &Snake1d, x: &[f32], alpha: &[f32]) -> Vec<f32> {
    (0..c.total() as usize)
        .map(|idx| {
            let ch = (idx / c.inner as usize) % c.c as usize;
            let a = alpha[ch];
            let s = (a * x[idx]).sin();
            x[idx] + (1.0 / (a + c.eps)) * s * s
        })
        .collect()
}
