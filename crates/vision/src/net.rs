// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared plumbing for conv-net blocks: NCHW shapes, the activation tap, and the
//! block-build context.
//!
//! ## Block abstraction
//!
//! Every block is constructed once (registering its parameter *names* and
//! pre-allocating SSA activation + grad buffers), then appends forward [`Step`]s
//! to a shared `Vec<Step>` and, separately, backward `Step`s (in reverse order)
//! to another.
//!
//! SSA discipline: every forward stage writes a FRESH buffer, which doubles as
//! the activation cache the backward pass reads. Multi-consumer / residual
//! gradients accumulate out-of-place via the `add2` kernel into fresh buffers.
//!
//! Buffers come from two places:
//!   * weights + their grads from a `ParamStore` (keyed by the names each block
//!     registers — see the per-block `param_list` helpers), and
//!   * activations + backward temporaries from plain [`Gpu::storage`].
//!
//! The [`Ctx`] bundles the `&Gpu`, the resolved [`ConvKernelIds`] and the
//! activation allocator, so a block says `ctx.act(n)` for a fresh buffer and
//! `ctx.step(ctx.ids.conv2d, ...)` for a dispatch.
//!
//! NOTE: blocks do not uniformly pre-record and replay. BatchNorm in train mode
//! needs a host-side interleave between `bn_stats` and `bn_train` (the host reads
//! mean/var between two submits), so those forwards run imperatively. Preserve
//! that split — collapsing the two submits into one reads stale statistics, and
//! only a train-mode value pin catches it.

use gpu_core::{Gpu, Step};

use crate::ids::ConvKernelIds;

/// An NCHW feature-map shape. Carried alongside buffers so blocks can compute
/// thread counts and the spatial dims kernels need.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

impl Shape {
    pub fn new(n: u32, c: u32, h: u32, w: u32) -> Shape {
        Shape { n, c, h, w }
    }
    /// Element count `N*C*H*W`.
    pub fn numel(&self) -> u32 {
        self.n * self.c * self.h * self.w
    }
    /// Output shape of a `K x K` conv with the given stride/pad.
    pub fn conv_out(&self, cout: u32, k: u32, stride: u32, pad: u32) -> Shape {
        let ho = (self.h + 2 * pad - k) / stride + 1;
        let wo = (self.w + 2 * pad - k) / stride + 1;
        Shape::new(self.n, cout, ho, wo)
    }
    /// Output shape of a dilated `K x K` conv: the kernel's effective extent is
    /// `dilation*(k-1) + 1`. Equals [`Self::conv_out`] at `dilation == 1`.
    pub fn conv_out_dilated(
        &self,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        dilation: u32,
    ) -> Shape {
        let eff = dilation * (k - 1) + 1;
        let ho = (self.h + 2 * pad - eff) / stride + 1;
        let wo = (self.w + 2 * pad - eff) / stride + 1;
        Shape::new(self.n, cout, ho, wo)
    }
}

/// An activation tap: observe (and optionally rewrite, in place) each conv's
/// input activation during a forward pass. Used by the NPU INT8 quantizer
/// (`brain-npu`) to (a) collect per-conv activation ranges for calibration
/// (read-only) and (b) simulate the INT8 quant→dequant effect in fp32 for the
/// hardware-free accuracy-parity gate (in-place rewrite). `name` is the conv's
/// unique prefix, which MUST match the exported ONNX node name — the calibrator
/// keys its scale map on it, so a prefix-format change silently maps ranges to
/// the wrong tensors.
///
/// The tap is only consulted on the eval-mode (inference) forward path, and only
/// when one is installed via [`Ctx::with_tap`] — every normal inference runs with
/// no tap and pays zero cost.
pub trait ActTap {
    fn tap(&self, name: &str, x: &mut [f32]);
}

/// Block-build context: a thin wrapper over the device that hands out fresh
/// activation buffers and records dispatch [`Step`]s. Held by reference while a
/// block records its forward/backward steps.
pub struct Ctx<'g> {
    pub gpu: &'g Gpu,
    /// Kernel indices resolved by NAME against the owning model's own
    /// `PIPELINES` — see [`ConvKernelIds::resolve`]. This is what lets one set of
    /// blocks serve models whose pipeline lists differ in order and content.
    pub ids: &'g ConvKernelIds,
    /// Optional activation tap (calibration / fake-quant). `None` on every
    /// normal forward — see [`ActTap`].
    pub tap: Option<&'g dyn ActTap>,
}

impl<'g> Ctx<'g> {
    pub fn new(gpu: &'g Gpu, ids: &'g ConvKernelIds) -> Ctx<'g> {
        Ctx { gpu, ids, tap: None }
    }
    /// A context whose conv forwards route their input through the given
    /// [`ActTap`]. Used only by the NPU calibrator / fake-quant simulator.
    pub fn with_tap(gpu: &'g Gpu, ids: &'g ConvKernelIds, tap: &'g dyn ActTap) -> Ctx<'g> {
        Ctx { gpu, ids, tap: Some(tap) }
    }
    /// A fresh activation / temporary buffer of `n` f32 elements.
    pub fn act(&self, n: u32) -> gpu_core::DeviceBuffer {
        self.gpu.storage(n as u64)
    }
    /// Record a dispatch with `u32` uniform params (use [`gpu_core::f`] to pack
    /// an f32 into the stream).
    pub fn step(
        &self,
        kind: usize,
        bufs: &[&gpu_core::DeviceBuffer],
        params: &[u32],
        threads: u32,
    ) -> Step {
        self.gpu.step(kind, bufs, params, threads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_out_matches_the_standard_formula() {
        let s = Shape::new(4, 3, 128, 128);
        assert_eq!(s.conv_out(16, 3, 2, 1), Shape::new(4, 16, 64, 64)); // stem
        assert_eq!(s.conv_out(8, 1, 1, 0), Shape::new(4, 8, 128, 128)); // 1x1
        assert_eq!(s.numel(), 4 * 3 * 128 * 128);
    }

    #[test]
    fn conv_out_dilated_agrees_with_conv_out_at_dilation_one() {
        let s = Shape::new(1, 96, 48, 48);
        for &(k, stride, pad) in &[(3u32, 1u32, 1u32), (1, 1, 0), (3, 2, 1), (5, 1, 2)] {
            assert_eq!(
                s.conv_out(96, k, stride, pad),
                s.conv_out_dilated(96, k, stride, pad, 1),
                "dilation=1 must be the identity case for k={k} s={stride} p={pad}"
            );
        }
    }

    /// ZipDepth's MinimalMultiScale: depthwise 3x3 dilation 2 with pad 2 is
    /// shape-preserving, exactly like dilation 1 with pad 1.
    #[test]
    fn conv_out_dilated_is_shape_preserving_for_the_dilated_depthwise_branch() {
        let s = Shape::new(1, 96, 48, 48);
        assert_eq!(s.conv_out_dilated(96, 3, 1, 2, 2), Shape::new(1, 96, 48, 48));
    }
}
