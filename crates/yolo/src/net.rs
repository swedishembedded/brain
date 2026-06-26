// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared plumbing for the YOLOv8 conv-net blocks (P2).
//!
//! This module defines the kernel-index registry and a [`Ctx`] helper that the
//! conv blocks ([`crate::blocks`]) and detection head ([`crate::head`]) share.
//!
//! ## Block abstraction
//!
//! Every block mirrors the [`gpt::Gpt`](../../gpt/src/model.rs) pattern: it is
//! constructed once (registering its parameters and pre-allocating SSA
//! activation + grad buffers), then it appends forward [`Step`]s to a shared
//! `Vec<Step>` and, separately, backward `Step`s (in reverse order) to another
//! shared vector. The blocks themselves never run anything; the owning model
//! records `fwd_steps`/`bwd_steps` once and replays them via
//! [`Gpu::submit`].
//!
//! SSA discipline: every forward stage writes a FRESH buffer, which doubles as
//! the activation cache the backward pass reads. Multi-consumer / residual
//! gradients accumulate out-of-place via the `add2` kernel into fresh buffers.
//!
//! Buffers come from two places, exactly as in `gpt`:
//!   * weights + their grads from a [`ParamStore`] (keyed by the names each
//!     block registers — see the per-block `param_list` helpers), and
//!   * activations + backward temporaries from plain [`Gpu::storage`].
//!
//! The [`Ctx`] bundles the `&Gpu` and the activation allocator so a block can
//! say `ctx.act(n)` for a fresh activation buffer and `ctx.step(KERNEL, ...)`.

use gpu_core::{Gpu, Step};

// ---- kernel indices (order MUST match `PIPELINES`) ----
pub const CONV2D: usize = 0;
pub const CONV2D_DX: usize = 1;
pub const CONV2D_DW: usize = 2;
pub const BN_STATS: usize = 3;
pub const BN_RUNNING: usize = 4;
pub const BN_TRAIN: usize = 5;
pub const BN_EVAL: usize = 6;
pub const BN_DSTATS: usize = 7;
pub const BN_DX: usize = 8;
pub const BN_DGAMMA: usize = 9;
pub const BN_DBETA: usize = 10;
pub const SILU: usize = 11;
pub const SILU_BWD: usize = 12;
pub const MAXPOOL5: usize = 13;
pub const MAXPOOL5_DX: usize = 14;
pub const UPSAMPLE2: usize = 15;
pub const UPSAMPLE2_DX: usize = 16;
pub const CONCAT2: usize = 17;
pub const CONCAT_SPLIT: usize = 18;
pub const ADD2: usize = 19;
pub const ADAMW: usize = 20;
pub const GRADNORM_SQ: usize = 21;
pub const GRAD_SCALE: usize = 22;
pub const CLIP_COEF: usize = 23;
pub const GRAD_SCALE_BUF: usize = 24;
// ---- P4 detection-loss kernels ----
pub const DFL_DECODE: usize = 25;
pub const DFL_GRAD: usize = 26;
pub const DFL_LOSS: usize = 27;
pub const DFL_LOSS_GRAD: usize = 28;
pub const CIOU: usize = 29;
pub const CIOU_GRAD: usize = 30;
pub const BCE_LOGITS: usize = 31;
pub const BCE_LOGITS_GRAD: usize = 32;
// ---- head bias (P12): per-output-channel bias on the final 1x1 head conv ----
pub const BIAS_ADD: usize = 33;
pub const BIAS_GRAD: usize = 34;
// ---- fused conv->BN(eval)->SiLU for inference (appended; keeps prior indices) ----
pub const CONV_ACT: usize = 35;
// ---- single-pass channel-concat placement (replaces the O(n^2) concat fold) ----
pub const CHAN_PLACE: usize = 36;
// ---- weight-tiled (workgroup-memory) conv variants for the GPU backend ----
pub const CONV2D_TILED: usize = 37;
pub const CONV_ACT_TILED: usize = 38;
// ---- register-tiled fused conv (4 output channels per invocation) ----
pub const CONV_ACT_REG: usize = 39;

/// Kernel registry passed to [`Gpu::new`] / [`Gpu::new_cpu`]. The position of
/// each entry is its kernel index (the `const`s above).
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    ("bn_stats", kernels::BN_STATS),
    ("bn_running", kernels::BN_RUNNING),
    ("bn_train", kernels::BN_TRAIN),
    ("bn_eval", kernels::BN_EVAL),
    ("bn_dstats", kernels::BN_DSTATS),
    ("bn_dx", kernels::BN_DX),
    ("bn_dgamma", kernels::BN_DGAMMA),
    ("bn_dbeta", kernels::BN_DBETA),
    ("silu", kernels::SILU),
    ("silu_bwd", kernels::SILU_BWD),
    ("maxpool5", kernels::MAXPOOL5),
    ("maxpool5_dx", kernels::MAXPOOL5_DX),
    ("upsample2", kernels::UPSAMPLE2),
    ("upsample2_dx", kernels::UPSAMPLE2_DX),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
    ("add2", kernels::ADD2),
    ("adamw", kernels::ADAMW),
    ("gradnorm_sq", kernels::GRADNORM_SQ),
    ("grad_scale", kernels::GRAD_SCALE),
    ("clip_coef", kernels::CLIP_COEF),
    ("grad_scale_buf", kernels::GRAD_SCALE_BUF),
    // ---- P4 detection-loss kernels (indices 25..=32) ----
    ("dfl_decode", kernels::DFL_DECODE),
    ("dfl_grad", kernels::DFL_GRAD),
    ("dfl_loss", kernels::DFL_LOSS),
    ("dfl_loss_grad", kernels::DFL_LOSS_GRAD),
    ("ciou", kernels::CIOU),
    ("ciou_grad", kernels::CIOU_GRAD),
    ("bce_logits", kernels::BCE_LOGITS),
    ("bce_logits_grad", kernels::BCE_LOGITS_GRAD),
    // ---- head bias (P12): indices 33..=34 ----
    ("bias_add", kernels::BIAS_ADD),
    ("bias_grad", kernels::BIAS_GRAD),
    // ---- fused conv->BN(eval)->SiLU (index 35) ----
    ("conv_act", kernels::CONV_ACT),
    // ---- single-pass channel-concat placement (index 36) ----
    ("chan_place", kernels::CHAN_PLACE),
    // ---- weight-tiled conv variants (index 37..=38) ----
    ("conv2d_tiled", kernels::CONV2D_TILED),
    ("conv_act_tiled", kernels::CONV_ACT_TILED),
    // ---- register-tiled fused conv (index 39) ----
    ("conv_act_reg", kernels::CONV_ACT_REG),
];

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
}

/// Block-build context: a thin wrapper over the device that hands out fresh
/// activation buffers and records dispatch [`Step`]s. Held by reference while a
/// block records its forward/backward steps.
pub struct Ctx<'g> {
    pub gpu: &'g Gpu,
}

impl<'g> Ctx<'g> {
    pub fn new(gpu: &'g Gpu) -> Ctx<'g> {
        Ctx { gpu }
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
