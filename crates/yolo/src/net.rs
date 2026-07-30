// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8's kernel registry: the `PIPELINES` array and its index constants.
//!
//! The block plumbing this module used to define — `Shape`, `Ctx`, `ActTap` —
//! now lives in [`vision`], so the conv blocks can be shared with other vision
//! models rather than forked. It is re-exported below; this module keeps only
//! what is genuinely yolo's: the pipeline array and its frozen index order.
//!
//! ## Why the index constants stay
//!
//! They are no longer the blocks' interface (see [`ids`]), but yolo's own
//! loss/optim dispatch still uses them and the order is frozen by the checkpoint
//! contract, so they are not dead weight — they are simply private to yolo now.

use std::sync::OnceLock;

use vision::ConvKernelIds;

// `Shape`, `Ctx` and `ActTap` now live in `crates/vision` so the conv blocks can
// be shared with other vision models. Re-exported here permanently: `brain-npu`
// imports `yolo::net::ActTap`, and `yolo::lib` re-exports `net::{Ctx, Shape}`.
pub use vision::{ActTap, Ctx, Shape};

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
// ---- fused conv + per-channel bias (detection head) ----
pub const CONV_BIAS: usize = 40;
/// `out += a`, single read_write binding (index 41). ADD2 cannot accumulate
/// into one of its own inputs: binding a buffer read-only AND read-write in
/// one dispatch is a wgpu usage-scope violation.
pub const ADD_INPLACE: usize = 41;

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
    // ---- fused conv + bias (index 40) ----
    ("conv_bias", kernels::CONV_BIAS),
    // ---- accumulate-in-place (index 41) ----
    ("add_inplace", kernels::ADD_INPLACE),
    // ---- conv-as-GEMM eval fast path (im2col + matmul_reg2 + conv_epilogue) ----
    ("im2col", kernels::IM2COL),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("conv_epilogue", kernels::CONV_EPILOGUE),
    // Cooperative grad-norm (optimiser): `gradnorm_part` + `clip_coef_wg` replace
    // the single-threaded `gradnorm_sq`/`clip_coef` walk. `optim::Optim` resolves
    // them BY NAME, so appending them here (and only here) is the whole opt-in.
    ("gradnorm_part", kernels::GRADNORM_PART),
    ("clip_coef_wg", kernels::CLIP_COEF_WG),
];

/// Kernel indices for the shared [`vision`] conv blocks, resolved BY NAME against
/// [`PIPELINES`] above — so the blocks never depend on this array's order.
///
/// The positional `const`s at the top of this module are NOT the blocks'
/// interface any more; they remain because yolo's own loss/optim dispatch and its
/// checkpoint contract freeze the index order. That is precisely why resolution
/// is by name: a second model's pipeline is ordered differently, and a shared
/// block holding one of these literals would dispatch the wrong kernel under it.
pub fn ids() -> &'static ConvKernelIds {
    static IDS: OnceLock<ConvKernelIds> = OnceLock::new();
    IDS.get_or_init(|| ConvKernelIds::resolve(PIPELINES))
}
