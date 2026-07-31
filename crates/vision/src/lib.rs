// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared conv-net building blocks for brain's vision models.
//!
//! ## Why this crate exists
//!
//! `crates/yolo` grew a set of genuinely generic conv blocks — `Conv`,
//! `Bottleneck`, `C2f`, `SPPF` — fully parameterized over channels, kernel,
//! stride and pad, with the channel schedule living entirely in `YoloConfig`.
//! Nothing about them is detection-specific. But they were unusable by any other
//! model because of **one** thing: they imported their kernel indices as
//! `use crate::net::{CONV2D, BN_STATS, ...}` — bare `usize` **positions** in
//! yolo's own `PIPELINES` array. A second model's pipeline list has a different
//! order, so those constants would silently dispatch the wrong kernels.
//!
//! That is not hypothetical. `wm-diamond` defines `K_NLC_NCHW = 8` in one module
//! and `= 18` in another for the same kernel, because it maintains two separate
//! pipeline arrays by hand.
//!
//! This crate breaks that coupling with [`ConvKernelIds::resolve`], which looks
//! every kernel up **by name** in whatever `PIPELINES` the owning model declares.
//! A model registers only the kernels it dispatches; anything it omits resolves
//! to [`NONE`] and panics *with the kernel's name* on use, rather than quietly
//! running whatever happened to sit at that index. Positional coupling becomes
//! structurally impossible rather than merely discouraged.
//!
//! The precedent is `model::block`, which does exactly this for the transformer
//! family (`KernelIds`: rmsnorm/rope/gqa/silu). That struct is deliberately not
//! widened here: its 16 fields are all Qwen-family, and none is a conv. Two
//! families, two seams.
//!
//! ## Why not `crates/model`
//!
//! `model::block`'s stated contract is "pure dispatch assembly — no WGSL, no
//! ParamStore, no buffer ownership": every function there is
//! `fn(&Gpu, &KernelIds, ..) -> Step` and owns nothing. A conv block is
//! categorically different — it owns device buffers, a name prefix, and
//! interior-mutable train/eval mode state. Putting it there would break that
//! module's invariant. `crates/model` is also a dependency of 14 crates, none of
//! which will ever dispatch a conv.
//!
//! The blocks do not implement `Model` and never touch the trainer. The one
//! thing this crate takes from `crates/model` is `model::block`'s **LayerNorm
//! dispatch seam** (`LayerNormIds` + `layernorm_fwd` / `ln_stats_fwd` /
//! `layernorm_dx_bwd`), used by [`blocks::LayerNorm2d`]: that seam is the single
//! place in the workspace that decides between the reference LayerNorm kernels
//! and the coalesced `*_rows` twins, keyed on the queried `DeviceCaps` via
//! `backend_api::select`. Re-deriving the choice here would be a second
//! selection site, free to drift from it — exactly the failure AGENTS.md names
//! ("a fast kernel a later model never learned about").

pub mod blocks;
pub mod bn;
pub mod fold;
pub mod ids;
pub mod net;
pub mod plumbing;

pub use blocks::{
    Act, Bottleneck, Conv, ConvNames, ConvSpec, ConvTrSpec, ConvTranspose, CxSpec, CXBlock,
    AvgPool, LayerNorm2d, Ln2dNames, MaxPool, NameStyle, Norm, PoolSpec, SppfSpec, C2f, PReLU,
    SPPF,
};
pub use bn::{BatchNorm, BnNames};
pub use fold::{fold_bn, BN_EPS};
pub use ids::{ConvKernelIds, NONE};
pub use net::{ActTap, Ctx, Shape};
pub use plumbing::{Acc, Cat, Up};
