// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The image substrate.
//!
//! ## Why this crate exists
//!
//! Image handling in brain was ~60 sites across 24 crates: three byte-identical
//! host bilinear resizes (`depth::predict`, `cli::depth_cli`, `cli::resident_depth`),
//! two independent P6 parsers, two letterboxes, fifteen CHW/HWC permutations, and
//! four inline `[0,1] <-> [-1,1]` maps. That is the `rmsnorm`-was-seven-times
//! failure mode from `AGENTS.md`, applied to pixels: every copy is a place the
//! `align_corners` convention, the pad fill, the channel order or the value range
//! can drift, and nothing compares the copies against each other.
//!
//! This crate is the one home. It is **not** a grab-bag: it holds exactly the
//! things that are image-shaped and model-agnostic.
//!
//! ## Being the home means the old copy is GONE, not shadowed
//!
//! A consolidation crate that only *adds* a definition raises the copy count.
//! Every item here therefore either has no predecessor, or the predecessor now
//! re-exports it:
//!
//! | item | previous site | state |
//! |---|---|---|
//! | [`letterbox::Letterbox`] / [`letterbox_rgb`] | `yolo::boxmath` | **moved**; `boxmath` re-exports, one definition |
//! | [`codec::decode_p6`] / [`codec::encode_p6`] | `events::ppm` | **re-exported**; `events` keeps the definition (it is wasm-reachable and must not gain `image`) |
//! | [`Shape`] | `vision::net` | **re-exported**; no second NCHW type |
//! | [`device::NONE`] | `vision::ids` | **re-exported**; one sentinel |
//! | [`Ctx`], [`mask`], [`tiling`], [`Normalization`] | — | net-new, or a kernel dispatch replacing host code |
//! | [`host::resize_bilinear_hwc`] | `depth::predict`, `cli::depth_cli`, `cli::resident_depth` | **moved**; six functions became one |
//! | [`color::yuyv_to_rgb`] | `capture::convert` | **moved**; `crates/capture` is V4L2-only again |
//! | [`pixels::chw_to_hwc`] / [`pixels::hwc_to_chw`] | `cli::image_io`, `npu::{calib,sim}`, `wm-display`, tests | **moved**; one generic pair |
//! | [`IMAGENET_MEAN`] / [`IMAGENET_STD`] | `depth::init`, `mirror::preprocess` | **moved**; one pair of arrays |
//! | [`Rgb8`] / [`codec::load`] | `mirror::preprocess::{RgbImage, load_ppm}` | **moved**; the second P6 parser is gone |
//!
//! `cli::depth_cli`'s calibration letterbox now calls [`letterbox_rgb`] with
//! `pad = 0.5` — bit-identical to the copy it replaced, which deliberately keeps
//! the ZipDepth INT8 scales stable while leaving survey §6.2 (calibration
//! preprocesses differently from inference) as its own gated fix.
//!
//! Still owed, each blocked on a numeric gate rather than on effort:
//! `zimage::pipeline::{feather_mask, downsample_mask}` (no in-tree inpaint
//! metric to gate the ramp against — see [`mask`]),
//! `zimage::caps::build_outpaint_canvas` (needs `pad2d.wgsl` to grow a
//! `pad_mode` word before edge-replication can be expressed), and
//! the fused `[0,1] -> [-1,1]` maps in `flux2::finetune` / `zimage::finetune` /
//! `zimage::pipeline` (host per-pixel arithmetic, which this crate deliberately
//! does not offer a host entry point for — see the table above).
//!
//! `data::{imageset, gen_detect}` cannot migrate at all: `imaging` -> `vision`
//! -> `model` -> `data`, so `data` depending on `imaging` is a dependency cycle.
//! Its `image` dependency therefore stays, and this crate's is a second (feature-
//! unified) declaration of the same crate rather than a second decoder.
//!
//! ## The line between this crate and a kernel
//!
//! `AGENTS.md`: *host math does not run on the accelerator*. A host loop over a
//! 4K image is invisible to `--device` and reports host numbers under a device
//! label. So:
//!
//! | kind of work | where it lives |
//! |---|---|
//! | per-pixel arithmetic over a whole image (resize, pad, crop, normalise, blur, dilate, composite) | a **kernel dispatch** — [`Ctx`], `crates/kernels/wgsl/*` |
//! | layout permutation, geometry, IO, codecs, policy | **host glue, here** |
//! | a reduction to a handful of scalars (mask IoU) | host, here — the readback dominates |
//!
//! Nothing in this crate re-implements a kernel on the host. The one host
//! resampler, [`letterbox::letterbox_rgb`], is here *because* it cannot be
//! dispatched: its nearest-neighbour rule is half-pixel
//! (`round((i+0.5)/scale - 0.5)`) and `resize_nearest.wgsl` is ONNX-asymmetric
//! (`floor(o*in/out)`). Those pick different source pixels for most ratios, so
//! retargeting it would move every YOLO detection. See that module.
//!
//! ## Subtly-different variants are typed parameters, never a silent choice
//!
//! Where the survey found two things that look like one function, the difference
//! is an explicit, documented parameter with a stated default:
//!
//! * [`AlignCorners`] — half-pixel vs corner-aligned resampling. Both are
//!   plausible, they differ by half a pixel, and a gradient check cannot tell
//!   them apart.
//! * [`Filter`] — nearest / bilinear / bicubic, each naming the exact reference
//!   function it reproduces. Note in particular that `Filter::Bicubic` is
//!   PyTorch's non-antialiased `a = -0.75` cubic and is **not** the same function
//!   as `mirror::preprocess::resize_bicubic` (PIL fixed-point, antialiased,
//!   `a = -0.5`) or `resize_bicubic_torch` (antialiased, f64). Antialiased
//!   downsampling has no kernel; this crate does not pretend otherwise.
//! * [`ChannelPolicy`] — what a 1-channel buffer means when RGB8 is wanted.
//! * [`Normalization`] — mean/std as data, so ImageNet, `(x-0.5)/0.5` and the
//!   identity are the same code path.
//! * [`letterbox::letterbox_rgb`]'s `pad` fill is a parameter (yolo passes
//!   `114/255`, depth's calibration passes `0.5`), never a constant.
//!
//! ## What this crate deliberately does NOT contain
//!
//! * **Model-specific target-size policies.** `depth::predict::target_size`,
//!   `mirror::preprocess::{resize_dims, adaptive_target}`,
//!   `qwenvl::preprocess::smart_resize` and `moondream::preprocess::select_tiling`
//!   are five different reference models' contracts. They may be hosted here
//!   side by side as named policies, but they must never be unified — and
//!   copying them here while the originals still exist would add five more
//!   duplicates, not remove any. They move when their callers move.
//! * **A host resize *for callers that hold a `Gpu`*.** `resize_bilinear.wgsl`
//!   with `AlignCorners::HalfPixel` is bit-equivalent to the host loop, and
//!   shipping a convenience twin next to a kernel is the trap. [`host`] carries
//!   exactly one host resampler, for the NPU paths that have no device handle at
//!   all; its module header states the case. Three copies became that one.

//! * **A hard threshold on the host.** [`mask::threshold`] is a device
//!   composition; there is no host twin to drift from it.

pub mod codec;
pub mod color;
pub mod device;
pub mod host;
pub mod letterbox;
pub mod mask;
pub mod pixels;
pub mod tiling;

pub use codec::{decode, load, save_ppm};
pub use color::{Normalization, IMAGENET_MEAN, IMAGENET_STD};
pub use device::{AlignCorners, Border, Ctx, Filter, ImagingKernelIds, PIPELINES};
pub use host::resize_bilinear_hwc;
pub use letterbox::{letterbox_rgb, Letterbox};
pub use pixels::{ChannelPolicy, Rect, Rgb8};
pub use tiling::{Tile, TilePlan, TileSpec};

/// The workspace's NCHW shape type, re-exported so callers need not also depend
/// on `brain-vision` to name an image's dimensions.
pub use vision::Shape;
